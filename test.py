# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../' )
import random
import shutil
import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from data.realestate10k_dataio import RealEstate10kVis
from data.acid_dataio import ACIDVis
import torch
import models
import configargparse
from torch.utils.data import DataLoader
from summary.summaries import *
from utils_training import utils
import config
from tqdm import tqdm
from imageio import imwrite, get_writer
from glob import glob
import time
import matplotlib.pyplot as plt
from models import CoPoNeRF
import lpips
from skimage.metrics import structural_similarity
import cv2
from torch.utils.tensorboard import SummaryWriter
# torch.manual_seed(0)


def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device))*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
   
    
    return theta
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--logging_root', type=str, default=config.logging_root)
p.add_argument('--data_root', type=str, default='./', required=False)
p.add_argument('--val_root', type=str, default=None, required=False)
p.add_argument('--network', type=str, default='relu')
p.add_argument('--category', type=str, default='donut')
p.add_argument('--conditioning', type=str, default='hyper')
p.add_argument('--experiment_name', type=str, required=True)
p.add_argument('--num_context', type=int, default=0)
p.add_argument('--batch_size', type=int, default=48)
p.add_argument('--max_num_instances', type=int, default=None)
p.add_argument('--num_trgt', type=int, default=1)
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--views', type=int, default=2)
p.add_argument('--n_skip', type=int, default=50)

# General training options
p.add_argument('--lr', type=float, default=5e-4)
p.add_argument('--num_epochs', type=int, default=40001)
p.add_argument('--reconstruct', action='store_true', default=False)
p.add_argument('--local', action='store_true', default=False)
p.add_argument('--local_coord', action='store_true', default=False)
p.add_argument('--learned_local_coord', action='store_true', default=False)
p.add_argument('--global_local_coord', action='store_true', default=False)
p.add_argument('--model', type=str, default='resnet')
p.add_argument('--autodecoder', action='store_true', default=False)
p.add_argument('--epochs_til_ckpt', type=int, default=10)
p.add_argument('--steps_til_summary', type=int, default=500)
p.add_argument('--iters_til_ckpt', type=int, default=10000)
p.add_argument('--checkpoint_path', default=None)

# Ablations
p.add_argument('--no_multiview', action='store_true', default=False)
p.add_argument('--no_sample', action='store_true', default=False)
p.add_argument('--no_latent_concat', action='store_true', default=False)
p.add_argument('--no_data_aug', action='store_true', default=False)

opt = p.parse_args()

img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device))

def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))


def make_circle(n, radius=0.1):
    angles = np.linspace(0, 4 * np.pi, n)
    x = np.cos(angles) * radius
    y = np.sin(angles) * radius

    coord = np.stack([x, y, np.zeros(n)], axis=-1)
    return coord


def multigpu_train(gpu, opt):
    if opt.gpus > 1:
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:1492', world_size=opt.gpus, rank=gpu)

    torch.cuda.set_device(gpu)
  

    summaries_dir = os.path.join(opt.logging_root, 'summaries')
    utils.cond_mkdir(summaries_dir)


    writer = SummaryWriter(summaries_dir, flush_secs=10)


    val_dataset = RealEstate10kVis(img_root="/workspace/PF-GeNeRF/temp/realestate/test",
                                      pose_root="/workspace/PF-GeNeRF/poses/realestate/test.mat",
                                      overlap="assets/overlap/realestate.npy",
                                 num_ctxt_views=opt.views, num_query_views=3, augment=True, n_skip=opt.n_skip)

    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, drop_last=True, num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn)

    model = CoPoNeRF.CoPoNeRF( n_view=opt.views)
    old_state_dict = model.state_dict()

    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))
        if opt.reconstruct:
            state_dict['latent_codes.weight'] = torch.zeros_like(state_dict['latent_codes.weight'])

        # state_dict['encoder.latent'] = old_state_dict['encoder.latent']

        model.load_state_dict(state_dict['model'], strict=False)

    model = model.cuda().eval()
    device = "gpu"

    with torch.no_grad():
        loss_fn_alex = lpips.LPIPS(net='vgg').cuda()

        mses = []
        psnrs = []
        lpips_list = []
        ssims = []
        rot =[]
        trans = []
        
        
        
        metrics = {k: {"mse" : [], "psnr" : [], "lpips" : [], "ssim" : [], "rot" : [],  "trans" : [], "angle_trans" : []} for k in ["all", "small", "medium", "large"]}
        for val_i, (model_input, gt, overlap) in enumerate(val_loader):
            print("{}/{} done.".format(val_i,len(val_loader)))
            if device == 'gpu':
                model_input = utils.dict_to_gpu(model_input)
                gt = utils.dict_to_gpu(gt)

        
            rgb_full = model_input['query']['rgb']
            uv_full = model_input['query']['uv']

        

            z, rel_pose, flow = model.get_z(model_input)

            if opt.views == 3:
                rgb_chunks = torch.chunk(rgb_full, 18, dim=2)
                uv_chunks = torch.chunk(uv_full, 18, dim=2)
            else:
                rgb_chunks = torch.chunk(rgb_full, 18, dim=2)
                uv_chunks = torch.chunk(uv_full, 18, dim=2)

            start = time.time()

            model_outputs = []
            
            for rgb_chunk, uv_chunk in zip(rgb_chunks, uv_chunks):
                model_input['query']['rgb'] = rgb_chunk
                model_input['query']['uv'] = uv_chunk
                model_output = model(model_input, z=z,rel_pose=rel_pose,val=True, flow=flow)
                del model_output['z']
                del model_output['coords']
                del model_output['at_wts']

                model_output['pixel_val'] = model_output['pixel_val'].cpu()

                model_outputs.append(model_output)
            model_output_copy = model_output
            model_output_full = {}
            
            for k in model_outputs[0].keys():
                
                if k =='rel_pose' or k =='gt_rel_pose' or k =='flow' or k == 'cyclic_consistency_error':
                    continue
                outputs = [model_output[k] for model_output in model_outputs]
                if k == "pixel_val":
                    val = torch.cat(outputs, dim=-3)
                elif k == 'mask_c2' or k=='matchability_cycle_mask':
                    val = torch.cat(outputs, dim=-1)
                else:
                    val = torch.cat(outputs, dim=-2)
                
                model_output_full[k] = val
            
            model_output = model_output_full
            model_output['rel_pose'] = model_output_copy['rel_pose']
            model_output['gt_rel_pose'] = model_output_copy['gt_rel_pose']
            model_output['flow'] = model_output_copy['flow']
            
            
            model_input['query']['rgb'] = rgb_full
         
            rgb = model_output_full['rgb'].view(2, 256, 256, 3)
           
           
            # Saving output image
            target =  gt['rgb'].squeeze(1).view(2, 256, 256, 3)
            rgb = torch.clamp(rgb, -1, 1)
            rgb = ((rgb + 1) * 0.5).detach() 
            target = ((target + 1) * 0.5).detach() 
         
            
            rot_distance_degrees =  compute_geodesic_distance_from_two_matrices(model_output['rel_pose'][:,:3,:3], model_output['gt_rel_pose'][:,:3,:3])
            rot.append(rot_distance_degrees)
            
            translation = torch.linalg.norm(model_output['rel_pose'][:,:3,3] - model_output['gt_rel_pose'][:,:3,3], dim=-1)
            trans.append(translation)
            
            norm_pred = model_output["rel_pose"][:,:3,3] / torch.linalg.norm(model_output["rel_pose"][:,:3,3], dim = -1).unsqueeze(-1)
            norm_gt =  model_output["gt_rel_pose"][:,:3,3] / torch.linalg.norm(model_output["gt_rel_pose"][:,:3,3], dim =-1).unsqueeze(-1)
            cosine_similarity_0 = torch.dot(norm_pred[0], norm_gt[0])
            cosine_similarity_1 = torch.dot(norm_pred[1], norm_gt[1])
            angle_radians_0 = torch.arccos(torch.clip(cosine_similarity_0, -1.0,1.0))
            angle_radians_1 = torch.arccos(torch.clip(cosine_similarity_1, -1.0,1.0))


            mse = img2mse(rgb, target)
            mse1 = img2mse(rgb[0], target[0])
            mse2 = img2mse(rgb[1], target[1])
            
            psnr1 = mse2psnr(mse1)
            psnr2 = mse2psnr(mse2)
            print("psnr1, psnr2", psnr1, psnr2) 
            psnr = mse2psnr(mse)
            
            mses.append(mse.item())
            psnrs.append(psnr.item())

            rgb_lpips = ((rgb.permute(0,3, 1, 2) - 0.5) * 2).cuda()
            target_lpips = ((target.permute(0,3,1,2) - 0.5) * 2).cuda()
            
            lpip1 = loss_fn_alex(rgb_lpips[0], target_lpips[0]).item()
            lpip2 = loss_fn_alex(rgb_lpips[1], target_lpips[1]).item()
            lpips_list.append((lpip1+lpip2)/2)

            rgb_np = rgb.detach().cpu().numpy()
            target_np = target.detach().cpu().numpy()
            ssim1 = structural_similarity(rgb_np[0], target_np[0], win_size=11, multichannel=True, gaussian_weights=True, channel_axis=-1,data_range = 1)
            ssim2 = structural_similarity(rgb_np[1], target_np[1], win_size=11, multichannel=True, gaussian_weights=True, channel_axis=-1,data_range = 1)
            ssims.append((ssim1+ssim2)/2)
            img_summaries(model, model_input, gt, None, model_output, writer, val_i, 'val_', img_shape=(model.H, model.W), n_view=opt.views)
            key1 = "large" if overlap[0] > 0.75 else ("medium" if overlap[0] >= 0.5 else "small")
            key2 = "large" if overlap[1] > 0.75 else ("medium" if overlap[1] >= 0.5 else "small")

            metrics["all"]["mse"].append(mse.item())
            metrics["all"]["psnr"].append(psnr.item())
            metrics["all"]["lpips"].append((lpip1+lpip2)/2)
            metrics["all"]["ssim"].append((ssim1+ssim2)/2)
            metrics["all"]["rot"].append(rot_distance_degrees)
            metrics["all"]["trans"].append(translation)
            metrics["all"]["angle_trans"].append((angle_radians_0+angle_radians_1)/2)

            metrics[key1]["mse"].append(mse1.item())
            metrics[key1]["psnr"].append(psnr1.item())
            metrics[key1]["lpips"].append(lpip1)
            metrics[key1]["ssim"].append(ssim1)
            metrics[key1]["rot"].append(rot_distance_degrees[0])
            metrics[key1]["trans"].append(translation[0])
            metrics[key1]["angle_trans"].append(angle_radians_0)

            metrics[key2]["mse"].append(mse2.item())
            metrics[key2]["psnr"].append(psnr2.item())
            metrics[key2]["lpips"].append(lpip2)
            metrics[key2]["ssim"].append(ssim2)
            metrics[key2]["rot"].append(rot_distance_degrees[1])
            metrics[key2]["trans"].append(translation[1])
            metrics[key2]["angle_trans"].append(angle_radians_1)
            
            for key in ["all", "small", "medium", "large"]:
                try:
                    print(f"{key}: PSNR: {np.mean(metrics[key]['psnr']):.4f}, SSIM: {np.mean(metrics[key]['ssim']):.4f},LPIPS: {np.mean(metrics[key]['lpips']):.4f}, MSE: {np.mean(metrics[key]['mse']):.4f},Rot_avg: {torch.mean(torch.stack(metrics[key]['rot'])):.4f}, Rot_median: {torch.median(torch.stack(metrics[key]['rot'])):.4f},Rot_std: {torch.std(torch.stack(metrics[key]['rot'])):.4f},Trans_avg: {torch.mean(torch.stack(metrics[key]['trans'])):.4f},Trans_median: {torch.median(torch.stack(metrics[key]['trans'])):.4f},Trans_std: {torch.std(torch.stack(metrics[key]['trans'])):.4f},  Avg_Trans_angle: {torch.mean(torch.stack(metrics[key]['angle_trans'])):.4f},Med_Trans_angle: {torch.median(torch.stack(metrics[key]['angle_trans'])):.4f},std_Trans_angle: {torch.std(torch.stack(metrics[key]['angle_trans'])):.4f} ")
                except:
                    continue
        import pdb
        pdb.set_trace()
        print("here")


if __name__ == "__main__":
    # manager = Manager()
    # shared_dict = manager.dict()

    opt = p.parse_args()
    if opt.gpus > 1:
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt,))
    else:
        multigpu_train(0, opt)
