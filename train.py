# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
import random

import torch
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from data.realestate10k_dataio import RealEstate10k
from data.acid_dataio import ACID
import torch
import models
import wrapper
import configargparse
from torch.utils.data import DataLoader
import models.loss_function as loss_functions
from models import CoPoNeRF
from summary.summaries import *
from assets import config
from utils_training.utils import MultiLR

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True)

p.add_argument('--logging_root', type=str, default=config.logging_root)
p.add_argument('--data_root', type=str, default='/media/temp/realestate', required=False)
p.add_argument('--val_root', type=str, default=None, required=False)
p.add_argument('--network', type=str, default='relu')
p.add_argument('--category', type=str, default='donut')
p.add_argument('--conditioning', type=str, default='hyper')
p.add_argument('--experiment_name', type=str, required=True)
p.add_argument('--num_context', type=int, default=0)
p.add_argument('--batch_size', type=int, default=12)
p.add_argument('--num_trgt', type=int, default=1)
p.add_argument('--views', type=int, default=2)
p.add_argument('--gpus', type=int, default=1)
p.add_argument('--n_skip', type=int, default=50)
# General training options
p.add_argument('--lr', type=float, default=5e-5  * 4)
p.add_argument('--l2_coeff', type=float, default=0.05)
p.add_argument('--num_epochs', type=int, default=40001)
p.add_argument('--cycle', action='store_true', default=False)
p.add_argument('--pose', action='store_true', default=False)
p.add_argument('--ssim', action='store_true', default=False)
p.add_argument('--depth', action='store_true', default=False)
p.add_argument('--model', type=str, default='resnet')
p.add_argument('--epochs_til_ckpt', type=int, default=100)
p.add_argument('--steps_til_summary', type=int, default=500)
p.add_argument('--iters_til_ckpt', type=int, default=10000)
p.add_argument('--checkpoint_path', default=None)


opt = p.parse_args()


def sync_model(model):
    for param in model.parameters():
        dist.broadcast(param.data, 0)

def worker_init_fn(worker_id):
    random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed)%(2**32-1))

def split_params(model):
    encoder_param = []
    decoder_param = []

    for name, param in model.named_parameters():
        if 'regressor' or 'cross_attention' or 'pre_pose_mlp' or 'feature_cost_aggregation' in name:
            decoder_param.append(param)
        else:
            encoder_param.append(param)

    return encoder_param, decoder_param

def multigpu_train(gpu, opt):
    if opt.gpus > 1:

        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:6010', world_size=opt.gpus, rank=gpu)

    torch.cuda.set_device(gpu)
    def create_dataloader_callback(sidelength, batch_size, query_sparsity):
        train_dataset = ACID(img_root="/workspace/data2/cross_attention_renderer/acid_full/train",
                                     pose_root="/workspace/data2/cross_attention_renderer/poses/acid/train.mat",
                                     num_ctxt_views=opt.views, num_query_views=1, query_sparsity=192,
                                     augment=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=True, num_workers=8, pin_memory=False, worker_init_fn=worker_init_fn)

        val_dataset = ACID(img_root="/workspace/data2/cross_attention_renderer/acid_full/test",
                                      pose_root="/workspace/data2/cross_attention_renderer/poses/acid/test.mat",
                                    num_ctxt_views=opt.views, num_query_views=1, augment=False)        
        val_loader = DataLoader(val_dataset, batch_size=6, shuffle=True, drop_last=True, num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn)

        return train_loader, val_loader

    model = CoPoNeRF.CoPoNeRF( n_view=opt.views)
   
    encoder_param, decoder_param = split_params(model)
    optimizer = torch.optim.Adam([
        {"params": encoder_param, "lr": opt.lr},
        {"params": decoder_param, "lr": opt.lr},
    ])
   
    scheduler = MultiLR(optimizer, [lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, 0.95), 
                 lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, 0.95)])


    if opt.checkpoint_path is not None:
        print(f"Loading weights from {opt.checkpoint_path}...")
        state_dict = torch.load(opt.checkpoint_path, map_location=torch.device('cpu'))
        state_dict= state_dict['model']

        model.load_state_dict(state_dict, strict=False)
        print("loaded")


    model = model.cuda()

    if opt.gpus > 1:
        sync_model(model)

    # Define the loss
    summary_fn = img_summaries
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    loss_fn = val_loss_fn = loss_functions.LFLoss(opt.l2_coeff, opt.depth, opt.pose, opt.cycle, opt.ssim)

    wrapper.training(model=model, dataloader_callback=create_dataloader_callback,
                                 dataloader_iters=(1000000,), dataloader_params=((64, opt.batch_size, 512), ),
                                 epochs=opt.num_epochs, lr=opt.lr, steps_til_summary=opt.steps_til_summary,
                                 epochs_til_checkpoint=opt.epochs_til_ckpt,
                                 model_dir=root_path, loss_fn=loss_fn, val_loss_fn=val_loss_fn,
                                 iters_til_checkpoint=opt.iters_til_ckpt, val_summary_fn=summary_fn,
                                 overwrite=True,
                                 optimizer=optimizer,
                                 clip_grad=True,
                                 rank=gpu, train_function=wrapper.train, gpus=opt.gpus, n_view=opt.views, scheduler = scheduler)

if __name__ == "__main__":
    opt = p.parse_args()
    if opt.gpus > 1:
   
        mp.spawn(multigpu_train, nprocs=opt.gpus, args=(opt,))
    else:
        multigpu_train(0, opt)