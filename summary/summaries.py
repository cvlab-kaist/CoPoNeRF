import matplotlib
import torch.nn.functional as F
matplotlib.use('Agg')
import numpy as np
import random
import matplotlib.pyplot as plt
from utils_training.utils import warp, convert_flow_to_mapping
import torch
from utils_training import utils
from packaging import version
import cv2
from torch import nn
import torchvision
from utils_training import geometry
from summary.inspect_epipolar_geometry import * 

def _pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap
def overlay_semantic_mask(im, ann, alpha=0.5, mask=None, colors=None, color=[255, 218, 185], contour_thickness=1):
    """
    example usage:
    image_overlaid = overlay_semantic_mask(im.astype(np.uint8), 255 - mask.astype(np.uint8) * 255, color=[255, 102, 51])
    """
    im, ann = np.asarray(im, dtype=np.uint8), np.asarray(ann, dtype=np.uint8)
    if im.shape[:-1] != ann.shape:
        raise ValueError('First two dimensions of `im` and `ann` must match')
    if im.shape[-1] != 3:
        raise ValueError('im must have three channels at the 3 dimension')

    colors = colors or _pascal_color_map()
    colors = np.asarray(colors, dtype=np.uint8)
    colors[-1, :] = color

    if mask is None:
        mask = colors[ann]

    fg = im * alpha + (1 - alpha) * mask

    img = im.copy()
    img[ann > 0] = fg[ann > 0]  # where the mask is zero (where object is), shoudlnt be any color

    if contour_thickness:  # pragma: no cover
        import cv2
        for obj_id in np.unique(ann[ann > 0]):
            contours = cv2.findContours((ann == obj_id).astype(
                np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
            cv2.drawContours(img, contours[0], -1, color,
                             contour_thickness)
    return img

def get_gt_correspondence_mask(flow):
    """Computes the mask of valid flows (that do not match to a pixel outside of the image). """

    mapping = convert_flow_to_mapping(flow, output_channel_first=True)
    if isinstance(mapping, np.ndarray):
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[:, 0] >= 0, mapping[:, 0] <= w-1)
            mask_y = np.logical_and(mapping[:, 1] >= 0, mapping[:, 1] <= h-1)
            mask = np.logical_and(mask_x, mask_y)
        else:
            _, h, w = mapping.shape
            mask_x = np.logical_and(mapping[0] >= 0, mapping[0] <= w - 1)
            mask_y = np.logical_and(mapping[1] >= 0, mapping[1] <= h - 1)
            mask = np.logical_and(mask_x, mask_y)
        mask = mask.astype(np.bool) if version.parse(torch.__version__) >= version.parse("1.1") else mask.astype(np.uint8)
    else:
        if len(mapping.shape) == 4:
            # shape is B,C,H,W
            b, _, h, w = mapping.shape
            mask = mapping[:, 0].ge(0) & mapping[:, 0].le(w-1) & mapping[:, 1].ge(0) & mapping[:, 1].le(h-1)
        else:
            _, h, w = mapping.shape
            mask = mapping[0].ge(0) & mapping[0].le(w-1) & mapping[1].ge(0) & mapping[1].le(h-1)
        mask = mask.bool() if version.parse(torch.__version__) >= version.parse("1.1") else mask.byte()
    return mask





def img_summaries(model, model_input, ground_truth, loss_summaries, model_output, writer, iter, prefix="", img_shape=(98, 144), n_view=1):
    predictions = model_output['rgb']
  
    predictions = predictions.view(*predictions.size()[:-2], img_shape[0], img_shape[1], 3)
    predictions = utils.flatten_first_two(predictions)
    predictions = predictions.permute(0, 3, 1, 2)
    predictions = torch.clamp(predictions, -1, 1)
   
    if 'at_wt' in model_output:
        at_wt = model_output['at_wt']
        ent = -(at_wt * torch.log(at_wt + 1e-5)).sum(dim=-1)
        ent = ent.mean()
        writer.add_scalar(prefix + "ent", ent, iter)
        
        if torch.isnan(ent):
            breakpoint()

    writer.add_image(prefix + "predictions",
                     torchvision.utils.make_grid(predictions, scale_each=False, normalize=True).cpu().numpy(),
                     iter)

    

    depth_img = model_output['depth_ray'].view(-1, img_shape[0], img_shape[1])
    depth_img = depth_img.detach().cpu().numpy() / 10.
    cmap = plt.get_cmap("jet")
    depth_img = cmap(depth_img)[..., :3]
    depth_img = depth_img.transpose((0, 3, 1, 2))

    depth_img = torch.Tensor(depth_img)
    writer.add_image(prefix + "depth_images",
                     torchvision.utils.make_grid(depth_img, scale_each=True, normalize=True).cpu().numpy(),
                     iter)
    
    
    context_images = utils.flatten_first_two(model_input['context']['rgb'])
    context_images = context_images.permute(0, 3, 1, 2)

    writer.add_image(prefix + "context_images",
                     torchvision.utils.make_grid(context_images, scale_each=False, normalize=True).cpu().numpy(),
                     iter)

    query_images = model_input['query']['rgb']
    query_images = query_images.view(*query_images.size()[:-2], img_shape[0], img_shape[1], 3)

    query_images = utils.flatten_first_two(query_images)
    query_images = query_images.permute(0, 3, 1, 2)
    writer.add_image(prefix + "query_images",
                     torchvision.utils.make_grid(query_images, scale_each=False, normalize=True).cpu().numpy(),
                     iter)

    
    epipolar_pred, epipolar_gt = inspect(model_input['context']['rgb'][:,1], model_input['context']['intrinsics'][:,1], 
            model_input['context']['rgb'][:,0], model_input['context']['intrinsics'][:,0], 
            model_output['rel_pose'],
             model_output['gt_rel_pose'] )
    
    _, _, h, w = model_output['flow'][0].size()
    
    flow = F.interpolate( model_output['flow'][0], 256, mode='bilinear') * (256 / h)
    flow2 = F.interpolate( model_output['flow'][1], 256, mode='bilinear') * (256 / h)
    
    
    cyclic_consistency_error = torch.norm(flow + warp(flow2, flow), dim=1).le(10)
    cyclic_consistency_error2 = torch.norm(flow2+ warp(flow, flow2), dim=1).le(10)
    mask_padded = cyclic_consistency_error * get_gt_correspondence_mask(flow)
    mask_padded2 = cyclic_consistency_error2 * get_gt_correspondence_mask(flow2)
    
    warped_img = []
    warped_img_mask = []
    for i in range(len(flow)):
        temp = warp((model_input['context']['rgb'][i,1].permute(2,0,1).unsqueeze(0)+ 1) * 127.5,flow[i]).squeeze(0).permute(1,2,0)
        warped_img.append(temp)
        warped_img_mask.append(overlay_semantic_mask(temp.cpu().numpy(), 255 - mask_padded[i].cpu().numpy()* 255, color=[255, 102, 51]))
     
    warped_img = torch.stack(warped_img).to(flow.device)

    warped_img_mask = np.stack(warped_img_mask)
    

    warped_img = torch.cat( (warped_img, (model_input['context']['rgb'][:,0]+1) * 127.5) , dim =-2 )
    warped_img = torch.cat(((model_input['context']['rgb'][:,1]+1) * 127.5, warped_img), dim =-2 )
    writer.add_image(prefix + "warped_img",torchvision.utils.make_grid(warped_img.permute(0,3,1,2), scale_each=False, normalize=True).cpu().numpy(),iter)

    writer.add_image(prefix + "masked_warped_img",torchvision.utils.make_grid(torch.from_numpy(warped_img_mask).float().permute(0,3,1,2), scale_each=False, normalize=True).cpu().numpy(),iter)
    warped_img = []
    warped_img_mask = []
    for i in range(len(flow)):
        temp = warp((model_input['context']['rgb'][i,0].permute(2,0,1).unsqueeze(0)+ 1) * 127.5,flow2[i]).squeeze(0).permute(1,2,0)
        warped_img.append(temp)
        warped_img_mask.append(overlay_semantic_mask(temp.cpu().numpy(), 255 - mask_padded2[i].cpu().numpy()* 255, color=[255, 102, 51]))
     
    warped_img = torch.stack(warped_img).to(flow.device)
    
    warped_img_mask = np.stack(warped_img_mask)
    

    
    warped_img = torch.cat( (warped_img, (model_input['context']['rgb'][:,1]+1) * 127.5) , dim =-2 )
    warped_img = torch.cat(((model_input['context']['rgb'][:,0]+1) * 127.5, warped_img), dim =-2 )
    writer.add_image(prefix + "warped_img_flip",torchvision.utils.make_grid(warped_img.permute(0,3,1,2), scale_each=False, normalize=True).cpu().numpy(),iter)
    writer.add_image(prefix + "masked_warped_img_flip",torchvision.utils.make_grid(torch.from_numpy(warped_img_mask).float().permute(0,3,1,2), scale_each=False, normalize=True).cpu().numpy(),iter)
    
    
    
    
    
    if epipolar_pred is not None or epipolar_gt is not None:

        writer.add_image(prefix + "epipolar_GT",torchvision.utils.make_grid(torch.from_numpy(epipolar_gt).permute(0,3,1,2), scale_each=False, normalize=True).cpu().numpy(),iter)
        
        writer.add_image(prefix + "epipolar_pred",
                        torchvision.utils.make_grid(torch.from_numpy(epipolar_pred).permute(0,3,1,2), scale_each=False, normalize=True).cpu().numpy(),
                        iter)
  

    writer.add_scalar(prefix + "flow_mean", flow.flatten(-2,-1).mean(-1)[0,0], iter)
    writer.add_scalar(prefix + "out_min", predictions.min(), iter)
    writer.add_scalar(prefix + "out_max", predictions.max(), iter)
    writer.add_scalar(prefix + "rot_distance", compute_geodesic_distance_from_two_matrices(model_output['rel_pose'][:,:3,:3], model_output['gt_rel_pose'][:,:3,:3]), iter)
   
    writer.add_scalar(prefix + "rot_distance_degrees_mean", compute_geodesic_distance_from_two_matrices_degree(model_output['rel_pose'][:,:3,:3], model_output['gt_rel_pose'][:,:3,:3]).mean(), iter)
    writer.add_scalar(prefix + "rot_distance_degrees_std", compute_geodesic_distance_from_two_matrices_degree(model_output['rel_pose'][:,:3,:3], model_output['gt_rel_pose'][:,:3,:3]).std(), iter)
    writer.add_scalar(prefix + "rot_distance_degrees_max", compute_geodesic_distance_from_two_matrices_degree(model_output['rel_pose'][:,:3,:3], model_output['gt_rel_pose'][:,:3,:3]).max(), iter)
    l2loss = nn.MSELoss()

    writer.add_scalar(prefix + "tran_L1", l2loss(model_output['rel_pose'][:,:3,3] , model_output['gt_rel_pose'][:,:3,3]), iter)
    
    writer.add_scalar(prefix + "trgt_min", query_images.min(), iter)
    writer.add_scalar(prefix + "trgt_max", query_images.max(), iter)

def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device))*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
   
    
    return theta.mean()
    
def compute_geodesic_distance_from_two_matrices_degree(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device))*-1 )
    
    
    theta = torch.acos(cos)
    
    #theta = torch.min(theta, 2*np.pi - theta)
   
    
    return theta/np.pi*180
