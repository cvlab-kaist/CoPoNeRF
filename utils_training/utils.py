from packaging import version
import torch
import numpy as np
import matplotlib.colors as colors
import torch.nn.functional as F
import torch.nn as nn
import os, struct, math
import numpy as np
import functools
import cv2
from typing import Any, List, Union, Tuple
from glob import glob
import collections
from copy import deepcopy


def combine_interleaved(t, inner_dims=(1,), agg_type="average"):
    if len(inner_dims) == 1 and inner_dims[0] == 1:
        return t
    t = t.reshape(-1, *inner_dims, *t.shape[1:])
    if agg_type == "average":
        t = torch.mean(t, dim=1)
    elif agg_type == "max":
        t = torch.max(t, dim=1)[0]
    else:
        raise NotImplementedError("Unsupported combine type " + agg_type)
    return t

def get_norm_layer(norm_type="instance", group_norm_groups=32):
    """Return a normalization layer
    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none
    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == "batch":
        norm_layer = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
    elif norm_type == "instance":
        norm_layer = functools.partial(
            nn.InstanceNorm2d, affine=False, track_running_stats=False
        )
    elif norm_type == "group":
        norm_layer = functools.partial(nn.GroupNorm, group_norm_groups)
    elif norm_type == "none":
        norm_layer = None
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer

def flow2kps(trg_kps, flow, n_pts, upsample_size=(256, 256)):
    _, _, h, w = flow.size()

    flow = F.interpolate(flow, upsample_size, mode='bilinear') * (upsample_size[0] / h)
    
    src_kps = []
    mask_list = []
    for trg_kps, flow in zip(trg_kps.long(), flow):
        size = trg_kps.size(0)
        mask_list.append(((0<=trg_kps) & (trg_kps<256))[:,0] & ((0<=trg_kps) & (trg_kps<256))[:,1])
        kp = torch.clamp(trg_kps.transpose(0,1).narrow_copy(1, 0, n_pts), 0, upsample_size[0]-1)
        estimated_kps = kp + flow[:, kp[1, :], kp[0, :]]
        
      
     
        src_kps.append(estimated_kps)

    return torch.stack(src_kps),  torch.stack(mask_list)


def to_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Convert N-dimensional points to homogeneous coordinates.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N).
    Returns:
        A torch.Tensor or numpy.ndarray with size (..., N+1).
    """
    if isinstance(points, torch.Tensor):
        pad = points.new_ones(points.shape[:-1]+(1,))
        return torch.cat([points, pad], dim=-1)
    elif isinstance(points, np.ndarray):
        pad = np.ones((points.shape[:-1]+(1,)), dtype=points.dtype)
        return np.concatenate([points, pad], axis=-1)
    else:
        raise ValueError

def from_homogeneous(points: Union[torch.Tensor, np.ndarray]):
    """Remove the homogeneous dimension of N-dimensional points.
    Args:
        points: torch.Tensor or numpy.ndarray with size (..., N+1).
    Returns:
        A torch.Tensor or numpy ndarray with size (..., N).
    """
    return points[..., :-1] / (points[..., -1:] + 1e-6)



def encode_relative_point(ray, transform):
    s = ray.size()
    b, ncontext = transform.size()[:2]

    ray = ray.view(b, ncontext, *s[1:])
    ray = torch.cat([ray, torch.ones_like(ray[..., :1])], dim=-1)
    ray = (ray[:, :, :, :, None, :] * transform[:, :, None, None, :4, :4]).sum(dim=-1)[..., :3]

    ray = ray.view(*s)
    return ray


def pose_inverse_4x4(mat: torch.Tensor, use_inverse: bool=False) -> torch.Tensor:
    """
    Transforms world2cam into cam2world or vice-versa, without computing the inverse.
    Args:
        mat (torch.Tensor): pose matrix (B, 4, 4) or (4, 4)
    """
    # invert a camera pose
    out_mat = torch.zeros_like(mat)

    if len(out_mat.shape) == 3:
        # must be (B, 4, 4)
        out_mat[:, 3, 3] = 1
        R,t = mat[:, :3, :3],mat[:,:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]

        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [...,3,4]

        out_mat[:, :3] = pose_inv
    else:
        out_mat[3, 3] = 1
        R,t = mat[:3, :3], mat[:3, 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1,-2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = torch.cat([R_inv,t_inv[...,None]],dim=-1) # [3,4]
        out_mat[:3] = pose_inv
    # assert torch.equal(out_mat, torch.inverse(mat))
    return out_mat

def batch_project_to_other_img(kpi: torch.Tensor, di: torch.Tensor, 
                               Ki: torch.Tensor, Kj: torch.Tensor, 
                               T_itoj: torch.Tensor, 
                               return_depth=False) -> torch.Tensor:
    """
    Project pixels of one image to the other. 
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        return_depth: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        di_j: Depth of the projections in image j, BxN
    """
    if len(di.shape) == len(kpi.shape):
        # di must be BxNx1
        di = di.squeeze(-1)
    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i * di[..., None]  # non-homogeneous coordinates

    kpi_3d_j = from_homogeneous(
            to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2))
    if return_depth:
        di_j = kpi_3d_j[..., -1]
        return kpi_j, di_j
    return kpi_j

def batch_sample_project_to_other_img(kpi: torch.Tensor, di: torch.Tensor, 
                               Ki: torch.Tensor, Kj: torch.Tensor, 
                               T_itoj: torch.Tensor, 
                               return_depth=False) -> torch.Tensor:
    """
    Project pixels of one image to the other. 
    Args:
        kpi: BxNx2 coordinates in pixels of image i
        di: BxN, corresponding depths of image i
        Ki: intrinsics of image i, Bx3x3
        Kj: intrinsics of image j, Bx3x3
        T_itoj: Transform matrix from coordinate system of i to j, Bx4x4
        return_depth: Bool

    Returns:
        kpi_j: Pixels projection in image j, BxNx2
        di_j: Depth of the projections in image j, BxN
    """
    
    if len(di.shape) == len(kpi.shape):
        # di must be BxNx1
        di = di.squeeze(-1)
   
    kpi_3d_i = to_homogeneous(kpi) @ torch.inverse(Ki).transpose(-1, -2)
    kpi_3d_i = kpi_3d_i[:,:,None] *  di[..., None] # non-homogeneous coordinates
    kpi_3d_j = from_homogeneous(to_homogeneous(kpi_3d_i) @ T_itoj.transpose(-1, -2).unsqueeze(1))
    kpi_j = from_homogeneous(kpi_3d_j @ Kj.transpose(-1, -2).unsqueeze(1))
    if return_depth:
        di_j = kpi_3d_j[..., -1]
        return kpi_j, di_j
    return kpi_j


class MultiLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lambda_factories, last_epoch=-1, verbose=False):
        self.schedulers = []
        values = self._get_optimizer_lr(optimizer)
        for idx, factory in enumerate(lambda_factories):
            self.schedulers.append(factory(optimizer))
            values[idx] = self._get_optimizer_lr(optimizer)[idx]
            self._set_optimizer_lr(optimizer, values)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        result = []
        for idx, sched in enumerate(self.schedulers):
            result.append(sched.get_last_lr()[idx])
        return result
    
    @staticmethod
    def _set_optimizer_lr(optimizer, values):
        for param_group, lr in zip(optimizer.param_groups, values):
            param_group['lr'] = lr

    @staticmethod
    def _get_optimizer_lr(optimizer):
        return [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch=None):
        if self.last_epoch != -1:
            values = self._get_optimizer_lr(self.optimizer)
            for idx, sched in enumerate(self.schedulers):
                sched.step()
                values[idx] = self._get_optimizer_lr(self.optimizer)[idx]
                self._set_optimizer_lr(self.optimizer, values)
        super().step()

def parse_comma_separated_integers(string):
    return list(map(int, string.split(',')))

def normalize_for_grid_sample(pixel_coords, H, W):
    pixel_coords[..., 0] = (pixel_coords[..., 0] / (W - 1)) * 2 - 1
    pixel_coords[..., 1] = (pixel_coords[..., 1] / (H - 1)) * 2 - 1
    return pixel_coords

def normalize_imagenet(x):
    ''' Normalize input images according to ImageNet standards.

    Args:
        x (tensor): input images
    '''
    x = x.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


def generate_mask_from_confidence_score(C2_points, confidence_score, n_pts, threshold = 0.8, upsample_size=(256, 256)):
    
    
   
    matchability = []
    for trg_kps, conf in zip(C2_points.long(), confidence_score):
        size = trg_kps.size(0)
        
        kp = torch.clamp(trg_kps.transpose(0,1).narrow_copy(1, 0, n_pts), 0, upsample_size[0]-1)
        ctxt2_conf = conf[kp[1], kp[0]] # confidence score at context 2
     

        matchability.append(ctxt2_conf)

     
   
    return  torch.stack(matchability)
  

def generate_mask_from_confidence_score_sample(C2_points, confidence_score, n_pts, threshold = 0.8, upsample_size=(256, 256)):

    matchability = []
    for trg_kps, conf in zip(C2_points.long(), confidence_score):
        size = trg_kps.size(0)
        kp = torch.clamp(trg_kps.transpose(0,1).narrow_copy(1, 0, n_pts), 0, upsample_size[0]-1)
        
        ctxt2_conf = conf[kp[:, :, 1], kp[:, :, 0]]

        
       
        matchability.append(ctxt2_conf)

    return torch.stack(matchability)

def convert_image(img, type):
    '''Expects single batch dimesion'''
    img = img.squeeze(0)

    if not 'normal' in type:
        img = detach_all(lin2img(img, mode='np'))

    if 'rgb' in type or 'normal' in type:
        img += 1.
        img /= 2.
    elif type == 'depth':
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img *= 255.
    img = np.clip(img, 0., 255.).astype(np.uint8)
    return img


def flatten_first_two(tensor):
    b, s, *rest = tensor.shape
    return tensor.view(b * s, *rest)


def parse_intrinsics(filepath, trgt_sidelength=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    if trgt_sidelength is not None:
        cx = cx / width * trgt_sidelength
        cy = cy / height * trgt_sidelength
        f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, world2cam_poses


def num_divisible_by_2(number):
    i = 0
    while not number % 2:
        number = number // 2
        i += 1

    return i


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def normalize(img):
    return (img - img.min()) / (img.max() - img.min())


def print_network(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("%d" % params)


def add_batch_dim_to_dict(ob):
    if isinstance(ob, collections.Mapping):
        return {k: add_batch_dim_to_dict(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(add_batch_dim_to_dict(k) for k in ob)
    elif isinstance(ob, list):
        return [add_batch_dim_to_dict(k) for k in ob]
    else:
        try:
            return ob[None, ...]
        except:
            return ob


def detach_all(tensor):
    return tensor.detach().cpu().numpy()


def lin2img(tensor, image_resolution=None, mode='torch'):
    if len(tensor.shape) == 3:
        batch_size, num_samples, channels = tensor.shape
    elif len(tensor.shape) == 2:
        num_samples, channels = tensor.shape

    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    if len(tensor.shape) == 3:
        if mode == 'torch':
            tensor = tensor.permute(0, 2, 1).view(batch_size, channels, height, width)
        elif mode == 'np':
            tensor = tensor.view(batch_size, height, width, channels)
    elif len(tensor.shape) == 2:
        if mode == 'torch':
            tensor = tensor.permute(1, 0).view(channels, height, width)
        elif mode == 'np':
            tensor = tensor.view(height, width, channels)

    return tensor


def pick(list, item_idcs):
    if not list:
        return list
    return [list[i] for i in item_idcs]


def get_mgrid(sidelen, dim=2, flatten=False):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.from_numpy(pixel_coords)

    if flatten:
        pixel_coords = pixel_coords.view(-1, dim)
    return pixel_coords


def dict_to_gpu(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(dict_to_gpu(k) for k in ob)
    elif isinstance(ob, list):
        return [dict_to_gpu(k) for k in ob]
    else:
        try:
            return ob.cuda()
        except:
            return ob

def dict_to_gpu_expand(ob):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_gpu_expand(v) for k, v in ob.items()}
    elif isinstance(ob, tuple):
        return tuple(dict_to_gpu_expand(k) for k in ob)
    elif isinstance(ob, list):
        return [dict_to_gpu_expand(k) for k in ob]
    else:
        try:
            return ob[None].cuda()
        except:
            return ob


def assemble_model_input(context, query, gpu=True):
    context = deepcopy(context)
    query = deepcopy(query)
    context['mask'] = torch.Tensor([1.])
    query['mask'] = torch.Tensor([1.])

    context = add_batch_dim_to_dict(context)
    context = add_batch_dim_to_dict(context)

    query = add_batch_dim_to_dict(query)
    query = add_batch_dim_to_dict(query)

    context['rgb'] = context['rgb_other']
    context['cam2world'] = context['cam2world_other']

    model_input = {'context': context, 'query': query, 'post_input': query}

    if gpu:
        model_input = dict_to_gpu(model_input)
    return model_input
   
def convert_flow_to_mapping(flow, output_channel_first=True):
    if not isinstance(flow, np.ndarray):
        # torch tensor
        if len(flow.shape) == 4:
            if flow.shape[1] != 2:
                # size is BxHxWx2
                flow = flow.permute(0, 3, 1, 2)

            B, C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
            yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
            grid = torch.cat((xx, yy), 1).float()

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(0,2,3,1)
        else:
            if flow.shape[0] != 2:
                # size is HxWx2
                flow = flow.permute(2, 0, 1)

            C, H, W = flow.size()

            xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
            yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
            xx = xx.view(1, H, W)
            yy = yy.view(1, H, W)
            grid = torch.cat((xx, yy), 0).float() # attention, concat axis=0 here

            if flow.is_cuda:
                grid = grid.cuda()
            mapping = flow + grid # here also channel first
            if not output_channel_first:
                mapping = mapping.permute(1,2,0).float()
        return mapping.float()
    else:
        # here numpy arrays
        if len(flow.shape) == 4:
            if flow.shape[3] != 2:
                # size is Bx2xHxW
                flow = flow.transpose(0, 2, 3, 1)
            # BxHxWx2
            b, h_scale, w_scale = flow.shape[:3]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))
            for i in range(b):
                mapping[i, :, :, 0] = flow[i, :, :, 0] + X
                mapping[i, :, :, 1] = flow[i, :, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(0,3,1,2)
        else:
            if flow.shape[0] == 2:
                # size is 2xHxW
                flow = flow.transpose(1,2,0)
            # HxWx2
            h_scale, w_scale = flow.shape[:2]
            mapping = np.copy(flow)
            X, Y = np.meshgrid(np.linspace(0, w_scale - 1, w_scale),
                               np.linspace(0, h_scale - 1, h_scale))

            mapping[:, :, 0] = flow[:, :, 0] + X
            mapping[:, :, 1] = flow[:, :, 1] + Y
            if output_channel_first:
                mapping = mapping.transpose(2, 0, 1)
        return mapping.astype(np.float32)


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
    
def huber_loss(pred: torch.Tensor, label: torch.Tensor, reduction: str='none'):
        return torch.nn.functional.huber_loss(pred, label, reduction=reduction) 

class MultiLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lambda_factories, last_epoch=-1, verbose=False):
        self.schedulers = []
        values = self._get_optimizer_lr(optimizer)
        for idx, factory in enumerate(lambda_factories):
            self.schedulers.append(factory(optimizer))
            values[idx] = self._get_optimizer_lr(optimizer)[idx]
            self._set_optimizer_lr(optimizer, values)
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        result = []
        for idx, sched in enumerate(self.schedulers):
            result.append(sched.get_last_lr()[idx])
        return result
    
    @staticmethod
    def _set_optimizer_lr(optimizer, values):
        for param_group, lr in zip(optimizer.param_groups, values):
            param_group['lr'] = lr

    @staticmethod
    def _get_optimizer_lr(optimizer):
        return [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch=None):
        if self.last_epoch != -1:
            values = self._get_optimizer_lr(self.optimizer)
            for idx, sched in enumerate(self.schedulers):
                sched.step()
                values[idx] = self._get_optimizer_lr(self.optimizer)[idx]
                self._set_optimizer_lr(self.optimizer, values)
        super().step()


def warp(x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        Args:
            x: [B, C, H, W] (im2)
            flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy),1).float()

        if x.is_cuda:
            grid = grid.to(flo.device)
        vgrid = grid + flo
        # makes a mapping out of the flow

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W-1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H-1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)

        output = nn.functional.grid_sample(x, vgrid)
        return output