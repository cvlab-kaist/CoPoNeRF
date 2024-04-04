import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

import numpy as np
from math import exp
from lietorch import SE3
import os
import lpips
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils_training.utils import  get_gt_correspondence_mask, huber_loss, warp

''' 
SSIM Loss
'''
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)]).type(torch.cuda.FloatTensor)
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, mask):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return (torch.sum((1 - ssim_map) * mask) / torch.sum(mask) / 3)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.channel = 3
        self.window = create_window(window_size, self.channel)
        self.windowMask = torch.ones(1, 1, self.window_size, self.window_size).cuda() / self.window_size / self.window_size
    def forward(self, img1, img2, match):
        (_, channel, _, _) = img1.size()
   
        
        ## maximize ssim
        
        return _ssim(img1, img2, self.window, self.window_size, self.channel, match)
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Recon Loss
'''
def image_loss(model_out, gt, mask=None):
    gt_rgb = gt['rgb']
    gt_rgb[torch.isnan(gt_rgb)] = 0.0
    rgb = model_out['rgb']
    rgb[torch.isnan(rgb)] = 0.0
    loss = torch.abs(gt_rgb - rgb).mean()
    return loss
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''
Pose Loss
'''
def compute_geodesic_distance_from_two_matrices(m1, m2):
    batch=m1.shape[0]
    m = torch.bmm(m1, m2.transpose(1,2)) #batch*3*3
    
    cos = (  m[:,0,0] + m[:,1,1] + m[:,2,2] - 1 )/2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device)) )
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).to(m1.device))*-1 )
    
    theta = torch.acos(cos)

    return theta.mean()
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class LFLoss():
    def __init__(self, l2_weight=1e-3, depth=False, pose=False, cycle = False, ssim = False, reg_weight=1e2):
       
        self.depth = depth
        self.pose = pose
      
        self.cycle = cycle
        self.ssim = ssim
        self.ssim_loss = SSIM()
        
       
        self.w1 = 0.01
        self.w2 = 1.0
        self.w3 = 1.0
    
      
    def __call__(self,model_input, model_out, gt, ITER, model=None, val=False):
        loss_dict = {}
        loss_dict['img_loss'] =   image_loss(model_out, gt)
        
        if self.ssim:
            _, _, h, w = model_out['flow'][0].size()
      
            flow = F.interpolate( model_out['flow'][0], 256, mode='bilinear') * (256 / h)
            flow2 = F.interpolate( model_out['flow'][1], 256, mode='bilinear') * (256 / h)
      
            cyclic_consistency_error = torch.norm(flow + warp(flow2, flow), dim=1).le(10)
            cyclic_consistency_error2 = torch.norm(flow2+ warp(flow, flow2), dim=1).le(10)
            mask_padded = cyclic_consistency_error * get_gt_correspondence_mask(flow)
            mask_padded2 = cyclic_consistency_error2 * get_gt_correspondence_mask(flow2)
    
            loss_dict['ssim_loss'] = self.w2 * (self.ssim_loss(warp(model_input['context']['rgb'][:,1].permute(0,3,1,2), flow), model_input['context']['rgb'][:,0].permute(0,3,1,2),mask_padded.unsqueeze(1)) +self.ssim_loss(warp(model_input['context']['rgb'][:,0].permute(0,3,1,2), flow2), model_input['context']['rgb'][:,1].permute(0,3,1,2),mask_padded2.unsqueeze(1)))/2
          
        if self.cycle:
            loss = torch.norm(model_out['T_to_C1_pts'] - model_out['C2_pts_to_C1'], dim=-1, keepdim=True)
            valid = torch.ones_like(loss).bool()
            valid_pixel = loss.detach().le(20)
            valid = valid & valid_pixel
            mask_c2 = model_out['mask_c2']
            mask_cycle = model_out['matchability_cycle_mask']
            
            loss_dict['cycle_loss'] = self.w1 * ((huber_loss(model_out['T_to_C1_pts'], model_out['C2_pts_to_C1']) * valid.float() * mask_c2.unsqueeze(-1) * mask_cycle.unsqueeze(-1)).sum() / ((valid.float() * mask_c2.unsqueeze(-1) * mask_cycle.unsqueeze(-1)).sum() +1e-6))

        if self.pose:

            loss_dict['pose_loss'] = self.w3 * ((compute_geodesic_distance_from_two_matrices(model_out['rel_pose'][:,:3,:3], model_out['gt_rel_pose'][:,:3,:3])      )+ torch.norm(model_out['rel_pose'][:,:3,3]- model_out['gt_rel_pose'][:,:3,3],dim=-1).mean() )


        return loss_dict, {}