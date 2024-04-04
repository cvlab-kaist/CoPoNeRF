import torch.nn.functional as F
import time
import torch
import torch.nn as nn
import numpy as np
from models.aggregation import UFC
from models.epipolar import project_rays
from models.lightfield import ResnetFC
import timm
from models.backbone import CrossBlock, SpatialEncoder
from copy import deepcopy
import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils_training.utils import *
from utils_training import geometry


class CoPoNeRF(nn.Module):
    def __init__(self,  n_view=1, npoints=64, num_hidden_units_phi=128):
        super().__init__()

        self.n_view = n_view
        self.npoints = 64

        if npoints:
            self.npoints = npoints
        self.no_sample = False
        self.repeat_attention = True
        # ---------------------------------------------------------------
        # pose estimation

        self.cross_attention = CrossBlock()
        self.pose_regressor = nn.Sequential(

            nn.Linear((16*16+6) *256 * 2, 512 ),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128 * 2),
            nn.ReLU(),
            
        )
        self.rotation_regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 6),
        )
        self.translation_regressor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        # ---------------------------------------------------------------
        # Feature and cost aggregation

        self.feature_cost_aggregation = UFC()
        
        # ---------------------------------------------------------------

        self.encoder = SpatialEncoder(use_first_pool=False, num_layers=5)
        self.latent_dim = 256*3 + 64
        self.conv_map = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3)
    
        self.query_encode_latent = nn.Conv2d(self.latent_dim +3  , self.latent_dim, 1)
        self.query_encode_latent_2 = nn.Conv2d(self.latent_dim, self.latent_dim // 2 , 1)
        self.corr_embed = nn.Conv2d(4096, self.latent_dim  , 1)
        self.latent_dim = self.latent_dim // 2

        self.num_hidden_units_phi = num_hidden_units_phi

        hidden_dim = 128

        self.latent_value = nn.Conv2d(self.latent_dim * self.n_view , self.latent_dim, 1)
        self.key_map = nn.Conv2d(self.latent_dim * self.n_view ,hidden_dim, 1)
        self.key_map_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
       

        self.query_embed = nn.Conv2d(16, hidden_dim, 1)
        self.query_embed_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.hidden_dim = hidden_dim

        self.latent_avg_query = nn.Conv2d(9+16, hidden_dim, 1)
        self.latent_avg_query_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.latent_avg_key = nn.Conv2d(self.latent_dim, hidden_dim, 1)
        self.latent_avg_key_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.query_repeat_embed = nn.Conv2d(16+128, hidden_dim, 1)
        self.query_repeat_embed_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.latent_avg_repeat_query = nn.Conv2d(9+16+128, hidden_dim, 1)
        self.latent_avg_repeat_query_2 = nn.Conv2d(hidden_dim, hidden_dim, 1)

        self.encode_latent = nn.Conv1d(self.latent_dim, 128, 1)

        self.phi = ResnetFC(self.n_view * 9, n_blocks=3, d_out=3,
                            d_latent=self.latent_dim * self.n_view, d_hidden=self.num_hidden_units_phi)

    def r6d2mat(self,d6: torch.Tensor) -> torch.Tensor:
        """
        Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
        using Gram--Schmidt orthogonalisation per Section B of [1].
        Args:
            d6: 6D rotation representation, of size (*, 6). Here corresponds to the two
                first two rows of the rotation matrix. 
        Returns:
            batch of rotation matrices of size (*, 3, 3)
        [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
        On the Continuity of Rotation Representations in Neural Networks.
        IEEE Conference on Computer Vision and Pattern Recognition, 2019.
        Retrieved from http://arxiv.org/abs/1812.07035
        """

        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)  # corresponds to row
    
    def extract_intrinsics(self,intrinsics):
        """
        Extracts the values of fx, fy, cx, and cy from intrinsic matrices.

        Args:
        intrinsics (numpy.ndarray): An array of shape (batch_size, num_view, 4, 4) containing intrinsic matrices.

        Returns:
        tuple: A tuple containing four lists (fx_list, fy_list, cx_list, cy_list) with shapes (batch_size, num_view).
        """
        batch_size, _, _, _ = intrinsics.shape

        fx_list = []
        fy_list = []
        cx_list = []
        cy_list = []

        for i in range(batch_size):
        
            fx_list.append(intrinsics[i, 0, 0, 0])
            fy_list.append(intrinsics[i, 0, 1, 1])
            cx_list.append(intrinsics[i, 0, 0, 2])
            cy_list.append(intrinsics[i, 0, 1, 2])
        
        fx_list = torch.stack(fx_list).reshape(batch_size, 1)
        fy_list = torch.stack(fy_list).reshape(batch_size, 1)
        cx_list = torch.stack(cx_list).reshape(batch_size, 1)
        cy_list = torch.stack(cy_list).reshape(batch_size, 1)
        
        return [fx_list, fy_list, cx_list, cy_list]
    
    def get_z(self, input, val=False):
        """
        Extract features, estimate pose and find correspondence fields.
        
        Args:
        input (dict): A dictionary containing the input data.
        Returns:
        tuple: extracted features, estimated pose, correspondence fields.
        """
      
        rgb = input['context']['rgb']
        B, n_ctxt, H, W, C = rgb.shape

        intrinsics = input['context']['intrinsics']
        context = input['context']
      
        # Flatten first two dims (batch and number of context)
        rgb = torch.flatten(rgb, 0, 1)
        intrinsics = torch.flatten(intrinsics, 0, 1)
        intrinsics = intrinsics[:, None, :, :]
        rgb = rgb.permute(0, -1, 1, 2) # (b*n_ctxt, ch, H, W)
        self.H, self.W = rgb.shape[-2], rgb.shape[-1]
        
        rgb = (rgb + 1) / 2.
        rgb = normalize_imagenet(rgb)
        rgb = torch.cat([rgb], dim=1)
      
        z = self.encoder.forward(rgb, None, self.n_view)[:3] # (b*n_ctxt, self.latent_dim, H, W)
        z_conv = self.conv_map(rgb[:(B*n_ctxt)])
        
        z_ctxts, flow_ctxts, c_ctxts = self.feature_cost_aggregation(z, self.n_view) # context 2 to context 1 flow and feature maps
        
        # Normalize intrinsics for a 0-1 image
        intrinsics_norm = context['intrinsics'].clone()
        intrinsics_norm[:, :, :2, :] = intrinsics_norm[:, :, :2, :] / self.H
        extracted_intrinsics = self.extract_intrinsics(intrinsics_norm)
        
        pose_feat_ctxt = self.cross_attention( z_ctxts[-1].flatten(-2,-1).transpose(-1,-2), corr = c_ctxts, intrinsics = extracted_intrinsics).reshape([B,-1]) # ctxt 1 and ctxt 2
        
        z_ctxts = z_ctxts + [z_conv]
       
        pose_latent_ctxt = self.pose_regressor(pose_feat_ctxt)[:,:128]
        rot_ctxt, tran_ctxt = self.rotation_regressor(pose_latent_ctxt), self.translation_regressor(pose_latent_ctxt)# Bxn_views x 9, Bxn_views x 3 
        R_ctxt = self.r6d2mat(rot_ctxt)[:, :3, :3] 

        estimated_rel_pose_ctxt = torch.cat((torch.cat((R_ctxt, tran_ctxt.unsqueeze(-1)), dim=-1),torch.FloatTensor([0,0,0,1]).expand(B,1,-1).to(tran_ctxt.device)), dim=1) #estimated pose between query and context 2
     
        return z_ctxts, estimated_rel_pose_ctxt, flow_ctxts

    def forward(self, input, z=None, rel_pose=None,val=False,  flow=None,  debug=False):
     
        out_dict = {}
        input = deepcopy(input)

        query = input['query']
        context = input['context']
        b, n_context = input['context']["rgb"].shape[:2]
        n_qry, n_qry_rays = query["uv"].shape[1:3]

        # Get img features
        if z is None:
            z_orig, estimated_rel_pose, flow_orig = self.get_z(input) # estimated pose [0] == context 1 to context 2, [1] == context 2 to query
            z = z_orig
            rel_pose = estimated_rel_pose
        else:
            z_orig = z
            estimated_rel_pose = rel_pose
            flow_orig = flow

        out_dict['flow'] = flow_orig
       
        up_flow = F.interpolate( flow_orig[0], 256, mode='bilinear') * (256 /  input['context']["rgb"].shape[-2])
        up_flow2 = F.interpolate( flow_orig[1], 256, mode='bilinear') * (256 / input['context']["rgb"].shape[-2])
    
        cyclic_consistency_error = torch.norm(up_flow + warp(up_flow2, up_flow), dim=1).le(10)
        cyclic_consistency_error2 = torch.norm(up_flow2+ warp(up_flow, up_flow2), dim=1).le(10)
        mask_padded = cyclic_consistency_error * get_gt_correspondence_mask(up_flow)
        mask_padded2 = cyclic_consistency_error2 * get_gt_correspondence_mask(up_flow2)

        # Get relative coordinates of the query and context ray in each context camera coordinate system
        context_cam2world = torch.matmul(torch.inverse(context['cam2world']), context['cam2world']) # Identity 
        if val:
            query_cam2world = torch.matmul(torch.inverse(context['cam2world'])[:, 0].unsqueeze(1), query['cam2world']) # relpose between query and context
            query_cam2world = torch.cat((query_cam2world, torch.matmul(pose_inverse_4x4(estimated_rel_pose).unsqueeze(1), query_cam2world)) ,dim=1) # estimated relpose
        else:
            query_cam2world = torch.matmul(torch.inverse(context['cam2world']), query['cam2world'])
      
        lf_coords = geometry.plucker_embedding(torch.flatten(query_cam2world, 0, 1), torch.flatten(query['uv'].expand(-1, query_cam2world.size(1), -1, -1).contiguous(), 0, 1), torch.flatten(query['intrinsics'].expand(-1, query_cam2world.size(1), -1, -1).contiguous(), 0, 1))
        lf_coords = lf_coords.reshape(b, n_context, n_qry_rays, 6)

        lf_coords.requires_grad_(True)
        out_dict['coords'] = lf_coords.reshape(b*n_context, n_qry_rays, 6)
        out_dict['uv'] = query['uv']

        # Compute epi line
        if self.no_sample:
            start, end, diff, valid_mask, pixel_val = geometry.get_epipolar_lines_volumetric(lf_coords, query_cam2world, context['intrinsics'], self.H, self.W, self.npoints, debug=debug)
        else:

            # Prepare arguments for epipolar line computation
            intrinsics_norm = context['intrinsics'].clone()
            # Normalize intrinsics for a 0-1 image
            intrinsics_norm[:, :, :2, :] = intrinsics_norm[:, :, :2, :] / self.H

            camera_origin = geometry.get_ray_origin(query_cam2world)
            ray_dir = lf_coords[..., :3]
            extrinsics = torch.eye(4).to(ray_dir.device)[None, None, :, :].expand(ray_dir.size(0), ray_dir.size(1), -1, -1)
            camera_origin = camera_origin[:, :, None, :].expand(-1, -1, ray_dir.size(2), -1)

            s = camera_origin.size()
            #breakpoint()
            # Compute 2D epipolar line samples for the image
            output = project_rays(torch.flatten(camera_origin, 0, 1), torch.flatten(ray_dir, 0, 1), torch.flatten(extrinsics, 0, 1), torch.flatten(intrinsics_norm, 0, 1))
           
            valid_mask = output['overlaps_image']
            start, end = output['xy_min'], output['xy_max']

            start = start.view(*s[:2], *start.size()[1:])
            end = end.view(*s[:2], *end.size()[1:])
            valid_mask = valid_mask.view(*s[:2], valid_mask.size(1))
            start = (start - 0.5) * 2 # -1  ~ 1 
            end = (end - 0.5) * 2

            start[torch.isnan(start)] = 0
            start[torch.isinf(start)] = 0
            end[torch.isnan(end)] = 0
            end[torch.isinf(end)] = 0

            diff = end - start

            valid_mask = valid_mask.float()
            start = start[..., :2]
            end = end[..., :2]

        diff = end - start
        interval = torch.linspace(0, 1, self.npoints, device=lf_coords.device)
   
        if (not self.no_sample):
            pixel_val = None
        else:
            pixel_val = torch.flatten(pixel_val, 0, 1) # projected pixel coordinate

        latents_out = []
        at_wts = []

        diff = end[:, :, :, None, :] - start[:, :, :, None, :]

        if pixel_val is None and (not self.no_sample):
            pixel_val = start[:, :, :, None, :] + diff * interval[None, None, None, :, None]
           
            pixel_val = torch.flatten(pixel_val, 0, 1)
        # until here, primary features are collected
        # Gather corresponding features on line
        interp_val = torch.cat([F.grid_sample(latent, pixel_val, mode='bilinear', padding_mode='border', align_corners=False) for latent in z], dim=1) #channel wise concat
       
      
        #pixel val -> 1 -> 1'  2-> 2'
        flow_composite = F.grid_sample(torch.stack((flow_orig[2], flow_orig[3]), dim=1).flatten(0,1), pixel_val)  #  2->1',    1->2'
        flow_interp_val = torch.cat([F.grid_sample(torch.stack((latent.view(b,n_context,*latent.shape[1:])[:,1],latent.view(b,n_context,*latent.shape[1:])[:,0]),dim=1).flatten(0,1), flow_composite.permute(0,2,3,1), mode='bilinear', padding_mode='border', align_corners=False) for latent in z], dim=1)
       

        # Find the 3D point correspondence in every other camera view
    
        # Find the nearest neighbor latent in the other frame when given 2 views
        # pt == (B x n_ctxt) x n_rays x n_samples x 3
        pt, _, _, _ = geometry.get_3d_point_epipolar(lf_coords.flatten(0, 1), pixel_val, context_cam2world.flatten(0, 1), self.H, self.W, context['intrinsics'].flatten(0, 1))
        if val:
            context_rel_cam2world_view1 = torch.matmul(torch.inverse(context['cam2world'][:, 0:1]), context['cam2world'][:,0].unsqueeze(1)) # relpose from 2 to 1    1 w2c * 2 c2w index[1]   -> index[0] indicates identity
            context_rel_cam2world_view1 = torch.cat((context_rel_cam2world_view1, estimated_rel_pose.unsqueeze(1) ), dim=1) 
            context_rel_cam2world_view2 = torch.matmul(torch.inverse(context['cam2world'][:, 1:2]), context['cam2world'][:,-1].unsqueeze(1)) # relpose from 1 to 2     #c2w      c2w_1 * c2w_2^t -> c2w_2 * c2w_1 ^T 
            context_rel_cam2world_view2 = torch.cat((pose_inverse_4x4(estimated_rel_pose).unsqueeze(1), context_rel_cam2world_view2 ), dim=1) 
        else:
            context_rel_cam2world_view1 = torch.matmul(torch.inverse(context['cam2world'][:, 0:1]), context['cam2world'])
            context_rel_cam2world_view2 = torch.matmul(torch.inverse(context['cam2world'][:, 1:2]), context['cam2world'])
        
            # project using relpose so pt contains context 1 3d point and context 2 3d point
            # context_rel_cam2world_view1 contains relpose 1->1 and 2->1 
        pt_view1 = encode_relative_point(pt, context_rel_cam2world_view1)  # so this is 1 x (1->1)  and 2 x (2->1) 

        pt_view2 = encode_relative_point(pt, context_rel_cam2world_view2)  # this is 1 x (1->2) and 2x (2->2)

        intrinsics_view1 = context['intrinsics'][:, 0]
        intrinsics_view2 = context['intrinsics'][:, 1]

        s = pt_view1.size()
        pt_view1 = pt_view1.view(b, n_context, *s[1:])
        pt_view2 = pt_view2.view(b, n_context, *s[1:])

        s = interp_val.size()
        interp_val = interp_val.view(b, n_context, *s[1:])
        flow_interp_val = flow_interp_val.view(b, n_context, *s[1:])
      

        interp_val_1 = interp_val[:, 0] #  primary feature of context1 
        interp_val_2 = interp_val[:, 1] # primary feature of context2

        pt_view1_context1 = pt_view1[:, 0] #  1 x (1->1)
        pt_view1_context2 = pt_view1[:, 1] # 2 x (2->1) 

        pt_view2_context1 = pt_view2[:, 0] #  1 x (1->2)
        pt_view2_context2 = pt_view2[:, 1] #  2 x (2->2)

        pixel_val_view2_context1 = geometry.project(pt_view2_context1[..., 0], pt_view2_context1[..., 1], pt_view2_context1[..., 2], intrinsics_view2)
        pixel_val_view2_context1 = normalize_for_grid_sample(pixel_val_view2_context1[..., :2], self.H, self.W)
        
        # projected to context 1 using intrinsic of context 1 so this is getting secondary feature of context 1
        # but this is coordinate
        pixel_val_view1_context2 = geometry.project(pt_view1_context2[..., 0], pt_view1_context2[..., 1], pt_view1_context2[..., 2], intrinsics_view1)  
        pixel_val_view1_context2 = normalize_for_grid_sample(pixel_val_view1_context2[..., :2], self.H, self.W)

        pixel_val_stack = torch.stack([pixel_val_view1_context2, pixel_val_view2_context1], dim=1).flatten(0, 1) # (b n_ctxt) n_ray n_sample 2 
        interp_val_nearest = torch.cat([F.grid_sample(latent, pixel_val_stack, mode='bilinear', padding_mode='zeros', align_corners=False) for latent in z], dim=1) # grid sampling from context 1 and 2 
        interp_val_nearest = interp_val_nearest.view(b, n_context, *s[1:])
        interp_val_nearest_1 = interp_val_nearest[:, 0] # seoncdary feature of context 1  this means that from context 2, using rel pose 2->1, grid sampling from context 1 image
        interp_val_nearest_2 = interp_val_nearest[:, 1] # secondary feature of context 2 

        pt_view1_context1 = torch.nan_to_num(pt_view1_context1, 0)
        pt_view2_context2 = torch.nan_to_num(pt_view2_context2, 0)
        pt_view1_context2 = torch.nan_to_num(pt_view1_context2, 0)
        pt_view2_context1 = torch.nan_to_num(pt_view2_context1, 0)
    
        pt_view1_context1 = pt_view1_context1.detach()
        pt_view2_context2 = pt_view2_context2.detach()

        
        interp_val_1_view_1 = torch.cat([interp_val_1, torch.tanh(pt_view1_context1 / 5.).permute(0, 3, 1, 2)], dim=1) # primary feature of context 1 
        interp_val_1_view_3 = torch.cat([interp_val_nearest_2, torch.tanh(pt_view2_context1 / 5.).permute(0, 3, 1, 2)], dim=1) # interp_val_nearest_2 is secondary feature of context 2  pt_view2_context1 point used to get the secondary feature

        interp_val_1_encode_1 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_1)))
        interp_val_1_encode_3 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_1_view_3)))


        interp_val_1_avg = torch.stack([interp_val_1_encode_1, interp_val_1_encode_3], dim=1).flatten(1, 2)

        interp_val_2_view_2 = torch.cat([interp_val_2, torch.tanh(pt_view2_context2 / 5.).permute(0, 3, 1, 2)], dim=1)
        interp_val_2_view_3 = torch.cat([interp_val_nearest_1, torch.tanh(pt_view1_context2 / 5.).permute(0, 3, 1, 2)], dim=1)

        interp_val_2_encode_2 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_2)))
        interp_val_2_encode_3 = self.query_encode_latent_2(F.relu(self.query_encode_latent(interp_val_2_view_3)))
    
    
        interp_val_2_avg = torch.stack([interp_val_2_encode_2, interp_val_2_encode_3 ], dim=1).flatten(1, 2) # b c 192 64
        
        interp_val = torch.stack([interp_val_1_avg, interp_val_2_avg], dim=1).flatten(0, 1)
     
        joint_latent = self.latent_value(interp_val)
        s = interp_val.size()
   
        # Compute key value
        key_val = self.key_map_2(F.relu(self.key_map(interp_val))) # (b*n_ctxt, n_pix, interval_steps, latent)
    
        # Get camera ray direction of each epipolar pixel coordinate 
        cam_rays = geometry.get_ray_directions_cam(pixel_val, context['intrinsics'].flatten(0, 1), self.H, self.W)

        # Ray direction of the query ray to be rendered 
        ray_dir = lf_coords[..., :3].flatten(0, 1)
        ray_dir = ray_dir[:, :, None]
        ray_dir = ray_dir.expand(-1, -1, cam_rays.size(2), -1)

        # 3D coordinate of each epipolar point in 3D
        # depth, _, _ = geometry.get_depth_epipolar(lf_coords.flatten(0, 1), pixel_val, query_cam2world, self.H, self.W, context['intrinsics'].flatten(0, 1))
        pt, dist, parallel, equivalent = geometry.get_3d_point_epipolar(lf_coords.flatten(0, 1), pixel_val, context_cam2world.flatten(0, 1), self.H, self.W, context['intrinsics'].flatten(0, 1))
        
        # Compute the origin of the query ray
        query_ray_orig = geometry.get_ray_origin(query_cam2world).flatten(0, 1)
        query_ray_orig = query_ray_orig[:, None, None]
        query_ray_orig_ex = torch.broadcast_to(query_ray_orig, cam_rays.size())

        # Compute depth of the computed 3D coordinate (with respect to query camera)
        depth = torch.norm(pt - query_ray_orig, p=2, dim=-1)[..., None]

        # Set NaN and large depth values to a finite value
        depth[torch.isnan(depth)] = 1000000
        depth[torch.isinf(depth)] = 1000000
        depth = depth.detach()

        pixel_dist = pixel_val[:, :, :1, :] - pixel_val[:, :, -1:, :]
        pixel_dist = torch.norm(pixel_dist, p=2, dim=-1)
  
        # Origin of the context camera ray (always zeros)
        cam_origin = torch.zeros_like(query_ray_orig_ex)

        # Encode depth with tanh to encode different scales of depth values depth values
        depth_encode = torch.cat([torch.tanh(depth), torch.tanh(depth / 10.), torch.tanh(depth / 100.), torch.tanh(depth / 1000.)], dim=-1)

        # Compute query coordinates by combining context ray info, query ray info, and 3D depth of epipolar line
        local_coords = torch.cat([cam_rays, cam_origin, ray_dir, depth_encode, query_ray_orig_ex], dim=-1).permute(0, 3, 1, 2)
        coords_embed = self.query_embed_2(F.relu(self.query_embed(local_coords)))
        
        # Multiply key and value pairs
      
        dot_at_joint = torch.einsum('bijk,bijk->bjk', key_val, coords_embed) / 11.31 
        dot_at_joint = dot_at_joint.view(b, n_context, n_qry_rays, self.npoints).permute(0, 2, 1, 3).reshape(b, n_qry_rays, n_context * (self.npoints)) 
        
        at_wt_joint = F.softmax(dot_at_joint, dim=-1) #* interp_matchability
        at_wt_joint = torch.flatten(at_wt_joint.view(b, n_qry_rays, self.n_view, self.npoints).permute(0, 2, 1, 3), 0, 1)

        z_local = (joint_latent * at_wt_joint[:, None, :, :]).sum(dim=-1)
        s = z_local.size()
      
        z_local = z_local.view(b, n_context, s[1], n_qry_rays)
        z_sum = z_local.sum(dim=1)
        z_local = torch.cat([z_sum for i in range(n_context)], dim=1).view(*s)

        at_wt = at_wt_joint
        at_wts.append(at_wt)
        
        # A second round of attention to gather additional information
        if self.repeat_attention:
            z_embed = self.encode_latent(z_local)
            z_embed_local = z_embed[:, :, :, None].expand(-1, -1, -1, local_coords.size(-1))

            # Concatenate the previous cross-attention vector as context for second round of attention
            query_embed_local = torch.cat([z_embed_local, local_coords], dim=1)
            query_embed_local = self.query_repeat_embed_2(F.relu(self.query_repeat_embed(query_embed_local)))

            dot_at = torch.einsum('bijk,bijk->bjk', query_embed_local, coords_embed) / 11.31 
            dot_at = dot_at.view(b, n_context, n_qry_rays, self.npoints).permute(0, 2, 1, 3).reshape(b, n_qry_rays, n_context * (self.npoints))
            at_wt_joint = F.softmax(dot_at, dim=-1)#* interp_matchability
           
            # Compute second averaged feature after cross-attention 
            at_wt_joint = torch.flatten(at_wt_joint.view(b, n_qry_rays, self.n_view, self.npoints).permute(0, 2, 1, 3), 0, 1)
            z_local = (joint_latent * at_wt_joint[:, None, :, :]).sum(dim=-1) + z_local
            z_local = z_local.view(b, n_context, s[1], n_qry_rays)

            z_sum = z_local.sum(dim=1)
            z_local = torch.cat([z_sum for i in range(n_context)], dim=1).view(*s)

        latents_out.append(z_local)

        z = torch.cat(latents_out, dim=1).permute(0, 2, 1).contiguous()
        out_dict['pixel_val'] = pixel_val.cpu()
        out_dict['at_wts'] = at_wts
     
        depth_squeeze = depth.view(b, n_context, n_qry_rays, -1).sum(dim=1)
        at_max_idx = at_wt[..., :].argmax(dim=-1)[..., None, None].expand(-1, -1, -1, 3)

        # Ignore points that are super far away
        pt_clamp = torch.clamp(pt, -100, 100)
        # Get the 3D point that is the average (along attention weight) across epipolar points
        world_point_3d_max = (at_wt[..., None] * pt_clamp).sum(dim=-2) # sum across samples on a line 

        
        
        s = world_point_3d_max.size()
        

        world_point_3d_max = world_point_3d_max.view(b, n_context, *s[1:]).sum(dim=1) # sum across context view
        world_point_3d_max = world_point_3d_max[:, :, None, :]

        # Compute the depth for epipolar line visualization
        world_point_3d_max = geometry.project_cam2world(world_point_3d_max[:, :, 0, :], query['cam2world'][:, 0]) # project from world to query
     
        depth_ray = world_point_3d_max[:, :, 2]

        
        T_to_C1_pts = batch_project_to_other_img(
        query['uv'].squeeze(1), depth_ray, query['intrinsics'][:,0, :3, :3], context['intrinsics'][:, 0 , :3, :3], query_cam2world[:,0]
        )
   
        T_to_C2_pts = batch_project_to_other_img(
        query['uv'].squeeze(1), depth_ray, query['intrinsics'][:,0, :3, :3], context['intrinsics'][:, 1 , :3, :3], query_cam2world[:,1]
        )
        
        matchability_cycle_mask = generate_mask_from_confidence_score(T_to_C2_pts,   mask_padded2, depth_ray.shape[-1])
        out_dict['matchability_cycle_mask'] = matchability_cycle_mask
       

        C2_pts_to_C1, mask_c2 = flow2kps(T_to_C2_pts, flow_orig[1], depth_ray.shape[-1])

    
        
        # Clamp depth to make sure things don't get too large due to numerical instability
        depth_ray = torch.clamp(depth_ray, 0, 10)
        out_dict['T_to_C1_pts'] = T_to_C1_pts
      
        out_dict['mask_c2'] = mask_c2
        out_dict['T_to_C2_pts'] = T_to_C2_pts
        out_dict['C2_pts_to_C1'] = C2_pts_to_C1.transpose(1,2)
        out_dict['at_wt'] = at_wt
        out_dict['at_wt_max'] = at_max_idx[:, :, :, 0]
        out_dict['depth_ray'] = depth_ray[..., None]
        
        out_dict['coords'] = torch.cat([out_dict['coords'], query_ray_orig_ex[:, :, 0, :]], dim=-1)

        # Plucker embedding for query ray 
        coords = out_dict['coords']
        s = coords.size()
        coords = torch.flatten(coords.view(b, n_context, n_qry_rays, s[-1]).permute(0, 2, 1, 3), -2, -1)

        zsize = z.size()
        z_flat = z.view(b, n_context, *zsize[1:]).permute(0, 2, 1, 3)

        z_flat = torch.flatten(z_flat, -2, -1)

        coords = torch.cat((z_flat, coords), dim=-1) # 832 18 

        # Light field decoder using the gather geometric context
        
        lf_out = self.phi(coords)
        rgb = lf_out[..., :3]

        # Mask invalid regions (no epipolar line correspondence) to be white
        valid_mask = valid_mask.bool().any(dim=1).float()
        rgb = rgb * valid_mask[:, :, None] + 1 * (1 - valid_mask[:, :, None])
        out_dict['valid_mask'] = valid_mask[..., None]

        rgb = rgb.view(b, n_qry, n_qry_rays, 3)
        
        out_dict['rgb'] = rgb
 
        # Return the multiview latent for each image (so we can cache computation of multiview encoder)
        out_dict['z'] = z_orig
        out_dict['rel_pose_flip'] = pose_inverse_4x4(estimated_rel_pose)
        out_dict['rel_pose'] = (estimated_rel_pose)
        out_dict['gt_rel_pose'] = torch.matmul(torch.inverse(context['cam2world'][:,0]), context['cam2world'][:,1])
        out_dict['gt_rel_pose_flip'] = torch.inverse(torch.matmul(torch.inverse(context['cam2world'][:,-1]), context['cam2world'][:,0]))
        return out_dict