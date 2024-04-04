import random
import os
import os.path as osp
import torch
import numpy as np
from glob import glob

current_dir = osp.dirname(os.path.abspath(__file__))
import sys
sys.path.append(current_dir + '/../')

from utils_training import  data_util
from collections import defaultdict
import os.path as osp
from imageio import imread
from torch.utils.data import Dataset
from pathlib import Path
import cv2
from tqdm import tqdm
from scipy.io import loadmat

import sys
import pdb

def augment(rgb, intrinsics, c2w_mat):

    H, W, _ = rgb.shape
    rgb = cv2.resize(rgb, (256, 256))
    xscale = 256 / W
    yscale = 256 / H

    intrinsics[0, 0] = intrinsics[0, 0] * xscale
    intrinsics[1, 1] = intrinsics[1, 1] * yscale

    return rgb, intrinsics, c2w_mat

class Camera(object):
    def __init__(self, entry):
        fx, fy, cx, cy = entry[1:5]
        self.intrinsics = np.array([[fx, 0, cx, 0],
                                    [0, fy, cy, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
        w2c_mat = np.array(entry[7:]).reshape(3, 4)
        w2c_mat_4x4 = np.eye(4)
        w2c_mat_4x4[:3, :] = w2c_mat
        self.w2c_mat = w2c_mat_4x4
        self.c2w_mat = np.linalg.inv(w2c_mat_4x4)


def unnormalize_intrinsics(intrinsics, h, w):
    intrinsics = intrinsics.copy()
    intrinsics[0] *= w
    intrinsics[1] *= h
    return intrinsics


def parse_pose_file(file):
    f = open(file, 'r')
    cam_params = {}
    for i, line in enumerate(f):
        if i == 0:
            continue
        entry = [float(x) for x in line.split()]
        id = int(entry[0])
        cam_params[id] = Camera(entry)
    return cam_params


def parse_pose(pose, timestep):
    timesteps = pose[:, :1]
    timesteps = np.around(timesteps)
    mask = (timesteps == timestep)[:, 0]
    pose_entry = pose[mask][0]
    camera = Camera(pose_entry)

    return camera


def get_camera_pose(scene_path, all_pose_dir, uv, views=1):
    npz_files = sorted(scene_path.glob("*.npz"))
    npz_file = npz_files[0]
    data = np.load(npz_file)
    all_pose_dir = Path(all_pose_dir)

    rgb_files = list(data.keys())

    timestamps = [int(rgb_file.split('.')[0]) for rgb_file in rgb_files]
    sorted_ids = np.argsort(timestamps)

    rgb_files = np.array(rgb_files)[sorted_ids]
    timestamps = np.array(timestamps)[sorted_ids]

    camera_file = all_pose_dir / (str(scene_path.name) + '.txt')
    cam_params = parse_pose_file(camera_file)
    # H, W, _ = data[rgb_files[0]].shape

    # Weird cropping of images
    H, W = 256, 456

    xscale = W / min(H, W)
    yscale = H / min(H, W)


    query = {}
    context = {}

    render_frame = min(128, rgb_files.shape[0])

    query_intrinsics = []
    query_c2w = []
    query_rgbs = []
    for i in range(1, render_frame):
        rgb = data[rgb_files[i]]
        timestep = timestamps[i]

        # rgb = cv2.resize(rgb, (W, H))
        intrinsics = unnormalize_intrinsics(cam_params[timestep].intrinsics, H, W)

        intrinsics[0, 2] = intrinsics[0, 2] / xscale
        intrinsics[1, 2] = intrinsics[1, 2] / yscale
        rgb = rgb.astype(np.float32) / 127.5 - 1

        query_intrinsics.append(intrinsics)
        query_c2w.append(cam_params[timestep].c2w_mat)
        query_rgbs.append(rgb)

    context_intrinsics = []
    context_c2w = []
    context_rgbs = []

    if views == 1:
        render_ids = [0]
    elif views == 2:
        render_ids = [0, min(len(rgb_files) - 1, 128)]
    elif views == 3:
        render_ids = [0, min(len(rgb_files) - 1, 128) // 2, min(len(rgb_files) - 1, 128)]
    else:
        assert False

    for i in render_ids:
        rgb = data[rgb_files[i]]
        timestep = timestamps[i]
     
        H, W = rgb.shape[0], rgb.shape[1] # Some images are height less than 360

        rgb = data_util.square_crop_img(rgb)
        rgb = cv2.resize(rgb, (256, 256)) # all images are reshaped into (256, 256)

        intrinsics = unnormalize_intrinsics(cam_params[timestep].intrinsics, 256, W*(256/H))
        xscale = W / min(H, W)
        yscale = H / min(H, W)

        intrinsics[0, 2] = intrinsics[0, 2] / xscale
        intrinsics[1, 2] = intrinsics[1, 2] / yscale


        rgb = rgb.astype(np.float32) / 127.5 - 1

        context_intrinsics.append(intrinsics)
        context_c2w.append(cam_params[timestep].c2w_mat)
        context_rgbs.append(rgb)

    query = {'rgb': torch.Tensor(query_rgbs)[None].float(),
             'cam2world': torch.Tensor(query_c2w)[None].float(),
             'intrinsics': torch.Tensor(query_intrinsics)[None].float(),
             'uv': uv.view(-1, 2)[None, None].expand(1, render_frame - 1, -1, -1)}
    ctxt = {'rgb': torch.Tensor(context_rgbs)[None].float(),
            'cam2world': torch.Tensor(context_c2w)[None].float(),
            'intrinsics': torch.Tensor(context_intrinsics)[None].float()}

    return {'query': query, 'context': ctxt}

class RealEstate10k():
    def __init__(self, img_root, pose_root,
                 num_ctxt_views, num_query_views, query_sparsity=None,
                 max_num_scenes=None, square_crop=True, augment=True, lpips=False):
        print("Loading RealEstate10k...")
        self.num_ctxt_views = num_ctxt_views
        self.num_query_views = num_query_views
        self.query_sparsity = query_sparsity

        all_im_dir = Path(img_root)
        # self.all_pose_dir = Path(pose_root)
        self.all_pose = loadmat(pose_root)
        self.lpips = lpips
        self.eval = eval

        self.all_scenes = sorted(all_im_dir.glob('*/'))
        dummy_img_path = str(next(self.all_scenes[0].glob("*.npz")))

        if max_num_scenes:
            self.all_scenes = list(self.all_scenes)[:max_num_scenes]

        data = np.load(dummy_img_path)
        key = list(data.keys())[0]
        im = data[key]

        print(im.shape)
        H, W = im.shape[:2]
        H, W = 256, 455
        self.H, self.W = H, W
        self.augment = augment

        self.square_crop = square_crop

        xscale = W / min(H, W)
        yscale = H / min(H, W)

        dim = min(H, W)

        self.xscale = xscale
        self.yscale = yscale

        # For now the images are already square cropped
        self.H = 256
        self.W = 455

        print(f"Resolution is {H}, {W}.")

        if self.square_crop:
            i, j = torch.meshgrid(torch.arange(0, dim), torch.arange(0, dim))
        else:
            i, j = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))

        self.uv = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)

        self.uv = self.uv[None].permute(0, -1, 1, 2).permute(0, 2, 3, 1)
        self.uv = self.uv.reshape(-1, 2)

        self.scene_path_list = list(Path(img_root).glob("*/"))


    def __len__(self):
        return len(self.all_scenes)

    def __getitem__(self, idx):
        for retry in range(1000):
                try:
                    scene_path = self.all_scenes[idx]
                    npz_files = sorted(scene_path.glob("*.npz"))

                    name = scene_path.name

                    if name not in self.all_pose:
                        return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

                    pose = self.all_pose[name]

                    if len(npz_files) == 0:
                        return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

                    npz_file = npz_files[0]
                    try:
                        data = np.load(npz_file)
                    except:
                        print(npz_file)
                        return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

                    rgb_files = list(data.keys())
                    window_size = 128

                    if len(rgb_files) <= 10:
                        return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

                    timestamps = [int(rgb_file.split('.')[0]) for rgb_file in rgb_files]
                    sorted_ids = np.argsort(timestamps)

                    rgb_files = np.array(rgb_files)[sorted_ids]
                    timestamps = np.array(timestamps)[sorted_ids]

                    assert (timestamps == sorted(timestamps)).all()
                    num_frames = len(rgb_files)
                    # window_size = 32
                    shift = np.random.randint(low=-1, high=2)

                    left_bound = 0
                    right_bound = int((num_frames - 1))
                    candidate_ids = np.arange(left_bound, right_bound)


                    nframe = 1
                    nframe_view = 50

                    if len(candidate_ids) < self.num_ctxt_views:
                        return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

                    id_feats = []

                    for i in range(self.num_ctxt_views):
                        if len(candidate_ids) == 0:
                            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

                        id_feat = np.random.choice(candidate_ids, size=1,
                                                replace=False)
                        candidate_ids = candidate_ids[(candidate_ids < (id_feat - nframe_view)) | (candidate_ids > (id_feat + nframe_view))]

                        id_feats.append(id_feat.item())

                    id_feat = np.array(id_feats)

                    if self.num_ctxt_views == 2:
                        low = np.min(id_feat) - 32
                        high = np.max(id_feat) + 32
                        low = max(low, 0)
                        high = min(high, num_frames - 1)

                        if high <= low:
                            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

                        id_render = np.random.randint(low=low, high=high, size=self.num_query_views)
                    elif self.num_ctxt_views == 1:
                        low = np.min(id_feat) - 64
                        high = np.max(id_feat) + 64
                        low = max(low, 0)
                        high = min(high, num_frames - 1)

                        if high <= low:
                            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

                        id_render = np.random.randint(low=low, high=high, size=self.num_query_views)
                    elif self.num_ctxt_views == 3:
                        low = np.min(id_feat) + 64
                        high = np.max(id_feat) - 64

                        if high <= low:
                            return self.__getitem__(random.randint(0, len(self.all_scenes) - 1))

                        id_render = np.random.randint(low=low, high=high, size=self.num_query_views)
                    else:
                        assert False

                    query_rgbs = []
                    query_intrinsics = []
                    query_c2w = []
                    uvs = []

                    for id in id_render:
                        rgb_file = rgb_files[id]
                        rgb = data[rgb_file]

                        if rgb.shape[0] == 360:
                            rgb = cv2.resize(rgb, (self.W, self.H))

                        if self.square_crop:
                            rgb = data_util.square_crop_img(rgb)

                        cam_param = parse_pose(pose, timestamps[id])

                        intrinsics = unnormalize_intrinsics(cam_param.intrinsics, self.H, self.W)

                        if self.square_crop:
                            intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                            intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

                        if self.augment:
                            rgb, intrinsics, cam_param.c2w_mat = augment(rgb, intrinsics, cam_param.c2w_mat)

                        rgb = rgb.astype(np.float32) / 127.5 - 1
                        full_rgb = rgb.copy()
                        img_size = rgb.shape[:2]
                        rgb = rgb.reshape((-1, 3))

                        mask_lpips = 0.0

                        if self.query_sparsity is not None:
                            if self.lpips:
                                mask_lpips = random.randint(0, 1)
                                if mask_lpips:
                                    uv = self.uv
                                    uv = uv.reshape((256, 256, 2))
                                    rgb = rgb.reshape((256, 256, 3))
                                    offset = 32
                                    x_offset, y_offset =  np.random.randint(0, 256-offset), np.random.randint(0, 256-offset)

                                    uv_select = uv[y_offset:y_offset+offset, x_offset:x_offset+offset]
                                    rgb_select = rgb[y_offset:y_offset+offset, x_offset:x_offset+offset]
                                    uv = uv_select.reshape((-1, 2))
                                    rgb = rgb_select.reshape((-1, 3))
                                else:
                                    uv = self.uv
                                    rix = np.random.permutation(uv.shape[0])
                                    rix = rix[:1024]
                                    uv = uv[rix]
                                    rgb = rgb[rix]
                            else:
                                uv = self.uv
                                rix = np.random.permutation(uv.shape[0])
                                rix = rix[:self.query_sparsity]
                                uv = uv[rix]
                                
                                rgb = rgb[rix]

                        else:
                            uv = self.uv

                        uvs.append(uv)
                        query_rgbs.append(rgb)
                        query_intrinsics.append(intrinsics)
                        query_c2w.append(cam_param.c2w_mat)

                    uvs = torch.Tensor(np.stack(uvs, axis=0)).float()
                    ctxt_rgbs = []
                    ctxt_intrinsics = []
                    ctxt_c2w = []

                    for id in id_feat:
                        rgb_file = rgb_files[id]
                        rgb = data[rgb_file]

                        if rgb.shape[0] == 360:
                            rgb = cv2.resize(rgb, (self.W, self.H))

                        if self.square_crop:
                            rgb = data_util.square_crop_img(rgb)

                        cam_param = parse_pose(pose, timestamps[id])

                        intrinsics = unnormalize_intrinsics(cam_param.intrinsics, self.H, self.W)

                        if self.square_crop:
                            intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                            intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

                        if self.augment:
                            rgb, intrinsics, cam_param.c2w_mat = augment(rgb, intrinsics, cam_param.c2w_mat)

                        rgb = rgb.astype(np.float32) / 127.5 - 1

                        ctxt_rgbs.append(rgb)
                        ctxt_intrinsics.append(intrinsics)
                        ctxt_c2w.append(cam_param.c2w_mat)

                    ctxt_rgbs = np.stack(ctxt_rgbs)
                    ctxt_intrinsics = np.stack(ctxt_intrinsics)
                    ctxt_c2w = np.stack(ctxt_c2w)

                    query_rgbs = np.stack(query_rgbs)
                    query_intrinsics = np.stack(query_intrinsics)
                    query_c2w = np.stack(query_c2w)

                    query = {'rgb': torch.from_numpy(query_rgbs).float(),
                            'cam2world': torch.from_numpy(query_c2w).float(),
                            'intrinsics': torch.from_numpy(query_intrinsics).float(),
                            'uv': uvs,
                            'full_rgb': full_rgb,
                            'mask': mask_lpips}

                    ctxt = {'rgb': torch.from_numpy(ctxt_rgbs).float(),
                            'cam2world': torch.from_numpy(ctxt_c2w).float(),
                            'intrinsics': torch.from_numpy(ctxt_intrinsics).float()}
                    break
                except Exception as e:
                    print("sampling failed", e)

        return {'query': query, 'context': ctxt}, query


class RealEstate10kVis():
    def __init__(self, img_root, pose_root,
                 num_ctxt_views, num_query_views, query_sparsity=None,
                 max_num_scenes=None, square_crop=True, augment=True, lpips=False,n_skip=None, overlap=None):
        print("Loading RealEstate10k...")
        self.num_ctxt_views = num_ctxt_views
        self.num_query_views = 3
        self.query_sparsity = query_sparsity
        self.n_skip =n_skip[0] if type(n_skip)==type([]) else n_skip
        all_im_dir = Path(img_root)
        self.all_pose = loadmat(pose_root)
        self.lpips = lpips
        self.overlap = np.load(overlap) if overlap is not None else None
        self.all_scenes = sorted(all_im_dir.glob('*/'))
        dummy_img_path = str(next(self.all_scenes[0].glob("*.npz")))

        if max_num_scenes:
            self.all_scenes = list(self.all_scenes)[:max_num_scenes]

        data = np.load(dummy_img_path)
        key = list(data.keys())[0]
        im = data[key]

        print(im.shape)
        H, W = im.shape[:2]
        H, W = 256, 455
        self.H, self.W = H, W
        self.augment = augment

        self.square_crop = square_crop

        xscale = W / min(H, W)
        yscale = H / min(H, W)

        dim = min(H, W)

        self.xscale = xscale
        self.yscale = yscale

        # For now the images are already square cropped
        self.H = 256
        self.W = 455

        print(f"Resolution is {H}, {W}.")

        if self.square_crop:
            i, j = torch.meshgrid(torch.arange(0, dim), torch.arange(0, dim))
        else:
            i, j = torch.meshgrid(torch.arange(0, W), torch.arange(0, H))

        self.uv = torch.stack([i.float(), j.float()], dim=-1).permute(1, 0, 2)

        self.uv = self.uv[None].permute(0, -1, 1, 2).permute(0, 2, 3, 1)
        self.uv = self.uv.reshape(-1, 2)

        self.scene_path_list = list(Path(img_root).glob("*/"))


    def __len__(self):
        return len(self.all_scenes)

    def __getitem__(self, idx):
        
        _idx = idx
        scene_path = self.all_scenes[idx]
        npz_files = sorted(scene_path.glob("*.npz"))

        name = scene_path.name

        def get_another():
            return self[idx-1 if idx >200  else idx+1]

        if name not in self.all_pose: return get_another()

        pose = self.all_pose[name]

        if len(npz_files) == 0:
            print("npz get another")
            return get_another()

        npz_file = npz_files[0]
        try:
            data = np.load(npz_file)
        except:
            print("npz load error get another")
            return get_another()

        rgb_files = list(data.keys())
        window_size = 128

        if len(rgb_files) <= 20:
            print("<20 rgbs error get another")
            return get_another()

        timestamps = [int(rgb_file.split('.')[0]) for rgb_file in rgb_files]
        sorted_ids = np.argsort(timestamps)

        rgb_files = np.array(rgb_files)[sorted_ids]
        timestamps = np.array(timestamps)[sorted_ids]

        assert (timestamps == sorted(timestamps)).all()
        num_frames = len(rgb_files)
        left_bound = 0
        right_bound = num_frames - 1
        candidate_ids = np.arange(left_bound, right_bound)



        id_feats = []

        n_skip=self.n_skip

        id_feat = np.array(id_feats)
        low = 0
        high = num_frames-1-n_skip*self.num_query_views

        if high <= low:
            n_skip = int(num_frames//(self.num_query_views+1))
            high = num_frames-1-n_skip*self.num_query_views

        base_i=0

        id_render = [base_i+i*n_skip for i in range(self.num_query_views)]

        query_rgbs = []
        query_intrinsics = []
        query_c2w = []
        uvs = []

        
        ctxt_rgbs = []
        ctxt_intrinsics = []
        ctxt_c2w = []
        
        for idx, id in enumerate(id_render):
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = data_util.square_crop_img(rgb)

            cam_param = parse_pose(pose, timestamps[id])

            intrinsics = unnormalize_intrinsics(cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 127.5 - 1
            if idx == 1:
                full_rgb = rgb.copy()
            img_size = rgb.shape[:2]
            rgb = rgb.reshape((-1, 3))

            mask_lpips = 0.0
            if self.query_sparsity is not None and idx == 1:  # train
                uv = self.uv
                rix = np.random.permutation(uv.shape[0])
                rix = rix[:self.query_sparsity]
                uv_q = uv[rix]
                rgb_q = rgb[rix]
            else :
                uv = self.uv
            
         
            uvs.append(uv)
            query_rgbs.append(rgb)
            query_intrinsics.append(intrinsics)
            query_c2w.append(cam_param.c2w_mat)

        uvs = torch.Tensor(np.stack(uvs, axis=0)).float()
        

        for id in id_feat:
            rgb_file = rgb_files[id]
            rgb = data[rgb_file]

            if rgb.shape[0] == 360:
                rgb = cv2.resize(rgb, (self.W, self.H))

            if self.square_crop:
                rgb = data_util.square_crop_img(rgb)


            cam_param = parse_pose(pose, timestamps[id])

            intrinsics = unnormalize_intrinsics(cam_param.intrinsics, self.H, self.W)

            if self.square_crop:
                intrinsics[0, 2] = intrinsics[0, 2] / self.xscale
                intrinsics[1, 2] = intrinsics[1, 2] / self.yscale

            if self.augment:
                rgb, intrinsics, cam_param.c2w_mat = augment(rgb, intrinsics, cam_param.c2w_mat)

            rgb = rgb.astype(np.float32) / 127.5 - 1

            ctxt_rgbs.append(rgb)
            ctxt_intrinsics.append(intrinsics)
            ctxt_c2w.append(cam_param.c2w_mat)

   
        query_rgbs = np.stack(query_rgbs)
        query_intrinsics = np.stack(query_intrinsics)
        query_c2w = np.stack(query_c2w)
        
        query = {'rgb': torch.from_numpy(query_rgbs).float()[1].unsqueeze(0) if self.query_sparsity is None else torch.from_numpy(rgb_q).float().unsqueeze(0),
                 'cam2world': torch.from_numpy(query_c2w).float()[1].unsqueeze(0),
                 'intrinsics': torch.from_numpy(query_intrinsics).float()[1].unsqueeze(0),
                 'full_rgb': full_rgb,
                 'uv': uvs[1].unsqueeze(0) if self.query_sparsity is None else uv_q.unsqueeze(0),
                 }
        
        ctxt = {'rgb': torch.stack((torch.from_numpy(query_rgbs).float()[0].reshape(256,256,3), torch.from_numpy(query_rgbs).float()[-1].reshape(256,256,3)), dim =0 ),
                 'cam2world': torch.stack((torch.from_numpy(query_c2w).float()[0], torch.from_numpy(query_c2w).float()[-1]), dim = 0) ,
                 'intrinsics': torch.stack((torch.from_numpy(query_intrinsics).float()[0], torch.from_numpy(query_intrinsics).float()[-1]), dim = 0) }
        
        return {'query': query, 'context': ctxt}, query,  self.overlap[_idx]