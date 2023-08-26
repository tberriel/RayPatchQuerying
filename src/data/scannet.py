import numpy as np
import imageio
from torch.utils.data import Dataset
import torch
from torch.distributions.normal import Normal
import torchvision
import torchvision.transforms.functional as F

import os

from src.utils.utils_geometry import eulerToMatrix,  unproject_rays, generateVirtualCameras


def colorJitter(img, fn_idx, b, c, s, h):
    for fn_id in fn_idx:
        if fn_id == 0 and b is not None:
            img = F.adjust_brightness(img, b)
        elif fn_id == 1 and c is not None:
            img = F.adjust_contrast(img, c)
        elif fn_id == 2 and s is not None:
            img = F.adjust_saturation(img, s)
        elif fn_id == 3 and h is not None:
            img = F.adjust_hue(img, h)
    return img

class ScanNetDataset(Dataset):
    def __init__(self, path, mode, points_per_item=2048, max_len=None,
                 canonical_view=True, full_scale=False, num_input_images = 5, num_target_images = 5, load_all_data=False, random_views=False, setting="default", target_reduced= 1, virtual_cameras = True, pose_jittering = True, sigma_t = 0.1, sigma_r =0.1, sigma_v = 0.25, mask_non_vis=True, discard_non_vis=False, h_rgb_in=128, w_rgb_in=192, h_rgb_tar=480, w_rgb_tar=640, h_rgb_un=480, w_rgb_un=640, normalize_resnet=True, color_jitter=True):
        """ Loads the ScanNet dataset
        Args:
            path (str): Path to dataset.
            mode (str): 'train', 'val', or 'test'.
            points_per_item (int): Number of target points per scene.
            max_len (int): Limit to the number of entries in the dataset.
            canonical_view (bool): Return data in canonical camera coordinates (like in SRT), as opposed
                to world coordinates.
            full_scale (bool): Return all available target points, instead of sampling.
        """
        self.path = os.path.join(path,mode)
        self.mode = mode
        self.num_target_pixels = points_per_item
        self.max_len = max_len
        self.canonical = canonical_view
        assert not(full_scale and discard_non_vis), "Full scale implies querying non visible pixels"
        self.full_scale = full_scale
        self.mask_non_vis = mask_non_vis
        if discard_non_vis:
            assert mask_non_vis, "Non visible points cannot be discarded if the mask is not computed" 
        self.discard_non_vis = discard_non_vis
        self.load_all_data = load_all_data
        self.random_views = random_views
        assert setting in ["default", "stereo", "interpolation", "extrapolation", "video"]
        self.setting = setting
        self.num_input_images = num_input_images
        self.num_target_images = num_target_images
        if target_reduced>1:
            assert full_scale, "Target reduction is only for full scale."
        self.target_reduced = target_reduced
        self.virtual_cameras = virtual_cameras
        self.pose_jittering = pose_jittering

        self.scene_paths = []
        self.render_kwargs = {
            'min_dist': 0.,
            'max_dist': 10.}

        self.views_per_scene = []         
        assert (h_rgb_un >= h_rgb_tar and h_rgb_un>=h_rgb_in) and (w_rgb_un >= w_rgb_tar and w_rgb_un>=w_rgb_in)
        self.h_rgb_in = h_rgb_in
        self.w_rgb_in = w_rgb_in
        self.h_rgb_tar = h_rgb_tar
        self.w_rgb_tar = w_rgb_tar
        self.h_rgb_un = h_rgb_un
        self.w_rgb_un = w_rgb_un
        
        self.resize_in = torchvision.transforms.Compose([torchvision.transforms.Resize([self.h_rgb_in, self.w_rgb_in], antialias=False)])
        self.resize_tar = torchvision.transforms.Compose([torchvision.transforms.Resize([self.h_rgb_tar, self.w_rgb_tar], antialias=False)])
        self.resize_un = torchvision.transforms.Compose([torchvision.transforms.Resize([self.h_rgb_un, self.w_rgb_un], antialias=False)])
        self.resize_reduced = torchvision.transforms.Compose([torchvision.transforms.Resize([self.h_rgb_tar//target_reduced, self.w_rgb_tar//target_reduced], antialias=False)])
        self.color_jitter = color_jitter
        self.normalize_resnet = None
        
        if normalize_resnet:
            self.normalize_resnet = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        
        drop_scenes = []
        scenes_list = os.listdir(self.path)
        scenes_list.sort()
        self.idxs_scene = []
        self.idxs_idx = []
        if self.setting == "stereo":
            self.list_pairs = []
            self.num_input_images = 2
            self.num_target_images = 2
            with open(os.path.join(path,"stereo_pairs_"+mode+".list"), "r") as file:
                for line in file:
                    if len(line)>2:
                        if line[2:14] not in self.scene_paths:
                            self.scene_paths.append(line[2:14])
                        self.idxs_scene.append(len(self.scene_paths)-1)
                        self.list_pairs.append([int(line[15:19]), int(line[25:29])])
        else:
            for scene in scenes_list:
                num_frames = len(os.listdir(os.path.join(path,mode, scene, "color")))  
                if num_frames < self.num_input_images+self.num_target_images:
                    drop_scenes.append(scene)
                else:
                    self.scene_paths.append(scene)
                    if self.setting in ["default", "extrapolation"]:
                        num_views = num_frames-(self.num_input_images+self.num_target_images)
                        init_view = 0
                    elif self.setting == "interpolation":
                        num_views = num_frames-10
                        init_view = 5
                        self.num_input_images = 2
                        self.num_target_images = 9
                    elif self.setting == "video":
                        num_views = num_frames-18
                        init_view = 9
                    self.views_per_scene.append(num_views)

                    for i in range(init_view, num_views):
                        self.idxs_scene.append(len(self.scene_paths)-1)
                        self.idxs_idx.append(i)
                
        print("Dropped scenes: {}".format(drop_scenes))

        if self.load_all_data:
            images =[]
            T_world_camera =[]
            depth =[]
            intrinsic_d_mat =[]
            for scene in self.scene_paths:
                scene_path = os.path.join(self.path, scene)
                t_images, t_T_world_camera, t_depth, t_intrinsic_d_mat = self._load_in_memory(scene_path)
                images.append(np.array(t_images))
                T_world_camera.append(np.array(t_T_world_camera))
                depth.append(np.array(t_depth))
                intrinsic_d_mat.append(np.array(t_intrinsic_d_mat))
                #del t_images, t_T_world_camera, t_depth, t_intrinsic_d_mat
            self.images = images
            self.T_world_camera = T_world_camera
            self.depth = depth
            self.intrinsic_d_mat = intrinsic_d_mat
        self.__precompute_Ks__()
        self.num_scenes = len(self.scene_paths)
        print(f'ScanNet {mode} dataset loaded: {len(self.idxs_scene)} sequences from {self.num_scenes} scenes.')

        self.t_error = lambda x: np.random.default_rng().normal(0.0, sigma_t, size=[x,3])
        self.r_error = lambda : np.random.default_rng().normal(0.0, sigma_r, size=[3])
        self.v_error = lambda x: np.random.default_rng().normal(0.0, sigma_v, size=[x,3])

    def __diff_Ks__(self, k0, k1, cosine=False, th=0.001):
        if cosine:
            return ((k0*k1).sum()/(np.linalg.det(k0)*np.linalg.det(k1)))<th
        else:
            return np.absolute(k0-k1[0]).sum()<th
        
    def __interpolate_rays__(self, rays, h, w):
            xmap = np.linspace(rays[0,0,0], rays[0,-1,0], w)
            ymap = np.linspace(rays[0,0,1], rays[-1,0,1], h)
            xmap, ymap = np.meshgrid(xmap, ymap, indexing='xy')
            rays_2d = np.stack((xmap, ymap, np.ones_like(xmap)), -1).astype(np.float32)
            return rays_2d

    def __precompute_Ks__(self):
        K_list = []
        K_list_idx = []
        h_d, w_d = 480, 640
        for fold in self.scene_paths:
            K_mat =  np.loadtxt(os.path.join(self.path+"/"+fold, 'intrinsic/intrinsic_depth.txt')).astype(np.float32)
            new=True
            for i,K_j in enumerate(K_list):
                if self.__diff_Ks__(K_mat, K_j) :
                    new = False
                    K_list_idx.append(i)
                    break
            if new:
                down_mat = np.array([
                    [self.w_rgb_un/w_d, 0, 0, 0],
                    [0, self.h_rgb_un/h_d, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ]) 
                K_mat = down_mat@K_mat
                K_list.append([K_mat,np.linalg.inv(K_mat[:3,:3])])
                K_list_idx.append(len(K_list)-1)
        self.K_list = K_list
        self.K_list_idx = K_list_idx

        # Unproject rays to store local rays
        umap = np.linspace(0.5, self.w_rgb_un-0.5, self.w_rgb_un)
        vmap = np.linspace(0.5, self.h_rgb_un-0.5, self.h_rgb_un)
        umap, vmap = np.meshgrid(umap, vmap, indexing='xy')
        points_2d = np.stack((umap, vmap, np.ones_like(umap)), -1).astype(np.float32)
        
        rays_list=[]
        for Ks in  self.K_list:
            local_rays = np.einsum("ij,mnj -> mni",Ks[1],points_2d)
            local_rays = np.concatenate((local_rays,np.ones(local_rays.shape[:-1])[...,None]),axis=-1).astype(np.float32)
            rays_list.append(local_rays)
        self.rays_list=rays_list

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return len(self.idxs_scene)

    def _load_in_memory(self, scene_path, input_views=None, target_views=None):
        def _order_list(list_files):
            list_names = []
            for file in list_files:
                name, _ = file.split(".")
                list_names.append(int(name))
            list_names.sort()
            return list_names

        folders = ["color","pose","depth"]
        fun = [imageio.imread, np.loadtxt, imageio.imread]
        suffix = ["jpg", "txt", "png"]
        kwarg = [{}, {"delimiter":" "}, {}]
        data_input = [] # "input_images","input_camera_pose", "input_depth"
        data_target = []# "target_images", "target_camera_pose", "taget_depth"

        for i, folder in enumerate(folders):
            if self.setting == "stereo" and input_views is not None:
                in_path = input_views 
                tar_path = target_views 
            else:
                path = os.listdir(scene_path+"/"+folder)
                path = _order_list(path)

                in_path = np.asarray(path)[input_views] if input_views is not None else path
                tar_path = np.asarray(path)[target_views] if target_views is not None else path

            var = [np.asarray(fun[i](
            os.path.join(scene_path, folder, str(j).zfill(4)+"."+suffix[i]), **kwarg[i])) for j in in_path] 
            data_input.append(var)
            del var
            
            if target_views  is not None:
                var = [np.asarray(fun[i](
                os.path.join(scene_path,folder, str(j).zfill(4)+"."+suffix[i]), **kwarg[i])) for j in tar_path] 
                data_target.append(var)
                del var

        data = data_input + data_target
        del data_input, data_target

        data = data + [np.loadtxt(os.path.join(scene_path, 'intrinsic/intrinsic_depth.txt'), delimiter=" ").astype(np.float32)]
        
        return data
    
    def __getitem__(self, idx):
        scene_idx = idx % self.num_scenes
        scene = self.scene_paths[scene_idx]

        if self.random_views:
            input_views = np.random.choice(np.arange(self.views_per_scene[scene_idx]), size=self.num_input_images, replace=False)
            target_views = np.array(list(set(range(self.views_per_scene[scene_idx])) - set(input_views)))        
            target_views = np.random.choice(target_views, size=self.num_target_images, replace=False)
        else:

            scene_idx = self.idxs_scene[idx]
            scene = self.scene_paths[scene_idx]
            if self.setting == "stereo":
                input_views = np.array(self.list_pairs[idx])
                target_views = np.array(self.list_pairs[idx])
            else:
                set_idx = self.idxs_idx[idx]
                interval = self.num_input_images//self.num_target_images
                if self.setting == "interpolation":
                    interval = 1
                    input_views = set([set_idx-5, set_idx+5])
                    target_views = set(range(set_idx-4, set_idx+5, interval))
                elif self.setting == "extrapolation":
                    input_views = set(range(set_idx, set_idx+self.num_input_images))
                    target_views = set(range(set_idx+self.num_input_images, set_idx+self.num_input_images+self.num_target_images))
                elif self.setting == "video":
                    input_views = [set_idx-9, set_idx-6, set_idx-3, set_idx+3, set_idx+6, set_idx+9]
                    target_views = [set_idx]
                else:
                    input_views = set(range(set_idx, set_idx+self.num_input_images+self.num_target_images, interval+1))
                    target_views = set(range(set_idx+1, set_idx+self.num_input_images+self.num_target_images+1, interval+1))

                input_views = np.array(list(input_views))
                target_views = np.array(list(target_views))

        scene_path = os.path.join(self.path, scene)

        if self.load_all_data:
            input_images = self.images[scene_idx][input_views]
            input_T_world_camera = self.T_world_camera[scene_idx][input_views]
            input_depth = self.depth[scene_idx][input_views]
            intrinsic_d_mat = self.intrinsic_d_mat[scene_idx]
            target_images = self.images[scene_idx][target_views]
            target_T_world_camera = self.T_world_camera[scene_idx][target_views]
            target_depth = self.depth[scene_idx][target_views]
        else:
            input_images, input_T_world_camera, input_depth, target_images, target_T_world_camera, target_depth, intrinsic_d_mat = self._load_in_memory(scene_path, input_views=input_views, target_views=target_views)
            
        input_images = np.stack(input_images, 0).astype(np.float32)/255.0
        input_images = np.transpose(input_images, (0,3,1,2))
        input_T_world_camera = np.stack(input_T_world_camera, axis=0).astype(np.float32)
        input_depth =  np.stack(input_depth, axis=0).astype(np.float32)/1000.0
        intrinsic_d_mat = self.K_list[self.K_list_idx[scene_idx]][0]

        target_images = np.stack(target_images, 0).astype(np.float32) /255.0
        target_images = np.transpose(target_images, (0,3,1,2))
        target_T_world_camera =  np.stack(target_T_world_camera, axis=0).astype(np.float32) # T_world_camera, to project use T_camera_world 
        target_depth =  np.stack(target_depth, axis=0).astype(np.float32)/1000.0


        if self.canonical:  # Transform to canonical camera coordinates
            sampled_idx = np.random.choice(np.arange(self.num_input_images+self.num_target_images),
                                            size=1,
                                            replace=False)
            if sampled_idx < self.num_input_images:
                canonical_extrinsic = np.linalg.inv(input_T_world_camera[sampled_idx])
            else:
                canonical_extrinsic = np.linalg.inv(target_T_world_camera[sampled_idx-self.num_input_images,])
            input_T_world_camera = np.einsum("ij,bjk->bik", canonical_extrinsic[0], input_T_world_camera)
            target_T_world_camera = np.einsum("ij,bjk->bik", canonical_extrinsic[0], target_T_world_camera)

        input_camera_pose = input_T_world_camera[...,:3,3]
        target_camera_pose =  target_T_world_camera[...,:3,3]

        local_rays =  self.rays_list[self.K_list_idx[scene_idx]]

        target_rays = np.einsum("bij,nmj -> bnmi",target_T_world_camera, local_rays)
        input_rays = np.einsum("bij,nmj -> bnmi",input_T_world_camera, local_rays) 

        if self.pose_jittering:            
            def translationNoise(e_r, e_t):
                T_error = np.eye(4)
                T_error[:3,:3] = eulerToMatrix(e_r)
                T_error[:3,3] = e_t
                return T_error

            T_error = translationNoise(self.r_error(), self.t_error(1)).astype(np.float32)
            input_T_world_camera = np.einsum("ij,bjk->bik", T_error, input_T_world_camera)
            input_rays = np.einsum("ij,bmnj->bmni", T_error, input_rays)
            input_camera_pose = input_T_world_camera[...,:3,3]

            target_T_world_camera = np.einsum("ij,bjk->bik", T_error, target_T_world_camera)
            target_rays = np.einsum("ij,bmnj->bmni", T_error, target_rays)
            target_camera_pose =  target_T_world_camera[...,:3,3]

        target_rays =  np.asarray(self.resize_tar(torch.tensor(target_rays).permute(0,3,1,2)).permute(0,2,3,1))
        input_rays = np.asarray(self.resize_in(torch.tensor(input_rays).permute(0,3,1,2)).permute(0,2,3,1))

        #target_rays =  self.__interpolate_rays__(target_rays, h=self.h_rgb_tar, w = self.w_rgb_tar)
        #input_rays =  self.__interpolate_rays__(input_rays, h=self.h_rgb_in, w = self.w_rgb_in)

        un_depth = np.asarray(self.resize_un(torch.tensor(input_depth)))
        un_rgb = np.asarray(self.resize_un(torch.tensor(input_images)))

        if self.h_rgb_un == self.h_rgb_in and self.w_rgb_un == self.w_rgb_in:
            input_depth = un_depth
            input_images = un_rgb
        else:
            input_depth = np.asarray(self.resize_in(torch.tensor(input_depth)))
            input_images = np.asarray(self.resize_in(torch.tensor(input_images)))

        target_images = np.asarray(self.resize_tar(torch.tensor(target_images)))
        target_depth = np.asarray(self.resize_tar(torch.tensor(target_depth)))

        if self.color_jitter and self.mode == "train" and torch.rand(1)<0.5:
            fn_idx, b, c, s, h = torchvision.transforms.ColorJitter.get_params(brightness=[0.92,1.08],contrast=[0.9,1.1],saturation=[0.8,1.2],hue=[-0.1,0.1])
            
            un_rgb = np.asarray(colorJitter(torch.tensor(un_rgb), fn_idx, b, c, s, h))
            target_images = np.asarray(colorJitter(torch.tensor(target_images), fn_idx, b, c, s, h))

            if self.h_rgb_un == self.h_rgb_in and self.w_rgb_un == self.w_rgb_in:
                input_images = un_rgb
            else:
                input_images = np.asarray(colorJitter(torch.tensor(input_images), fn_idx, b, c, s, h))

        if self.normalize_resnet is not None:
            input_images = np.asarray(self.normalize_resnet(torch.tensor(input_images)))

        target_real_mask = np.ones([target_T_world_camera.shape[0],self.h_rgb_tar,self.w_rgb_tar]).astype(np.bool)
        
        if self.virtual_cameras or self.mask_non_vis:

            ### Downsample inputs, target, and adjust intrinsic matrix to new resolutions
            down_mat_tar = np.array([
                [self.w_rgb_tar/self.w_rgb_un, 0, 0, 0],
                [0, self.h_rgb_tar/self.h_rgb_un, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            intrinsic_d_mat_tar = down_mat_tar@intrinsic_d_mat

            d_th = 0.2
            mask_depth = un_depth>d_th
            # Hacer upsamplig de localr rays, un_depth y un_rgb para trabajar a mayor res?
            #un_rays =  self.__interpolate_rays__(un_rays, h=self.h_rgb_un, w = self.w_rgb_un)
            un_rays = np.asarray(self.resize_un(torch.tensor(local_rays).permute(2,0,1)).permute(1,2,0))
            points_3d_camera = un_rays[...,:3] * un_depth[...,None]
            points_3d_camera = np.concatenate([points_3d_camera,np.ones(points_3d_camera.shape[:-1])[...,None]], axis = -1)
            input_points_3d_world = np.einsum("bij,bnmj -> bnmi",input_T_world_camera,points_3d_camera)

            if self.virtual_cameras:
                e_v = self.v_error(self.num_input_images)
                e_c = self.v_error(self.num_input_images)

                T_virtual = generateVirtualCameras(input_T_world_camera, points_3d_camera, mask_depth, e_v, e_c)
                virtual_rays = np.einsum("bij,nmj -> bnmi", T_virtual, un_rays)
                virtual_rays = np.asarray(self.resize_tar(torch.tensor(virtual_rays).permute(0,3,1,2)).permute(0,2,3,1))

                target_T_world_camera = np.concatenate([target_T_world_camera,T_virtual])   
                target_camera_pose =  target_T_world_camera[...,:3,3]
                target_rays = np.concatenate([target_rays, virtual_rays])        
                target_real_mask = np.concatenate([target_real_mask,np.zeros([T_virtual.shape[0], self.h_rgb_tar,self.w_rgb_tar])]).astype(np.bool)
            
            projected_points_rgb = un_rgb.transpose(0,2,3,1)[mask_depth]
            # Project 3D points into target views            
            input_points_3d_tcam = np.einsum("bij,mj -> bmi",np.linalg.inv(target_T_world_camera), input_points_3d_world[mask_depth])
            projected_points = np.einsum("ij,nmj->nmi", intrinsic_d_mat_tar[:3,:], input_points_3d_tcam)
            projected_points = (projected_points[...,:-1]/projected_points[...,-1:]).astype(np.float64)
            in_bounds_u = np.logical_and(projected_points[...,0]>0,projected_points[...,0]<self.w_rgb_tar) 
            in_bounds_v = np.logical_and(projected_points[...,1]>0,projected_points[...,1]<self.h_rgb_tar)
            in_bounds = np.logical_and(in_bounds_u, in_bounds_v)

            target_vis_mask = np.zeros([target_T_world_camera.shape[0], self.h_rgb_tar, self.w_rgb_tar]).astype(np.bool)
            if self.mask_non_vis and self.h_rgb_tar <= self.h_rgb_un and self.w_rgb_tar <= self.w_rgb_un:
                for i in range(target_images.shape[0]):
                    target_vis_mask[i, projected_points[i,in_bounds[i],1].astype(int),projected_points[i,in_bounds[i],0].astype(int)] = True
            else:
                target_vis_mask[:target_images.shape[0]] = True # When training compute gradient also for parts not seen which have gt
            
            target_vis_mask[:target_images.shape[0]][target_depth<d_th] = False

            if self.virtual_cameras:
                virtual_depth = np.zeros_like(target_depth)
                virtual_rgb = np.zeros_like(target_images)
                for i in range(target_images.shape[0], target_vis_mask.shape[0]):
                    ii = i-target_images.shape[0]
                    target_vis_mask[i, projected_points[i,in_bounds[i],1].astype(int),projected_points[i,in_bounds[i],0].astype(int)] = True
                    virtual_depth[ii, projected_points[i,in_bounds[i],1].astype(int),projected_points[i,in_bounds[i],0].astype(int)] = input_points_3d_tcam[i,in_bounds[i],2] # Assign z in local virtual frame, not in world frame
                    virtual_rgb[ii, :,projected_points[i,in_bounds[i],1].astype(int),projected_points[i,in_bounds[i],0].astype(int)] = projected_points_rgb[in_bounds[i]]
                target_depth = np.concatenate([target_depth, virtual_depth])
                target_images= np.concatenate([target_images, virtual_rgb])

        else:
            target_vis_mask = np.ones_like(target_depth,dtype=np.bool)

        if self.target_reduced > 1:
            target_rays = np.asarray(self.resize_reduced(torch.tensor(target_rays).permute(0,3,1,2)).permute(0,2,3,1))

        _, height, width, _ = target_rays.shape
        target_shape = np.array([target_images.shape[0], self.h_rgb_tar, self.w_rgb_tar, 3],dtype=np.int32)
        target_pixels = np.reshape(target_images.transpose(0,2,3,1), (-1,3))
        target_camera_pose = np.repeat(target_camera_pose[:,None], height*width, axis=1)
        target_camera_pose = np.reshape(target_camera_pose, (-1,3))
        target_rays = np.reshape(target_rays, (-1,4))
        target_depth = target_depth.flatten()
        target_vis_mask = target_vis_mask.flatten()
        target_real_mask = target_real_mask.flatten()

        if not self.full_scale:
            if self.discard_non_vis:
                mask = target_vis_mask>0
                target_pixels = target_pixels[mask]
                target_camera_pose = target_camera_pose[mask] 
                target_rays = target_rays[mask]
                target_depth = target_depth[mask]
                target_vis_mask = target_vis_mask[mask]
                target_real_mask = target_real_mask[mask]
            
            num_pixels = target_pixels.shape[0]
            sampled_idxs = np.random.choice(np.arange(num_pixels),
                                            size=(self.num_target_pixels,),
                                            replace=False)

            target_pixels = target_pixels[sampled_idxs]
            target_camera_pose = target_camera_pose[sampled_idxs] 
            target_rays = target_rays[sampled_idxs]
            target_depth = target_depth[sampled_idxs]
            target_vis_mask = target_vis_mask[sampled_idxs]
            target_real_mask = target_real_mask[sampled_idxs]

            
        result = {
            'input_images':      input_images,              # [k, 3, h, w]
            'input_camera_pos':  input_camera_pose,                                    # [p, 3]
            'input_rays':        input_rays[...,:3],           # [k, h, w, 3]
            'input_depth':       input_depth,           # [k, h, w, 3]
            'target_pixels':     target_pixels,                                  # [p, 3]
            'target_images':     target_images,
            'target_camera_pos': target_camera_pose,                                    # [p, 3]
            'target_rays':       target_rays[...,:3],                                    # [p, 3]
            'target_depth':       target_depth,                                    # [p, 3]
            'target_vis_mask':  target_vis_mask,
            'target_real_mask':  target_real_mask,
            'sceneid':           idx,                                         # int
            'target_shape':      target_shape,                        # [5]]
        }

        if self.canonical:
            result['transform'] = canonical_extrinsic                         # [3, 4] (optional)

        return result

