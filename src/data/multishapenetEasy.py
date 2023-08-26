from torch.utils.data import Dataset
import torchvision
import torch
from torch import tensor
import imageio
import numpy as np 
import os


class MultishapenetEasyDataset(Dataset):
    """
    MultiShapeNet Easy dataloader based on the implementation by Karl Stelzner
    https://github.com/stelzner/obsurf/blob/main/obsurf/data.py
    """
    def __init__(self, path, mode, max_n=6, num_target_images=2, max_views=None, points_per_item=2048, do_frustum_culling=False,
                 target_reduced=False, full_scale=False, canonical_view=False, max_len=None, importance_cutoff=0.5, h= 60, w=80):
        self.path = path
        self.mode = mode
        self.max_n = max_n
        self.full_scale = full_scale
        self.target_reduced = target_reduced
        self.points_per_item = points_per_item
        self.do_frustum_culling = do_frustum_culling
        self.max_len = max_len
        self.importance_cutoff = importance_cutoff
        assert h <= 240 and w <= 320, "Cannot reshape to higher than original resolution"
        self.h = h # 240
        self.w = w # 320
        assert num_target_images<=2 and num_target_images>0, "Number of target images should be either 1 or 2"
        self.num_target_images = num_target_images

        self.render_kwargs = {
            'min_dist': 0.,
            'max_dist': 20.}
        self.max_num_entities = 5 
        self.mode = mode
        self.start_idx, self.end_idx = {'train': (0, 80000),
                                        'val': (80000, 80500),
                                        'test': (90000, 100000)}[mode]


        self.metadata = np.load(os.path.join(path, 'metadata.npz'))
        self.metadata = {k: v for k, v in self.metadata.items()}

        num_objs = (self.metadata['shape'][self.start_idx:self.end_idx] > 0).sum(1)
        num_available_views = self.metadata['camera_pos'].shape[1]
        if max_views is None:
            self.num_views = num_available_views
        else:
            assert(max_views <= num_available_views)
            self.num_views = max_views

        self.idxs = np.arange(self.start_idx, self.end_idx)[num_objs <= max_n]

        self.resize_reduced = torchvision.transforms.Compose([torchvision.transforms.Resize([self.h//target_reduced, self.w//target_reduced], antialias=False)])

        self.transform_d = torchvision.transforms.Compose([
            torchvision.transforms.Resize([self.h, self.w], antialias=False)
        ])
        print(f'Initialized MultiShapeNetEasy {mode} set, {len(self.idxs)} examples')

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return len(self.idxs) * self.num_views

    def __downsample__(self, array):
        return np.array(self.transform_d(torch.from_numpy(array)))

    def __getitem__(self, idx, noisy=True):
        scene_idx = idx % len(self.idxs)
        view_idx = idx // len(self.idxs)

        input_views = [idx // len(self.idxs)]
        target_views = np.array(list(set(range(3)) - set(input_views)))

        scene_idx = self.idxs[scene_idx]

        imgs = [np.asarray(imageio.imread(
            os.path.join(self.path, 'images', f'img_{scene_idx}_{v}.png')))
            for v in range(self.num_views)]
        depths = [np.asarray(imageio.imread(
            os.path.join(self.path, 'depths', f'depths_{scene_idx}_{v}.png')))
            for v in range(self.num_views)]
        imgs = np.stack([img[..., :3].astype(np.float32) / 255 for img in imgs])
        # Convert 16 bit integer depths to floating point numbers.
        # 0.025 is the normalization factor used while drawing the depthmaps.
        depths = np.stack([d.astype(np.float32) / (65536 * 0.025) for d in depths])

        metadata = {k: v[scene_idx] for (k, v) in self.metadata.items()}
        all_rays = []
        all_camera_pos = metadata['camera_pos'][:self.num_views]
        for i in range(self.num_views):
            cur_rays = get_camera_rays(all_camera_pos[i], noisy=False, height=self.h, width=self.w)
            all_rays.append(cur_rays)
        all_rays = np.stack(all_rays, 0).astype(np.float32)

        # For the shapenet dataset, the depth images represent the z-coordinate in camera space.
        # Here, we convert this into Euclidian depths.
        new_depths = []
        depths = self.__downsample__(depths)
        for i in range(self.num_views):
            new_depth = zs_to_depths(depths[i], all_rays[i], all_camera_pos[i])
            new_depths.append(new_depth)
        depths = np.stack(new_depths, 0)

        imgs = self.__downsample__(imgs.transpose(0,3,1,2))

        input_images = imgs[input_views]
        input_depth = depths[input_views]
        input_rays = all_rays[input_views]
        input_camera_pos = all_camera_pos[input_views]

        target_images =imgs[target_views]
        target_depth = depths[target_views]
        target_rays = all_rays[target_views]
        target_camera_pos = all_camera_pos[target_views]
        if self.mode != "train":
            masks = [np.asarray(imageio.imread(
                os.path.join(self.path, 'masks', f'masks_{scene_idx}_{v}.png')))
                for v in range(self.num_views)]
            masks = np.stack([d for d in masks])
            masks_cat = np.zeros(list(masks.shape)+[5], dtype=np.uint8)
            np.put_along_axis(masks_cat, masks[...,None], 1, axis=-1)
            masks_cat = self.__downsample__(masks_cat.transpose(0,3,1,2)).transpose(0,2,3,1)
            target_masks = masks_cat[target_views]
        else: 
            target_masks = np.array([0])

        if self.num_target_images==1:
            view = np.array(np.random.choice(2))
            target_images = target_images[view][None,...]
            target_depth = target_depth[view][None,...]
            target_rays = target_rays[view][None,...]
            target_camera_pos = target_camera_pos[view][None,...]

        target_vis_mask = np.ones_like(target_depth,dtype=np.bool)

        
        if self.target_reduced > 1:
            target_rays = np.asarray(self.resize_reduced(tensor(target_rays).permute(0,3,1,2)).permute(0,2,3,1))


        b, h, w, c = target_rays.shape
        target_shape = np.array([target_images.shape[0], self.h, self.w, 3],dtype=np.int32)
        target_pixels = np.reshape(target_images.transpose(0,2,3,1), (-1,3))
        target_depth = target_depth.flatten()
        target_rays = target_rays.reshape((-1,3))
        target_camera_pos = np.repeat(target_camera_pos[:,None], h * w, axis=1)
        target_camera_pos = target_camera_pos.reshape((-1, 3))
        target_vis_mask = target_vis_mask.flatten()

        if not self.full_scale:
            num_points = target_rays.shape[0]
            # If we have fewer points than we want, sample with replacement
            replace = num_points < self.points_per_item
            sampled_idxs = np.random.choice(np.arange(num_points),
                                            size=(self.points_per_item,),
                                            replace=replace)

            target_pixels = target_pixels[sampled_idxs]
            target_rays = target_rays[sampled_idxs]
            target_camera_pos = target_camera_pos[sampled_idxs]
            target_depth = target_depth[sampled_idxs]
            target_vis_mask = target_vis_mask[sampled_idxs]
        data = {
            'input_images':      input_images,              # [k, 3, h, w]
            'input_camera_pos':  input_camera_pos,                                    # [p, 3]
            'input_rays':        input_rays[:,:,:,:3],           # [k, h, w, 3]
            'input_depth':       input_depth,           # [k, h, w, 3]
            'target_pixels':     target_pixels,                                  # [p, 3]
            'target_images':     target_images,
            'target_camera_pos': target_camera_pos,                                    # [p, 3]
            'target_rays':       target_rays[...,:3],                                    # [p, 3]
            'target_depth':      target_depth,                                    # [p, 3]
            #'target_vis_mask':  target_vis_mask,
            'target_vis_mask':   target_vis_mask,
            'target_shape':      target_shape,
            'target_masks':      target_masks

        }
        return data 


def zs_to_depths(zs, rays, camera_pos):
    view_axis = -camera_pos
    view_axis = view_axis / np.linalg.norm(view_axis, axis=-1, keepdims=True)
    factors = np.einsum('...i,i->...', rays, view_axis)
    depths = zs / factors
    return depths

def get_camera_rays(c_pos, width=320, height=240, focal_length=0.035, sensor_width=0.032, noisy=False,
                    vertical=None, c_track_point=None):
    #c_pos = np.array((0., 0., 0.))
    # The camera is pointed at the origin
    if c_track_point is None:
        c_track_point = np.array((0., 0., 0.))

    if vertical is None:
        vertical = np.array((0., 0., 1.))

    c_dir = (c_track_point - c_pos)
    c_dir = c_dir / np.linalg.norm(c_dir)

    img_plane_center = c_pos + c_dir * focal_length

    # The horizontal axis of the camera sensor is horizontal (z=0) and orthogonal to the view axis
    img_plane_horizontal = np.cross(c_dir, vertical)
    #img_plane_horizontal = np.array((-c_dir[1]/c_dir[0], 1., 0.))
    img_plane_horizontal = img_plane_horizontal / np.linalg.norm(img_plane_horizontal)

    # The vertical axis is orthogonal to both the view axis and the horizontal axis
    img_plane_vertical = np.cross(c_dir, img_plane_horizontal)
    img_plane_vertical = img_plane_vertical / np.linalg.norm(img_plane_vertical)

    # Double check that everything is orthogonal
    def is_small(x, atol=1e-7):
        return abs(x) < atol

    assert(is_small(np.dot(img_plane_vertical, img_plane_horizontal)))
    assert(is_small(np.dot(img_plane_vertical, c_dir)))
    assert(is_small(np.dot(c_dir, img_plane_horizontal)))

    # Sensor height is implied by sensor width and aspect ratio
    sensor_height = (sensor_width / width) * height

    # Compute pixel boundaries
    horizontal_offsets = np.linspace(-1, 1, width+1) * sensor_width / 2
    vertical_offsets = np.linspace(-1, 1, height+1) * sensor_height / 2

    # Compute pixel centers
    horizontal_offsets = (horizontal_offsets[:-1] + horizontal_offsets[1:]) / 2
    vertical_offsets = (vertical_offsets[:-1] + vertical_offsets[1:]) / 2

    horizontal_offsets = np.repeat(np.reshape(horizontal_offsets, (1, width)), height, 0)
    vertical_offsets = np.repeat(np.reshape(vertical_offsets, (height, 1)), width, 1)

    if noisy:
        pixel_width = sensor_width / width
        pixel_height = sensor_height / height
        horizontal_offsets += (np.random.random((height, width)) - 0.5) * pixel_width
        vertical_offsets += (np.random.random((height, width)) - 0.5) * pixel_height

    horizontal_offsets = (np.reshape(horizontal_offsets, (height, width, 1)) *
                          np.reshape(img_plane_horizontal, (1, 1, 3)))
    vertical_offsets = (np.reshape(vertical_offsets, (height, width, 1)) *
                        np.reshape(img_plane_vertical, (1, 1, 3)))

    image_plane = horizontal_offsets + vertical_offsets

    image_plane = image_plane + np.reshape(img_plane_center, (1, 1, 3))
    c_pos_exp = np.reshape(c_pos, (1, 1, 3))
    rays = image_plane - c_pos_exp
    ray_norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / ray_norms
    return rays.astype(np.float32)