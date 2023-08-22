import torch
import torchvision.transforms as T
import numpy as np
import math
import src.utils.visualize as vis
from src.utils import nerf
import pytorch_lightning as pl
import os
import matplotlib.pyplot as plt
from src.model import SRT, DeFiNe
from PIL import Image

class Visualizer():
    def __init__(self, model, cfg, render_kwargs, out_dir="", device="cpu"):
        super().__init__()
        if cfg["data"]["dataset"] == "msn_easy":
            #self.vis_fun = self.visualize_around
            self.vis_fun = self.visualize_target_views
        elif cfg["data"]["dataset"][:7] == "scannet":
            self.vis_fun = self.visualize_target_views
        elif cfg["data"]["dataset"] == "msn":
            #self.vis_fun = self.visualize_around
            self.vis_fun = self.visualize_target_views
        else:
            raise "Visualization function not implemented!"

        self.dataset = cfg["data"]["dataset"][:7]
        self.model = model
        self.cfg = cfg
        self.render_kwargs = render_kwargs
        self.out_dir = out_dir
        self.device = device
        self.reduced = cfg["data"]["kwargs"]["target_reduced"]

        if self.dataset == "scannet":
            self.denormalize_resnet = T.Compose([
                T.Normalize([0., 0., 0.],[1/0.229, 1/0.224, 1/0.225]),
                T.Normalize([-0.485, -0.456, -0.406],[1.,1.,1.]),
            ])
        
    def visualize(self, vis_data, label="", save_split=False, batch=0) -> None:
        data = dict()
        for key, value in vis_data.items():
            data[key] = value.to(self.device)
        self.vis_fun(self.model.to(self.device), data, label, save_split=save_split, batch=batch)
        return

    def visualize_around(self, model, data, label, save_split, batch=None):
        self.model.eval()
        with torch.no_grad():
            input_images = data.get('input_images')
            input_camera_pos = data.get('input_camera_pos')
            input_rays = data.get('input_rays')
            
            camera_pos_base = input_camera_pos[:, 0]
            input_rays_base = input_rays[:, 0]
            if self.reduced >1:
                #_, num_target_images, c, height, width = data.get('target_images').shape
                num_target_images,height,width,c = map(int, data.get("target_shape")[0].tolist())
                resize_reduced = T.Compose([T.Resize([height//self.reduced, width//self.reduced], antialias=False)])
                input_rays_base = resize_reduced(input_rays_base.permute(0,3,1,2)).permute(0,2,3,1)


            if 'transform' in data:
                # If the data is transformed in some different coordinate system, where
                # rotating around the z axis doesn't make sense, we first undo this transform,
                # then rotate, and then reapply it.
                
                transform = data['transform']
                inv_transform = torch.inverse(transform)
                camera_pos_base = nerf.transform_points_torch(camera_pos_base, inv_transform)
                input_rays_base = nerf.transform_points_torch(
                    input_rays_base, inv_transform.unsqueeze(1).unsqueeze(2), translate=False)
            else:
                transform = None

            input_images_np = np.transpose(input_images.cpu().numpy(), (0, 1, 3, 4, 2))

            z = model.encoder(input_images, input_camera_pos, input_rays)

            batch_size, num_input_images, height, width, _ = input_rays.shape

            num_angles = 6

            columns = []
            for i in range(num_input_images):
                header = 'input' if num_input_images == 1 else f'input {i+1}'
                columns.append((header, input_images_np[:, i], 'image'))

            all_extras = []
            for i in range(num_angles):
                angle = i * (2 * math.pi / num_angles)
                angle_deg = (i * 360) // num_angles

                camera_pos_rot = nerf.rotate_around_z_axis_torch(camera_pos_base, angle)
                rays_rot = nerf.rotate_around_z_axis_torch(input_rays_base, angle)

                if transform is not None:
                    camera_pos_rot = nerf.transform_points_torch(camera_pos_rot, transform)
                    rays_rot = nerf.transform_points_torch(
                        rays_rot, transform.unsqueeze(1).unsqueeze(2), translate=False)

                img, extras = self.render_image(z, camera_pos_rot, rays_rot, model,  **self.render_kwargs)
                all_extras.append(extras)
                columns.append((f'render {angle_deg}°', img.cpu().numpy(), 'image'))

            for i, extras in enumerate(all_extras):
                if 'depth' in extras:
                    depth_img = extras['depth'].unsqueeze(-1) / self.render_kwargs['max_dist']
                    depth_img = depth_img.view(batch_size, height, width, 1)
                    columns.append((f'depths {angle_deg}°', depth_img.cpu().numpy(), 'image'))

            output_img_path = os.path.join(self.out_dir, f'renders_around-{label}')
            vis.draw_visualization_grid(columns, output_img_path)
    
    def visualize_target_views(self, model, data, label, save_split, batch=0):
        self.model.eval()
        with torch.no_grad():
            input_images = data.get('input_images')
            input_camera_pos = data.get('input_camera_pos')
            input_rays = data.get('input_rays')
            
            batch_size, num_input_images, height, width, _ = input_rays.shape

            target_images = data.get('target_images')
            num_target_images,height,width,c = map(int, data.get("target_shape")[0].tolist())
            #_, num_target_images, c, height, width = target_images.shape
            target_camera_pos = data.get('target_camera_pos')
            target_camera_pos = target_camera_pos.reshape(batch_size, num_target_images, int(height/self.reduced), int(width/self.reduced), 3)
            target_rays = data.get('target_rays')
            target_rays_full = data.get('target_rays_full')
            target_depth =  data.get('target_depth')

            target_rays = target_rays.reshape(batch_size, num_target_images, int(height/self.reduced), int(width/self.reduced),3)
            if target_depth is not None:
                target_depth = target_depth.reshape(batch_size, num_target_images, height, width,1) 

            if self.dataset=="scannet":
                input_images_np = self.denormalize_resnet(input_images)
            else:
                input_images_np = input_images
            
            input_images_np = np.transpose(input_images_np.cpu().numpy(), (0, 1, 3, 4, 2))
            target_images_np = np.transpose(target_images.cpu().numpy(), (0, 1, 3, 4, 2))


            z = model.encoder(input_images, input_camera_pos, input_rays)

            batch_size, num_input_images, height, width, _ = input_rays.shape

            columns = []
            for i in range(num_input_images):
                header = 'input' if num_input_images == 1 else f'input {i+1}'
                columns.append((header, input_images_np[:, i], 'image'))
            for i in range(target_images.shape[1]):
                header = 'target' if target_images.shape[1] == 1 else f'target {i+1}'
                columns.append((header, target_images_np[:, i], 'image'))

            all_extras = []
            depth = []
            images = []

            cm = plt.get_cmap("jet")
            for i in range(num_target_images):
                img, extras = self.render_image(z, target_camera_pos[:,i], target_rays[:,i], model, **{'target_rays_full': target_rays_full}, **self.render_kwargs)
                columns.append((f'render target {i+1}', img.cpu().numpy(), 'image'))
                if 'logdepth' in extras:
                    depth_img = extras.pop('logdepth').squeeze(-1).exp()

                    depth.append(depth_img)
                    depth_img = cm(depth_img.cpu().numpy()/10.0)

                    gt_depth = cm(target_depth[:,i].cpu().numpy()/10.0)
                    
                    columns.append((f'GT depth target {i+1}', gt_depth.squeeze(-2), 'image'))
                    columns.append((f'depth rendered {i+1}', depth_img, 'image'))
                if 'segmentation' in extras:
                    pred_seg = extras['segmentation'].cpu()
                    columns.append((f'pred seg {i+1}', pred_seg.argmax(-1).numpy()+1, 'clustering'))
                all_extras.append(extras)

                images.append(img.cpu().numpy())

            output_img_path = os.path.join(self.out_dir, f'renders-{label+"_"+str(batch)}')
            vis.draw_visualization_grid(columns, output_img_path)

            if save_split:
                self.save_split(columns, batch=batch)


    def render_image(self, z, camera_pos, rays, model, **render_kwargs):
        """
        Args:
            z [n, k, c]: set structured latent variables
            camera_pos [n, 3]: camera position
            rays [n, h, w, 3]: ray directions
            render_kwargs: kwargs passed on to decoder
        """
        batch_size, height, width = rays.shape[:3]
        h_out, w_out =  height*self.reduced,  width*self.reduced
        if len(camera_pos.shape) != len(rays.shape):
            camera_pos = camera_pos.unsqueeze(1).repeat(1, height*width, 1)
        else:
            camera_pos = camera_pos.flatten(1, 2)
        
        rays = rays.flatten(1, 2)

        max_num_rays = self.cfg['data']['num_points'] * \
                self.cfg['training']['batch_size'] // rays.shape[0]
        num_rays = rays.shape[1]
        img = torch.zeros_like(rays)
        all_extras = []

        if isinstance(model, SRT) and self.cfg['model']['decoder'] == 'featurefield':
            if model.decoder_rp.resize_out:
                h_out = model.decoder_rp.h_out
                w_out = model.decoder_rp.w_out
        
        if self.cfg['model']['decoder'] == 'featurefield':
            img, agg_extras = model.decoder(
                z=z, target_camera_pos=camera_pos, target_rays=rays,
                **render_kwargs)
                
            for key in agg_extras:
                if key[:9] == "logdepth_":
                    continue
                agg_extras[key] = agg_extras[key].view(batch_size, h_out, w_out, -1)
        else:
            for i in range(0, num_rays, max_num_rays):
                img[:, i:i+max_num_rays], extras = model.decoder(
                    z=z, target_camera_pos=camera_pos[:, i:i+max_num_rays], target_rays=rays[:, i:i+max_num_rays],
                    **render_kwargs)
                    
                all_extras.append(extras)
            agg_extras = {}
            for key in all_extras[0]:
                agg_extras[key] = torch.cat([extras[key] for extras in all_extras], 1)
                agg_extras[key] = agg_extras[key].view(batch_size, height, width, -1)

        img = img.view(batch_size, h_out, w_out, 3)

        return img, agg_extras
        
    def save_split(self, columns, batch=0):
        len_batch = columns[0][1].shape[0]
        num_cols = len(columns)
        num_segments = 1
        for i in range(num_cols):
            column_type = columns[i][2]
            if column_type == 'clustering':
                num_segments = max(num_segments, columns[i][1].max())
        colors = vis.get_clustering_colors(num_segments)
        for i in range(len_batch):
            os.makedirs(self.out_dir+"/imgs",exist_ok=True)
            gen_string = self.out_dir+"/imgs/batch_"+str(batch)+"_sample_"+str(i)+"_"
            for j in range(num_cols):
                img_tmp = columns[j][1][i]
                if columns[j][2] == "clustering":
                    img_tmp = vis.visualize_2d_cluster(img_tmp, colors)
                img = Image.fromarray(np.uint8(img_tmp*255))
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                img.save(gen_string+columns[j][0].replace(' ','_')+".jpg")