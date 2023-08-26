import os
import imageio.v2 as imageio
import torch
import torchvision
import numpy as np
from multiprocessing import Pool
import argparse
    
def load_scene(scene_path, paths, pose="pose"):
    unfiltered_images = [np.asarray(imageio.imread(
        os.path.join(scene_path, 'color', i+".jpg"))) for i in paths]
    unfiltered_depth = [np.asarray(imageio.imread(
        os.path.join(scene_path, 'depth', i+".png"))) for i in paths]
    unfiltered_T_world_camera = [np.loadtxt(
        os.path.join(scene_path, pose, i+".txt"), delimiter=" ") for i in paths]
    intrinsic_mat = np.loadtxt(os.path.join(scene_path, 'intrinsic/intrinsic_depth.txt'), delimiter=" ").astype(np.float32)
    extrinsic_mat = np.loadtxt(os.path.join(scene_path, 'intrinsic/extrinsic_depth.txt'), delimiter=" ").astype(np.float32)
    intrinsic_rgb_mat = np.loadtxt(os.path.join(scene_path, 'intrinsic/intrinsic_color.txt'), delimiter=" ").astype(np.float32)

    corrupted_frames = []
    images = []
    depth = []
    T_world_camera = []
    path_indices = []

    for i,mat in enumerate(unfiltered_T_world_camera):
        a = np.isinf(mat)

        if a.sum() > 0:
            corrupted_frames.append(i)
        else:
            images.append(unfiltered_images[i])
            depth.append(unfiltered_depth[i])
            T_world_camera.append(unfiltered_T_world_camera[i])
            path_indices.append(i)   
    
    images = np.stack(images, 0).astype(np.float32)
    depth = np.stack(depth, 0).astype(np.float32)
    T_world_camera =  np.stack(T_world_camera, axis=0).astype(np.float32)
    
    return images, depth, T_world_camera, intrinsic_rgb_mat, intrinsic_mat, extrinsic_mat, path_indices

def process_scene(images, depth, intrinsic_rgb_mat, intrinsic_d_mat, h_rgb, w_rgb, h_d = 128, w_d = 192, downscale_depth=False):
    transform_rgb = torchvision.transforms.Compose([
        torchvision.transforms.Resize([h_rgb, w_rgb])
    ])

    images = torch.tensor(np.transpose(images, (0,3,1,2)))
    _, _, h_shape_v, h_shape_u = images.shape
    if intrinsic_rgb_mat[0,0] > 1000:
        h_shape_v = 968
        h_shape_u = 1296
    images = transform_rgb(images)
    images=images.permute(0,2,3,1)

    down_mat = np.eye(len(intrinsic_rgb_mat.diagonal()))
    down_mat[0,0] = w_rgb/h_shape_u
    down_mat[1,1] = h_rgb/h_shape_v
    intrinsic_rgb_mat = down_mat@intrinsic_rgb_mat

    depth = torch.tensor(depth)
    if downscale_depth:
        """DeFiNe does not downscale depth target"""
        transform_d = torchvision.transforms.Compose([
            torchvision.transforms.Resize([h_d, w_d])
        ])
        _, d_shape_v, d_shape_u = depth.shape
        depth = transform_d(depth)

        down_mat = np.eye(len(intrinsic_d_mat.diagonal()))
        down_mat[0,0] = w_d/d_shape_u
        down_mat[1,1] = h_d/d_shape_v
        intrinsic_d_mat = down_mat@intrinsic_d_mat
        
    return images.numpy(), depth.numpy(), intrinsic_rgb_mat, intrinsic_d_mat

def process(scene, data_path, proces_path, h_rgb, w_rgb, i =None): 
    print("Scene {}: {}".format(i, scene))
    scene_path = os.path.join(data_path, scene)
    proces_scene_path = os.path.join(proces_path, scene)
    os.makedirs(proces_scene_path, exist_ok=True)
    modes_list = os.listdir(scene_path)
    for mode in modes_list:
        proces_mode_path = os.path.join(proces_scene_path, mode)
        os.makedirs(proces_mode_path, exist_ok=True)

    tmp_paths =  os.listdir(scene_path+"/color")
    paths = []
    for  path in tmp_paths:
        path = int(path[:-4])
        paths.append(path)
    paths.sort()
    paths_tar = [str(x).zfill(4) for x in paths]
    paths = [str(x) for x in paths]

    images, depth, T_world_camera, intrinsic_rgb_mat, intrinsic_d_mat, extrinsic_d_mat, path_indices = load_scene(scene_path, paths)
    
    images, depth, intrinsic_rgb_mat, intrinsic_d_mat = process_scene(images, depth, intrinsic_rgb_mat, intrinsic_d_mat, h_rgb, w_rgb)
    T_world_camera= torch.tensor(T_world_camera)

    np.savetxt(proces_scene_path+"/intrinsic/intrinsic_depth.txt", intrinsic_d_mat)
    np.savetxt(proces_scene_path+"/intrinsic/intrinsic_color.txt", intrinsic_rgb_mat)
    np.savetxt(proces_scene_path+"/intrinsic/extrinsic_depth.txt", extrinsic_d_mat)
    for i, idx in enumerate(path_indices):
        imageio.imwrite(proces_scene_path+"/color/"+paths_tar[idx]+".jpg", images[i].astype(np.uint8))
        imageio.imwrite(proces_scene_path+"/depth/"+paths_tar[idx]+".png", depth[i].astype(np.uint16))
        np.savetxt(proces_scene_path+"/pose/"+paths_tar[idx]+".txt", T_world_camera[i])

def process_data(source_path, objective_path, h_rgb=480, w_rgb=640, parallel = True, num_cores = 12):
    for set_ in ["val","train"]:
        data_path = source_path+set_
        proces_path = objective_path+set_
        os.makedirs(proces_path, exist_ok=True)
        scene_list = os.listdir(data_path)
        if parallel:
            args = [(scene_list[i], data_path, proces_path, h_rgb, w_rgb, i) for i in range(len(scene_list))]
            with Pool(processes=num_cores) as pool:
                pool.starmap(process,args)
                pool.close()
        else:
            for i,scene in enumerate(scene_list):
                process(scene, data_path, proces_path, h_rgb, w_rgb, i=i)

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess ScanNet dataset.'
    )
    parser.add_argument('inPath', type=str, help='Path to input data file.')
    parser.add_argument('outPath', type=str, help='Path to output data file.')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel on multiple processors.')
    parser.add_argument('--num-cores', type=int, default=12, help='Number of processors for parallel execution.')
    parser.add_argument('--h', type=int, default=480, help='Output height for RGB data.')
    parser.add_argument('--w', type=int, default=640, help='Output widt for RGB data.')

    args = parser.parse_args()
    process_data(args.inPath, args.outPath, h_rgb = args.h, w_rgb=args.w, parallel=args.parallel, num_cores=args.num_cores)
