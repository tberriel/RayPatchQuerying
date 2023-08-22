# decod the scanNet dataset, with 5 frame interval 
import glob 
import os 
import subprocess
import ipdb

def _read_split_file( filepath):
    '''
    Read data split txt file provided for Robust Vision
    ''' 
    with open(filepath) as f:
        trajs = f.readlines()
    trajs = [ x.strip() for x in trajs ]
    return trajs 

def decode_sens_file( in_out, cmd_prex='./sens' ):
    input_file = in_out[0]
    output_fldr = in_out[1]
    if not os.path.exists(output_fldr):
        os.makedirs( output_fldr)
    cmd = '%s %s %s'%(cmd_prex, input_file, output_fldr)
    subprocess.call(cmd, shell= True) 

def main():
    import argparse
    parser = argparse.ArgumentParser() 

    parser.add_argument('--dataset_path', required =True, type=str, 
        help='The path to the scannet dataset, suppose the data is organized as ${dataset_path}/scene####_##/')
    parser.add_argument('--output_path', required =True, type=str, 
        help='The path to the output folder path')
    parser.add_argument('--split_file', required =True, type=str, 
        help='The path to the split txt file ')
    parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
    parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
    parser.add_argument('--export_poses', dest='export_poses', action='store_true')
    parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
    parser.add_argument('--skip_frames', type=int, default=1, help='Interval between saved frames. If frames_list is set, this argument will be ignored.')
    parser.add_argument('--frames_list_path', type=str, default=None, help='Path to list of frames to save. If set, skip_frames argument will be ignored.')
    parser.set_defaults(export_depth_images=True, export_color_images=True, export_poses=True, export_intrinsics=True)

    args = parser.parse_args()
    extra_args = ""
    if args.export_depth_images:
        extra_args += '--export_depth_images '
    if args.export_color_images:
        extra_args += '--export_color_images '
    if args.export_poses:
        extra_args += '--export_poses '
    if args.export_intrinsics:
        extra_args += '--export_intrinsics '

    dataset_path = args.dataset_path
    output_base_path =  args.output_path 
    trajs = _read_split_file( args.split_file )
    
    #sample_fldrs = sorted(glob.glob('%s/scene*'%(dataset_path)))
    
    sample_fldrs = [ '%s/%s'%(dataset_path, traj) for traj in trajs ]
    
    if args.frames_list_path:
        scenes_dict=dict()
        with open(args.frames_list_path,'r') as f_out:
            for line in f_out:
                scene, frames = line.strip().split(":")
                frames = frames.replace(","," ")
                scenes_dict[scene] = frames
                
    for idx, sample_fldr in enumerate( sample_fldrs) :
        sample_idx = sample_fldr[-7:]
        if sample_idx not in scenes_dict.keys():
            continue
        sens_file_path = '%s/scene%s.sens'%(sample_fldr, sample_idx)
    
        # make dir 
        output_fldr = '%s/scene%s'%(output_base_path, sample_idx)
        if not os.path.exists( output_fldr):
            os.makedirs( output_fldr)
    
        # do the decoding 
        if args.frames_list_path:
            cmd = 'python ./reader.py --filename %s --output_path %s --frames_list %s %s'%(sens_file_path, output_fldr, scenes_dict[sample_idx], extra_args)
        else:
            cmd = 'python ./reader.py --filename %s --output_path %s --skip_frames %s %s'%(sens_file_path, output_fldr, args.skip_frames ,extra_args)
        #cmd = './sens %s %s'%(sens_file_path, output_fldr)
        print('traj %d of %d '%(idx, len(sample_fldrs)) )
        print('Decoding %s ...'%(sens_file_path) )
        subprocess.call(cmd, shell=True) 


if __name__== "__main__":
    main()

