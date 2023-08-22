import argparse
import os, sys

from SensorData import SensorData

# params
parser = argparse.ArgumentParser()
# data paths
parser.add_argument('--filename', required=True, help='path to sens file to read')
parser.add_argument('--output_path', required=True, help='path to output folder')
parser.add_argument('--export_depth_images', dest='export_depth_images', action='store_true')
parser.add_argument('--export_color_images', dest='export_color_images', action='store_true')
parser.add_argument('--export_poses', dest='export_poses', action='store_true')
parser.add_argument('--export_intrinsics', dest='export_intrinsics', action='store_true')
parser.add_argument('--skip_frames', type=int, default=1, help='Interval between saved frames. If frames_list is set, this argument will be ignored.')
parser.add_argument('--frames_list', nargs='+', type=int, default=None, help='List of saved frames. If set, skip_frames argument will be ignored.')
parser.set_defaults(export_depth_images=False, export_color_images=False, export_poses=False, export_intrinsics=False)

opt = parser.parse_args()
print(opt)


def main():
  if not os.path.exists(opt.output_path):
    os.makedirs(opt.output_path)
  # load the data
  sys.stdout.write('loading %s...' % opt.filename)
  sd = SensorData(opt.filename)
  sys.stdout.write('loaded!\n')
  if opt.export_depth_images:
    sd.export_depth_images(os.path.join(opt.output_path, 'depth'), frame_skip=opt.skip_frames, frames_list=opt.frames_list)
  if opt.export_color_images:
    sd.export_color_images(os.path.join(opt.output_path, 'color'), frame_skip=opt.skip_frames, frames_list=opt.frames_list)
  if opt.export_poses:
    sd.export_poses(os.path.join(opt.output_path, 'pose'), frame_skip=opt.skip_frames, frames_list=opt.frames_list)
  if opt.export_intrinsics:
    sd.export_intrinsics(os.path.join(opt.output_path, 'intrinsic'))


if __name__ == '__main__':
    main()
