
from src.model import OSRT, SRT, DeFiNe
from fvcore.nn import FlopCountAnalysis
from src import data
from torch import nn
import torch
import yaml
import argparse
import numpy as np
import copy
import random
import math
import time

class GenericProfile(nn.Module):
    def __init__(self,model, device="gpu", render_depth=True):
        super().__init__()
        self.device = device

        self.encoder = model.encoder.to(self.device)
        self.decoder = model.decoder
        if isinstance(model, DeFiNe):
            self.decoder_rp_rgb = model.decoder_rp_rgb.to(self.device)
            self.decoder_att_rgb = model.decoder_att_rgb.to(self.device)
            if render_depth:
                self.decoder_rp_d = model.decoder_rp_d.to(self.device)
                self.decoder_att_d = model.decoder_att_d.to(self.device)
                self.d_max = model.d_max
                self.d_min = model.d_min
            else:
                self.decoder = self._decoder
        else:
            self.decoder_rp = model.decoder_rp.to(self.device)
            self.decoder_att = model.decoder_att.to(self.device)

    def _decoder(self, z, target_camera_pos, target_rays):

        extras = {}
        pred_pixels = self.decoder_rp_rgb(self.decoder_att_rgb(z, target_camera_pos, target_rays))
        if len(pred_pixels.shape)>3:
            pred_pixels = pred_pixels.flatten(1,3)

        return torch.sigmoid(pred_pixels), extras
    
    def forward(self, batch):        
        z = self.encoder(batch.get('input_images').to(self.device), batch.get("input_camera_pos").to(self.device), batch.get('input_rays').to(self.device))
        torch.cuda.reset_peak_memory_stats()
        pred_pixels, extras = self.decoder(z, batch.get("target_camera_pos" ).to(self.device), batch.get('target_rays').to(self.device))
        return pred_pixels, extras

    def time_interactive(self, batch, iter, rep, device="gpu"):
        with torch.no_grad():
            z = self.encoder(batch.get('input_images').to(device), batch.get("input_camera_pos").to(device), batch.get('input_rays').to(device))
            counter = 0.0
            times = torch.Tensor(size=[rep,iter])
            for i in range(rep):
                for j in range(iter):
                    t0 = time.perf_counter()
                    pred_pixels, extras = self.decoder(z, batch.get("target_camera_pos" ).to(device), batch.get('target_rays').to(device))
                    pred_pixels = pred_pixels.to("cpu")
                    for key, value in extras.items():
                        extras[key] = value.to("cpu") 
                    times[i,j]= time.perf_counter()-t0
        counter = times[2:].sum(1)
        stdev = counter.std()
        counter = counter.mean()
        return counter, stdev
        
def load_datasample(cfg, max_eval, batch_size=1):

    dataset = data.get_dataset('val', cfg['data'],
                                    max_len=max_eval, full_scale=True)

    flop_val = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=False, worker_init_fn=data.worker_init_fn)
    data_sample = next(iter(flop_val))  
    return data_sample

def load_dict(cfg, resolution, reduction, set_scannet_input=False):
    
    
    if cfg['data']['dataset'] == "scannet":
        cfg['data']["kwargs"]["setting"] = "default"
        cfg['data']["kwargs"]["num_input_images"] = 1
        cfg['data']["kwargs"]["num_target_images"] = 1
        cfg['data']["kwargs"]["virtual_cameras"] = False
        cfg['data']["kwargs"]["pose_jittering"] = False
        cfg['data']["kwargs"]["mask_non_vis"] = False
        cfg['data']["kwargs"]["discard_non_vis"] = False
        if set_scannet_input:
            cfg['data']["kwargs"]["h_rgb_in"]=resolution[0]
            cfg['data']["kwargs"]["w_rgb_in"]=resolution[1]
        cfg['data']["kwargs"]["h_rgb_tar"]=resolution[0]
        cfg['data']["kwargs"]["w_rgb_tar"]=resolution[1]

    elif cfg['data']['dataset'] == "msn_easy":
        cfg['data']["kwargs"]["num_target_images"] = 1
        cfg['data']["kwargs"]["h"]=resolution[0]
        cfg['data']["kwargs"]["w"]=resolution[1]
    
    decoder_dict = {}
    if cfg["model"]["decoder"] == "featurefield":
        for red in reduction:
            cfg_red = copy.deepcopy(cfg)
            
            cfg_red["data"]["kwargs"]["target_reduced"] = red
            cfg_red['model']['rp_kwargs']['h_in'] = int(resolution[0]/red)
            cfg_red['model']['rp_kwargs']['w_in'] = int(resolution[1]/red)
            cfg_red['model']['rp_kwargs']['h_out'] = resolution[0]
            cfg_red['model']['rp_kwargs']['w_out'] = resolution[1]
            cfg_red['model']['rp_kwargs']['upsample'] = int(math.log(red,2))
            decoder_dict["featurefield_"+str(red)] = cfg_red
    else:
        decoder_dict = {"lightfield": cfg}

    return decoder_dict


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 3D scene representation model.'
    )
    parser.add_argument('config_path', type=str, help='Path to config file.')
    parser.add_argument('--set-scannet-input', action='store_true', help='Train using distributed gpus.')
    parser.add_argument('--render-depth', action='store_true', help='Decode both RGB and Depth image in DeFiNe arch.')
    parser.add_argument('--srt', action='store_true', help='Evaluate SRT arch.')
    parser.add_argument('--max-eval', type=int, help='Limit the number of scenes in the evaluation set.')
    parser.add_argument('--distributed', action='store_true', help='Train using distributed gpus.')
    parser.add_argument('--flops', action='store_true', help='Profil flops')
    parser.add_argument('--time', action='store_true', help='Profil time')
    parser.add_argument('--frames', type=int, default=300, help='Frames processed to compute FPS.')
    parser.add_argument('--cuda', action='store_true', help='Run profile in GPU.')
    parser.add_argument('--save', action='store_true', help='Save results in csv file.')
    parser.add_argument('--batch_size', type=int, default=1, help='Dataloader batch size.')


    args = parser.parse_args()
    if  args.cuda and (not torch.cuda.is_available()):
        print("No GPU is available!")
    device = torch.device("cuda:0" if (args.cuda and torch.cuda.is_available()) else "cpu")
    print("Using device {}".format(device))

    with open(args.config_path+"/config.yaml", 'r') as f:
        base_cfg = yaml.load(f, Loader=yaml.CLoader)
    model_dict = dict()
    if base_cfg["model"]["base"]=="define":
        model_dict["define"] = [DeFiNe, [[128,192],[240,320],[480,640]], [4,8,16,32]]
    elif base_cfg["model"]["base"]=="srt":
        model_dict["srt"] = [SRT, [[60,80], [120,160], [240,320]], [2,4,8, 16]]
    elif base_cfg["model"]["base"]=="osrt":
        model_dict["osrt"] = [OSRT, [[120,160]],[4,8]]
    else:
        print("Wrong configuration file!")
        exit

    if args.save:
        with open(args.config_path+"/results.csv", 'a') as f:
                        f.writelines("\nArch, Decoder, h, w, GigaFLOPs, fps")
    for arch, item in model_dict.items():
        model_class = item[0]
        reduction = item[2]

        for res in item[1]:
            decoder_dict = load_dict(copy.deepcopy(base_cfg), res, reduction, args.set_scannet_input)
            
            for decoder, cfg in decoder_dict.items():
                data_sample = load_datasample(cfg, args.max_eval, batch_size=args.batch_size)

                model = model_class(cfg["model"])
                profile_model = GenericProfile(model, device=device, render_depth=args.render_depth).eval()

                time_interactive=math.inf
                flops = "NaN"
                stdev = "NaN"
                if args.flops:
                    flop_count = FlopCountAnalysis(profile_model, data_sample)
                    flops = flop_count.total()/1000**3
                    vRAM = torch.cuda.max_memory_allocated()/1024**3
                    torch.cuda.reset_peak_memory_stats()
                if args.time:
                    times = []
                    time_interactive, stdev = profile_model.time_interactive(data_sample, args.frames, 10, device)

                print("\nArch: {}; Decoder: {}; res: {}; GigaFLOPs: {}; render {} imgs: {} pm {}; FPS: {}; peak vRAM: {}".format(arch, decoder, res, flops, args.frames, time_interactive, stdev, args.frames/time_interactive, vRAM))
                if args.save:
                    with open(args.config_path+"/results.csv", 'a') as f:
                        f.writelines("\n{}, {}, {}, {}, {}, {}".format(arch, decoder, res[0], res[1], flops, args.frames/time_interactive))


