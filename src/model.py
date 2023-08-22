from torch import nn
import torch
import pytorch_lightning as pl
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError, MeanAbsoluteError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from src.criterions import SquareRelativeDifference, AbsoluteRelativeDifference
from src.utils.common import compute_adjusted_rand_index
from src.encoder import SRTEncoder, DeFiNeEncoder, OSRTEncoder
from src.decoder import SRTDecoder, DeFiNeDecoder, RayPatchDecoder, SlotMixerDecoder
import time

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1, gamma=0.5):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    if gamma is None:
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
    else:
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                step = (current_step-num_warmup_steps)/(num_training_steps)
            return gamma**step

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

def get_step_schedule_with_warmup(optimizer, num_warmup_steps, epoch_its = 2945, epoch_fr = 80, last_epoch=-1, gamma=0.5):
    """
    Create a schedule with a learning rate that decreases by a factor every n steps, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            epoch = int(current_step/(epoch_its*epoch_fr))
        return gamma**epoch

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


class GenericModel(pl.LightningModule):
    def __init__(self, cfg, log_id = None, time_elapsed=0., lpips = False, ssim = False):
        super().__init__()

        self.perceptual_loss=cfg.get("perceptual_loss", False)
        if self.perceptual_loss:
            from lpips import LPIPS

            self.loss_colour=LPIPS(net='vgg', verbose=False)# LearnedPerceptualImagePatchSimilarity(net_type='vgg')

            for param in self.loss_colour.parameters():
                param.requires_grad=False
        else:
            self.loss_colour = MeanSquaredError()
        self.val_metrics = nn.ModuleDict({
            'psnr': PeakSignalNoiseRatio(data_range=1.0, dim=(1,2,3))
            })
        if lpips:
            self.val_metrics['lpips'] = LearnedPerceptualImagePatchSimilarity(net_type='vgg')# -> requires lpips library
        if ssim:
            self.val_metrics['ssim'] = StructuralSimilarityIndexMeasure(data_range=1.0) # -> large memory footprint
        self.hparams["lr"]=cfg["lr"]
        self.hparams["lr_config"]=cfg["lr_config"]
        self.hparams["log_id"] = log_id
        self.hparams["time_elapsed"] = time_elapsed
        self.t0 = 0
        self.save_hyperparameters()

    def on_train_start(self):
        if self.logger is not None:
            self.hparams["log_id"] =  self.logger.version
        return 
    """
    def on_before_batch_transfer(self, batch, dataloader_idx):
        for key, item in batch.items():
            batch[key] = torch.from_numpy(item)
        return batch
    """
    def on_train_batch_start(self, batch: dict, batch_idx: int) -> None:
        self.t0 = time.perf_counter()
        return 

    def on_train_batch_end(self, outputs, batch: dict, batch_idx: int) -> None:
        self.hparams["time_elapsed"] += time.perf_counter()-self.t0
        self.log("t",self.hparams["time_elapsed"])
        return 

    def forward_pass(self, batch):        
        z = self.encoder(batch.get('input_images').float(), batch.get("input_camera_pos").float(), batch.get('input_rays').float())
        pred_pixels, extras = self.decoder(z, batch.get("target_camera_pos" ), batch.get('target_rays'))
        return pred_pixels, extras

    def training_step(self, batch, batch_idx):
        pred_pixels, extras = self.forward_pass(batch)
        loss = self.compute_loss(pred_pixels, extras, batch) 
        return {"loss":loss}

    def compute_loss(self, pred_pixels, extras, batch, append_key = "", sync_dist=False): 

        vis_mask = batch.get("target_vis_mask")
        if torch.isnan(pred_pixels).sum()>0:
            raise "End training, network generated NaN values"

        n,h,w,c = map(int, batch.get("target_shape")[0].tolist())
        if self.perceptual_loss:
            #for param in self.loss_colour.parameters():
            #    print(param.requires_grad)  
            loss_rgb = self.loss_colour(pred_pixels.reshape((-1,h,w,c)).permute(0,3,1,2)*2.0-1.0,batch.get("target_pixels").reshape((-1,h,w,c)).permute(0,3,1,2)*2.0-1.0).mean()
            self.log("lpips"+append_key, loss_rgb,sync_dist=sync_dist)
        else:
            loss_rgb = self.loss_colour(pred_pixels[vis_mask], batch.get("target_pixels")[vis_mask])
            self.log("mse"+append_key, loss_rgb,sync_dist=sync_dist)

        loss = 0.+loss_rgb

        if "target_depth" in batch.keys() and "logdepth" in extras.keys():
            log_depth = extras["logdepth"][vis_mask]
            tar_depth = batch.get("target_depth").flatten()[vis_mask.flatten()]
            log_depth = log_depth[tar_depth>0].flatten()
            tar_depth = tar_depth[tar_depth>0].flatten()
            for key, metric in self.loss_depth.items():
                loss_d = metric(log_depth, tar_depth.log())
                loss += loss_d 
                self.log(key+append_key, loss_d,sync_dist=sync_dist)
        return loss

    def validation_step(self, batch, batch_idx):
        pred_pixels, extras = self.forward_pass(batch)
        n,h,w,c = map(int, batch.get("target_shape")[0].tolist())

        loss = self.compute_loss(pred_pixels, extras, batch, append_key="_val", sync_dist=True) 

        pbar = dict()
        tar_pixels = batch.get("target_pixels")
        for key, metric in self.val_metrics.items():
            pred = pred_pixels.reshape(-1,h,w,c).permute(0,3,1,2)
            tar = tar_pixels.reshape(-1,h,w,c).permute(0,3,1,2)
            if key == "lpips":
                # lpips expects values in [-1,1]
                pred = pred*2 - 1
                tar = tar*2 - 1
            if key =="ssim":
                pbar[key] = metric(pred.cpu(), tar.cpu()) 
            else:    
                pbar[key] =  metric(pred, tar)

            self.log(key,pbar[key],sync_dist=True)

        if 'logdepth' in extras.keys() and 'target_depth' in batch.keys():
            vis_mask = batch.get("target_vis_mask")
            log_depth = extras["logdepth"][vis_mask]
            tar_depth = batch.get("target_depth").flatten()[vis_mask.flatten()]
            log_depth = log_depth[tar_depth>0].flatten()
            tar_depth = tar_depth[tar_depth>0].flatten()
            for key, metric in self.val_metrics_depth.items():
                if key=="rmse_d":
                    pbar[key] =  metric(log_depth.exp(), tar_depth)
                else:
                    pbar[key] =  metric(log_depth, tar_depth.log())
                self.log(key,pbar[key],sync_dist=True)
        
        return {"loss": loss, "progress_bar": pbar}
    

    def test_step(self, batch, batch_idx):
        pred_pixels, extras = self.forward_pass(batch)
        n,h,w,c = map(int, batch.get("target_shape")[0].tolist())
        b = batch.get("target_shape").shape[0]

        loss = self.compute_loss(pred_pixels, extras, batch, append_key="_val", sync_dist=True) 

        pbar = dict()
        if batch.get("target_vis_mask").sum()>0:
            tar_pixels = batch.get("target_pixels")
            for key, metric in self.val_metrics.items():
                pred = pred_pixels.reshape(-1,h,w,c).permute(0,3,1,2)
                tar = tar_pixels.reshape(-1,h,w,c).permute(0,3,1,2)
                if key == "lpips":
                    # lpips expects values in [-1,1]
                    pred = pred*2 - 1
                    tar = tar*2 - 1
                if key =="ssim":
                    pbar[key] = metric(pred.cpu(), tar.cpu()) 
                else:    
                    pbar[key] =  metric(pred, tar)
                self.log(key,pbar[key],sync_dist=True)

            if 'logdepth' in extras.keys() and 'target_depth' in batch.keys():
                    vis_mask = batch.get("target_vis_mask")
                    log_depth = extras["logdepth"][vis_mask]
                    tar_depth = batch.get("target_depth").flatten()[vis_mask.flatten()]
                    log_depth = log_depth[tar_depth>0]
                    tar_depth = tar_depth[tar_depth>0]
                    for key, metric in self.val_metrics_depth.items():
                        if key=="rmse_d":
                            # Validation rmse, compute the root of the mean across all batch's pixels. It's not
                            # the correct way but it's simpler/faster and gives an intuition.
                            # Test rmse, computes the mean of the root of the mean across each image's pixels
                            h_d, w_d = h, w
                            tar_depth_t = batch.get("target_depth").reshape((b,-1,h_d,w_d))
                            sq_e = ((extras["logdepth"].exp().reshape((b,-1,h_d,w_d))-tar_depth_t))**2
                            mask = torch.logical_and(vis_mask.reshape((b,-1,h_d,w_d)),tar_depth_t>0)
                            weight = mask.sum((2,3))
                            rmse = torch.where(mask>0, sq_e/weight[...,None,None], mask*1.0).sum((2,3)).sqrt().mean(0) # Average only valid points
                            rmse_avg =0
                            weight = (weight*1.0).mean(0)
                            for i in range(rmse.shape[0]):
                                pbar[key+"_"+str(i)] = rmse[i]
                                self.log(key+"_"+str(i),pbar[key+"_"+str(i)],sync_dist=True)
                                rmse_avg += rmse[i]*weight[i]/weight.sum()
                            pbar[key+"_avg"] = rmse_avg
                            self.log(key+"_avg",pbar[key+"_avg"],sync_dist=True)
                        else:
                            pbar[key] =  metric(log_depth, tar_depth.log())
                            if torch.isnan(pbar[key]):
                                print("What")
                            self.log(key,pbar[key],sync_dist=True)    
        
        return {"loss": loss, "progress_bar": pbar}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
        if self.hparams["lr_config"] == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.cfg['lr_warmup'],
                num_training_steps=self.hparams.cfg['decay_it'],
                gamma=self.hparams.cfg.get('gamma', None)
            )
        elif self.hparams["lr_config"] == "step":
            scheduler = get_step_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.cfg['lr_warmup'],
                epoch_fr=self.hparams.cfg.get('epoch_fr', 40),
                epoch_its=self.hparams.cfg.get('epoch_its', 7875)
            )
        interval = "step"
        frequency = 1
        return [optimizer], [{"scheduler":scheduler, "interval":interval, "frequency":frequency, 'name':'lr'}]
    

class SRT(GenericModel):
    def __init__(self, cfg, log_id = None, time_elapsed=0., lpips = False, ssim = False):
        super().__init__(cfg, log_id, time_elapsed, lpips, ssim)
        self.encoder = SRTEncoder(**cfg['encoder_kwargs'])
        if cfg['decoder'] == 'featurefield':
            self.decoder_rp = RayPatchDecoder(**cfg['rp_kwargs'])
            out_dims_att = cfg['rp_kwargs'].get("conv_features", 128)
        else:
            self.decoder_rp = torch.nn.Identity()
            out_dims_att = cfg['decoder_kwargs'].get("out_dims", 3)

        cfg['decoder_kwargs']["out_dims"] = out_dims_att
        self.decoder_att = SRTDecoder( **cfg['decoder_kwargs'])

        if ['depth']:
            self.loss_depth = nn.ModuleDict({
                "abs_logd": MeanAbsoluteError(), 
                })

            self.val_metrics_depth = nn.ModuleDict({
                'absDiff_d': AbsoluteRelativeDifference(pred_log=True, target_log=True), 
                'squareDiff_d': SquareRelativeDifference(pred_log=True, target_log=True),
                'rmse_d': MeanSquaredError(squared=False)
                })

        self.save_hyperparameters()

    def decoder(self, z, target_camera_pos, target_rays, **kwargs):
        output = self.decoder_rp(self.decoder_att(z, target_camera_pos, target_rays, **kwargs))
        if len(output.shape)>3:
            output = output.flatten(1,3)
        out_dict = dict()
        if output.shape[2] == 4 :
            out_dict["logdepth"] = output[...,3]
            output = output[...,:3]
        return  torch.sigmoid(output), out_dict


class DeFiNe(GenericModel):
    def __init__(self, cfg, log_id = None, time_elapsed=0., lpips = False, ssim = False):
        super().__init__(cfg, log_id, time_elapsed, lpips, ssim)
        self.encoder = DeFiNeEncoder(**cfg['encoder_kwargs'])

        if cfg['decoder'] == 'featurefield':
            self.decoder_rp_rgb = RayPatchDecoder(out_dims=3, **cfg['rp_kwargs'])
            self.decoder_rp_d = RayPatchDecoder(out_dims=1, **cfg['rp_kwargs'])
            out_dims_att_d = out_dims_att_rgb = cfg['rp_kwargs'].get("conv_features", 128)
        else:
            self.decoder_rp_rgb = self.decoder_rp_d =torch.nn.Identity() 
            out_dims_att_d = 1
            out_dims_att_rgb=3 
        self.decoder_att_rgb = DeFiNeDecoder(**cfg['decoder_kwargs'], out_dims = out_dims_att_rgb)        
        self.decoder_att_d = DeFiNeDecoder(**cfg['decoder_kwargs'], out_dims = out_dims_att_d)
        self.scale_depth=True
        
        self.d_min = 0.1
        self.d_max = 10
        self.lambda_v=0.5
        self.lambda_s=5.0
        self.loss_depth = nn.ModuleDict({
            "abs_logd": MeanAbsoluteError(), 
            })

        self.val_metrics_depth = nn.ModuleDict({
            'absDiff_d': AbsoluteRelativeDifference(pred_log=True, target_log=True), 
            'squareDiff_d': SquareRelativeDifference(pred_log=True, target_log=True),
            'rmse_d': MeanSquaredError(squared=False)
            })
            
    def decoder(self, z, target_camera_pos, target_rays, **kwargs):
        extras = {}
        pred_pixels = self.decoder_rp_rgb(self.decoder_att_rgb(z, target_camera_pos, target_rays, **kwargs))
        extras["logdepth"] = (torch.sigmoid(self.decoder_rp_d(self.decoder_att_d(z, target_camera_pos, target_rays, **kwargs)))*(self.d_max-self.d_min)+self.d_min).log()
        if len(pred_pixels.shape)>3:
            pred_pixels = pred_pixels.flatten(1,3)
            extras["logdepth"] = extras["logdepth"].flatten(1)
        else:
            extras["logdepth"] = extras["logdepth"].squeeze(-1)

        return torch.sigmoid(pred_pixels), extras
        
    def compute_loss(self, pred_pixels, extras, batch, append_key = "", sync_dist=False): 

        real_mask = batch.get("target_real_mask")
        vis_mask = batch.get("target_vis_mask")
        mask_real = torch.logical_and(vis_mask,real_mask)
        mask_virt = torch.logical_and(vis_mask,~real_mask)
        n_real = mask_real.sum()
        n_virt = mask_virt.sum()
        if torch.isnan(pred_pixels).sum()>0:
            raise "End training, network generated NaN values"
        if self.perceptual_loss:
            n,h,w,c = map(int, batch.get("target_shape")[0].tolist())
            n_real_b = (real_mask*1.0).reshape((-1,h,w)).mean((1,-1)).bool()
            loss_rgb = self.loss_colour(pred_pixels.reshape((-1,h,w,c)).permute(0,3,1,2)*2.0-1.0, batch.get("target_pixels").reshape((-1,h,w,c)).permute(0,3,1,2)*2.0-1.0)
            self.log("lpips_r"+append_key, loss_rgb[n_real_b].mean(),sync_dist=sync_dist)
            loss = 0.+self.lambda_s*loss_rgb[n_real_b].mean()
            if (~n_real_b).sum() > 1:
                loss += self.lambda_s*self.lambda_v*loss_rgb[~n_real_b].mean()
        else:
            loss_mse_r = self.loss_colour(pred_pixels[mask_real], batch.get("target_pixels")[mask_real])
            loss_mse_v = self.loss_colour(pred_pixels[mask_virt], batch.get("target_pixels")[mask_virt])

            self.log("mse_r"+append_key, loss_mse_r,sync_dist=sync_dist)
            self.log("mse_v"+append_key, loss_mse_v,sync_dist=sync_dist)
            self.log("mse"+append_key, (loss_mse_r*n_real+loss_mse_v*n_virt)/(n_real+n_virt),sync_dist=sync_dist)

            loss = 0.+self.lambda_s*loss_mse_r + self.lambda_s*self.lambda_v*loss_mse_v

        if "target_depth" in batch.keys() and "logdepth" in extras.keys():
            loss_d_r = 0.0 
            loss_d_v = 0.0

            log_depth_r = extras["logdepth"][mask_real].flatten()
            tar_depth_r = batch.get("target_depth").flatten()[mask_real.flatten()]
            log_depth_v = extras["logdepth"][mask_virt].flatten()
            tar_depth_v = batch.get("target_depth").flatten()[mask_virt.flatten()]
            for key, metric in self.loss_depth.items():
                loss_d_r_temp = metric(log_depth_r[tar_depth_r>0], tar_depth_r[tar_depth_r>0].log())
                loss_d_v_temp = metric(log_depth_v[tar_depth_v>0], tar_depth_v[tar_depth_v>0].log())
                loss_d_r += loss_d_r_temp 
                loss_d_v += loss_d_v_temp
                self.log(key+"_r"+append_key, loss_d_r_temp,sync_dist=sync_dist)
                self.log(key+"_v"+append_key, loss_d_v_temp,sync_dist=sync_dist)
                self.log(key+append_key, (loss_d_r_temp*n_real+loss_d_v_temp*n_virt)/(n_real+n_virt),sync_dist=sync_dist)

            loss += loss_d_r +self.lambda_v*loss_d_v
            

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], betas=[0.9, 0.999], weight_decay=1e-4)
        if self.hparams["lr_config"] == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.cfg['lr_warmup'],
                num_training_steps=self.hparams.cfg['decay_it']
            )
        elif self.hparams["lr_config"] == "step":
            scheduler = get_step_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.cfg['lr_warmup'],
                epoch_fr=self.hparams.cfg.get('epoch_fr', 80),
                epoch_its=self.hparams.cfg.get('epoch_its', 2845)
            )
        interval = "step"
        frequency = 1
        return [optimizer], [{"scheduler":scheduler, "interval":interval, "frequency":frequency, 'name':'lr'}]

class ESRT(GenericModel):
    def __init__(self, cfg, log_id = None, time_elapsed=0., lpips = False, ssim = False):
        super().__init__(cfg, log_id, time_elapsed, lpips, ssim)
        self.encoder = ESRTEncoder(**cfg['encoder_kwargs'])
        self.decoder_rp = RayPatchDecoder(**cfg['rp_kwargs'])
        out_dims_att = cfg['rp_kwargs'].get("conv_features", 128)

        cfg['decoder_kwargs']["out_dims"] = out_dims_att
        self.decoder_att = SRTDecoder( **cfg['decoder_kwargs'])

        if ['depth']:
            self.loss_depth = nn.ModuleDict({
                "abs_logd": MeanAbsoluteError(), 
                })

            self.val_metrics_depth = nn.ModuleDict({
                'absDiff_d': AbsoluteRelativeDifference(pred_log=True, target_log=True), 
                'squareDiff_d': SquareRelativeDifference(pred_log=True, target_log=True),
                'rmse_d': MeanSquaredError(squared=False)
                })

        self.save_hyperparameters()

    def decoder(self, z, target_camera_pos, target_rays, **kwargs):
        output = self.decoder_rp(self.decoder_att(z, target_camera_pos, target_rays, **kwargs))
        if len(output.shape)>3:
            output = output.flatten(1,3)
        out_dict = dict()
        if output.shape[2] == 4 :
            out_dict["logdepth"] = output[...,3]
            output = output[...,:3]
        return  torch.sigmoid(output), out_dict


class OSRT(GenericModel):
    def __init__(self, cfg, log_id = None, time_elapsed=0., lpips = False, ssim = False):
        super().__init__(cfg, log_id, time_elapsed, lpips, ssim)
        self.encoder = OSRTEncoder(**cfg['encoder_kwargs'])

        if cfg['decoder'] == 'featurefield':
            self.decoder_rp = RayPatchDecoder(**cfg['rp_kwargs'])
            out_dims_att = cfg['rp_kwargs'].get("conv_features", 128)
        else:
            self.decoder_rp = torch.nn.Identity()
            out_dims_att=3 
        cfg['decoder_kwargs']["out_dims"] = out_dims_att
        self.decoder_att = SlotMixerDecoder( **cfg['decoder_kwargs'])

        self.save_hyperparameters()

    def decoder(self, z, target_camera_pos, target_rays, **kwargs):
        out_dict = dict()
        output, out_dict = self.decoder_att(z, target_camera_pos, target_rays)
        output = self.decoder_rp(output)
        if not self.training and isinstance(self.decoder_rp, RayPatchDecoder):
            out_dict["segmentation"] = torch.nn.functional.interpolate(out_dict["segmentation"].reshape([-1,self.decoder_rp.h_in, self.decoder_rp.w_in,5]).permute(0,3,1,2),size =output.shape[-3:-1], mode="bilinear").permute(0,2,3,1).reshape([-1,2,self.decoder_rp.h_out, self.decoder_rp.w_out,5])
        if len(output.shape)>3:
            output = output.flatten(1,3)
        if output.shape[2] == 4 :
            out_dict["logdepth"] = output[...,3]
            output = output[...,:3]
        return  torch.sigmoid(output), out_dict
    
    def test_step(self, batch, batch_idx):
        pred_pixels, extras = self.forward_pass(batch)
        n,h,w,c = map(int, batch.get("target_shape")[0].tolist())
        b = batch.get("target_shape").shape[0]

        loss = self.compute_loss(pred_pixels, extras, batch, append_key="_val", sync_dist=True) 

        pbar = dict()
        if batch.get("target_vis_mask").sum()>0:
            tar_pixels = batch.get("target_pixels")
            for key, metric in self.val_metrics.items():
                pred = pred_pixels.reshape(-1,h,w,c).permute(0,3,1,2)
                tar = tar_pixels.reshape(-1,h,w,c).permute(0,3,1,2)
                if key == "lpips":
                    # lpips expects values in [-1,1]
                    pred = pred*2 - 1
                    tar = tar*2 - 1
                if key =="ssim":
                    pbar[key] = metric(pred.cpu(), tar.cpu()) 
                else:    
                    pbar[key] =  metric(pred, tar)
                self.log(key,pbar[key],sync_dist=True)
        if isinstance(self.decoder_rp, RayPatchDecoder):
            pred_seg = extras['segmentation'].flatten(1,3)
        else:
            pred_seg = extras['segmentation']

        true_seg = batch['target_masks'].float().flatten(1,3)

        pbar['ari'] = compute_adjusted_rand_index(true_seg.transpose(1, 2),
                                                        pred_seg.transpose(1, 2)).mean()
        self.log('ari',pbar['ari'],sync_dist=True)

        pbar['fg_ari'] = compute_adjusted_rand_index(true_seg.transpose(1, 2)[:, 1:],
                                                            pred_seg.transpose(1, 2)).mean()
        self.log('fg_ari',pbar['fg_ari'],sync_dist=True)

        return {"loss": loss, "progress_bar": pbar}