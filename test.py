import os

import yaml
import argparse
import torch
import pytorch_lightning as pl

from src.utils.visualizer import Visualizer
from src.model import SRT, DeFiNe, OSRT, ESRT
from src import data
import numpy as np
import random

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    
    # Arguments
    parser = argparse.ArgumentParser(
        description='Evaluate a 3D scene representation model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--max-eval', type=int, help='Limit the number of scenes in the evaluation set. Valid only for MSN_East dataset.')
    parser.add_argument('--eval-split', type=str, default="val", help='Select the data split to evaluate on. Available splits are train, val and test. Test split valid only for MSN_Easy dataset.')
    parser.add_argument('--float16', action='store_true', help='Train using float16 precision.')
    parser.add_argument('--tf32', action='store_true', help='Train using TF32 precision for all matrix multiplication. Available only on ampere GPUs.')
    parser.add_argument('--full-scale', action='store_true', help='Evaluate on full images.')
    parser.add_argument('--arch', type=str, default='srt', help='Model base architecture. \'define\' for DeFiNe arch, else SRT will be loaded.')
    parser.add_argument('--vis', action='store_true', help='Save visualization.')
    parser.add_argument('--num_batches', type=int, default=8, help='Number of batches to visualize.')
    parser.add_argument('--lpips', action="store_true" , help='Compute LPIPS metric in validation')
    parser.add_argument('--ssim', action="store_true" , help='Compute SSIM metric in validation')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)

    if cfg["model"]["base"] == "define":
        model = DeFiNe
        batch_size = 2
        if args.eval_split is None:
            args.eval_split = "val"
    elif cfg["model"]["base"] == "esrt":
        model = ESRT
        batch_size = 64
        if args.eval_split is None:
            args.eval_split = "test"
    elif cfg["model"]["base"] == "osrt":
        model = OSRT
        batch_size = 8
        if args.eval_split is None:
            args.eval_split = "test"
    else:
        assert cfg["model"]["base"] == "srt"
        model = SRT
        batch_size = 2
        if args.eval_split is None:
            args.eval_split = "test"

    assert args.eval_split == "train" or args.eval_split == "val" or args.eval_split == "test", "Invalid split. Valid splits are train, val and test. Test split valid only for MSN_Easy dataset."
    print(f'Loading {args.eval_split} set...')
    cfg_val = cfg['data'].copy()
    if cfg['data']['dataset'] == "scannet":
        cfg_val["kwargs"] = cfg['data']["kwargs"].copy()
        cfg_val["kwargs"]["virtual_cameras"] = False
        cfg_val["kwargs"]["pose_jittering"] = False
        cfg_val["kwargs"]["mask_non_vis"] = False
        cfg_val["kwargs"]["discard_non_vis"] = False

    eval_dataset = data.get_dataset(args.eval_split, cfg_val,
                                    max_len=args.max_eval, full_scale=args.full_scale)

    num_workers = cfg['training'].get('num_workers',1)
    seed= 42
    pl.seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    val_loader = torch.utils.data.DataLoader(
        eval_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)

    print('Evaluation data loaded.')
    out_dir = os.path.dirname(args.config)

    assert os.path.exists(os.path.join(out_dir, f'model_best.ckpt')), "No  'model_best.ckpt' checkpoint found."
    ckpt_path=os.path.join(out_dir, f'model_best.ckpt')
    model = model.load_from_checkpoint(ckpt_path, cfg=cfg["model"], ssim=args.ssim)
    
    if args.vis:
        visualizer = Visualizer(model, cfg, eval_dataset.render_kwargs, out_dir=out_dir)

        for i, vis_batch in enumerate(iter(val_loader)):
            if i == args.num_batches:
                break
            visualizer.visualize(vis_batch, label = "val", save_split=True, batch=i)


    if args.float16:
        precision = 16
    else:
        precision = 32
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    trainer = pl.Trainer(
        log_every_n_steps=cfg['training']['print_every'],
        default_root_dir=out_dir,
        logger= False,
        precision=precision,
        accelerator='auto',
        #auto_select_gpus=True,
        deterministic=True,
        benchmark=False,
        )
    
    if args.lpips and "lpips" not in model.val_metrics.keys():
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        model.val_metrics["lpips"] = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    trainer.test(
        model,
        dataloaders=val_loader,
    )

