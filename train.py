import torch

import os
import argparse
import yaml

from src import data
from src.model import SRT, DeFiNe, ESRT, OSRT
from src.utils.visualizer import Visualizer

from src.model import SRT
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins import TorchSyncBatchNorm


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a 3D scene representation model.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--exit-after', type=int, help='Exit after this many training iterations.')
    parser.add_argument('--test', action='store_true', help='When evaluating, use test instead of validation split. Valid only for MSN_East dataset.')
    parser.add_argument('--dont-train', action='store_true', default=False, help='Do not start training.')
    parser.add_argument('--visnow', action='store_true', help='Run visualization.')
    parser.add_argument('--wandb', action='store_true', help='Log run to Weights and Biases.')
    parser.add_argument('--max-eval', type=int, help='Limit the number of scenes in the evaluation set.')
    parser.add_argument('--full-scale', action='store_true', help='Evaluate on full images.')
    parser.add_argument('--accumulate', type=int, default=1, help='Set cycles for gradient accumulation.')
    parser.add_argument('--float16', action='store_true', help='Train using float16 precision.')
    parser.add_argument('--tf32', action='store_true', help='Train using TF32 precision for all matrix multiplication. Available only on ampere GPUs.')
    parser.add_argument('--distributed', action='store_true', help='Train using distributed gpus.')
    parser.add_argument('--devices', type=int, default=1 , help='Set number of devices for distributed training')
    parser.add_argument('--new_id', action='store_true', help="If set overwrites checkpoint wandb_id")
    parser.add_argument('--gradient_clip', action="store_true" , help='Enable gradient clipping')
    parser.add_argument('--lpips', action="store_true" , help='Compute LPIPS metric in validation')
    parser.add_argument('--ssim', action="store_true" , help='Compute SSIM metric in validation')

    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.CLoader)
   
    if args.exit_after is not None:
        max_it = args.exit_after
    elif 'max_it' in cfg['training']:
        max_it = cfg['training']['max_it']
    else:
        max_it = 1000000

    out_dir = os.path.dirname(args.config)
    
    # Initialize dataset
    print('Loading training set...')
    if cfg["model"]["decoder"] == "featurefield":
        assert args.full_scale, "RayPatch decoder requires full scale images to train."

        # RayPatch decoder requires full scale images to train.
        # Standard decoders are trained without full scale images to see more views in the same batch; using full scale images requires far more memory.
        # Both models are evaluated in full scale.

    train_dataset = data.get_dataset('train', cfg['data'], full_scale=args.full_scale, distributed=args.distributed)

    eval_split = 'test' if args.test else 'val'
    print(f'Loading {eval_split} set...')
    cfg_val = cfg['data'].copy()
    if cfg['data']['dataset'] == "scannet":
        """Remove augmentation for validation if using Scannet dataset"""
        cfg_val["kwargs"] = cfg['data']["kwargs"].copy()
        cfg_val["kwargs"]["virtual_cameras"] = False
        cfg_val["kwargs"]["pose_jittering"] = False
        cfg_val["kwargs"]["mask_non_vis"] = False
        cfg_val["kwargs"]["discard_non_vis"] = False
    eval_dataset = data.get_dataset(eval_split, cfg_val,
                                    max_len=args.max_eval, full_scale=True,distributed=args.distributed)
    # Initialize data loaders
    batch_size = cfg['training']['batch_size']//args.devices 
    print("Batch size \n * Real: {} \n * Per device: {}".format(batch_size*args.devices, batch_size))
    
    train_sampler = val_sampler = None
    project = cfg["data"]["dataset"]
    num_workers = cfg['training'].get('num_workers',1)
    shuffle = False
    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        assert num_workers == 1, "Our MSN dataset is implemented as Tensorflow iterable, and does not currently support multiple PyTorch workers per process. Is also shouldn't need any, since Tensorflow uses multiple workers internally."
    else:
        shuffle = True
        print(f'Using {num_workers} workers per process for data loading.')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        sampler=train_sampler, shuffle=shuffle,
        worker_init_fn=data.worker_init_fn, persistent_workers=True)
        
    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=max(1,batch_size//16), num_workers=num_workers,
        sampler=val_sampler,
        pin_memory=True, worker_init_fn=data.worker_init_fn, persistent_workers=True)

    print('Data loaders initialized.')

    # Load model

    callbacks=[]
    plugins=[]
        
    if 'wandb' not in cfg['model']:
        cfg['model']['wandb'] = args.wandb
    if cfg["model"]["base"] == "define":
        model = DeFiNe
    elif cfg["model"]["base"] == "esrt":
        model = ESRT
    elif cfg["model"]["base"] == "osrt":
        model = OSRT
    else:
        assert cfg["model"]["base"] =="srt"
        model = SRT
        
    print('Model created.')

    if args.distributed:
        strategy= DDPStrategy(find_unused_parameters=False)
        decoder_norm = cfg['model']["rp_kwargs"].get("norm",None) if cfg['model'].get("rp_kwargs") is not None else None
        if cfg['model']["encoder_kwargs"].get("norm",None) == "Batch" or decoder_norm == "Batch":
            plugins += [TorchSyncBatchNorm()]
    else:
        strategy = "auto"

    # Try to automatically resume
    if os.path.exists(os.path.join(out_dir, f'model-v1.ckpt')):
        ckpt_path=os.path.join(out_dir, f'model-v1.ckpt')
        model = model.load_from_checkpoint(ckpt_path, cfg=cfg["model"], ssim=args.ssim)
        run_id = model.hparams.log_id
    elif os.path.exists(os.path.join(out_dir, f'model.ckpt')):
        ckpt_path=os.path.join(out_dir, f'model.ckpt')
        model = model.load_from_checkpoint(ckpt_path, cfg=cfg["model"], ssim=args.ssim)
        run_id = model.hparams.log_id
    else:
        model = model(cfg["model"], lpips=args.lpips, ssim=args.ssim)
        ckpt_path = None
        run_id = None
    if args.new_id:
        run_id = None

    logger = True #Default TensorBorad logger
    if args.wandb:
        if run_id is None:
            print(f'Sampled new wandb run_id.')
        else:
            print(f'Resuming wandb with existing run_id {run_id}.')
            
        logger = pl.loggers.WandbLogger(
            project=project,
            name=os.path.dirname(args.config).split("/")[-1],
            resume="allow",
            id=run_id,
            config=cfg 
        )
        #logger.watch(model, log="all", log_freq=1000)

    if isinstance(train_dataset, torch.utils.data.IterableDataset):
        ckpt_callback_backup = ModelCheckpoint(
            every_n_train_steps= cfg['training']['backup_every'],
            filename="model_backup",
            save_top_k=-1,
            dirpath=out_dir,
        )
        ckpt_callback_last = ModelCheckpoint(
            every_n_train_steps= cfg['training']['checkpoint_every'],
            filename="model",
            dirpath=out_dir,
            save_on_train_epoch_end=True
        )
    else:    
        ckpt_callback_backup = ModelCheckpoint(
            monitor="epoch",
            mode="max",
            every_n_epochs=max(1,int(cfg['training']['backup_every']/(len(train_loader)/args.devices))),
            filename="model_backup",
            save_top_k=-1,
            dirpath=out_dir,
        )
        ckpt_callback_last = ModelCheckpoint(
            monitor="epoch",
            mode="max",
            every_n_epochs=max(int(cfg['training']['checkpoint_every']/(len(train_loader)/args.devices)),1),
            filename="model",
            dirpath=out_dir,
            save_on_train_epoch_end=True
        )

    ckpt_callback_best = ModelCheckpoint(
        every_n_train_steps= cfg['training']['validate_every'],
        monitor=cfg['training']['model_selection_metric'],
        mode=cfg['training']['model_selection_mode'],
        filename="model_best",
        dirpath=out_dir,
        save_on_train_epoch_end=False
    )
    lr_monitor = LearningRateMonitor(
        logging_interval='step'
    )

    callbacks += [ckpt_callback_last,ckpt_callback_best,ckpt_callback_backup,lr_monitor]

    if args.gradient_clip:
        gradient_clip_val= 1.0
    else:
        gradient_clip_val= 0.0
    if args.float16:
        precision = 16
    else:
        precision = 32
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.visnow:
        # Loaders for visualization scenes
        train_vis_dataset = data.get_dataset('train', cfg['data'], full_scale=args.full_scale)
        vis_loader_val = torch.utils.data.DataLoader(
            eval_dataset, batch_size=12, shuffle=shuffle, worker_init_fn=data.worker_init_fn)
        vis_loader_train = torch.utils.data.DataLoader(
            train_vis_dataset, batch_size=12, shuffle=shuffle, worker_init_fn=data.worker_init_fn)
        print('Visualization data loaded.')
        data_vis_val = next(iter(vis_loader_val))  
        data_vis_train = next(iter(vis_loader_train)) 

        visualizer = Visualizer(model, cfg, train_dataset.render_kwargs, out_dir=out_dir)

        visualizer.visualize(data_vis_train, label = "train")
        visualizer.visualize(data_vis_val, label = "val", save_split=True)

    trainer = pl.Trainer(
        check_val_every_n_epoch=None, 
        val_check_interval=cfg['training']['validate_every']*args.accumulate,
        log_every_n_steps=cfg['training']['print_every'],
        default_root_dir=out_dir,
        max_steps=max_it,
        logger= logger,
        accumulate_grad_batches=args.accumulate,
        precision=precision,
        accelerator='auto',
        devices=args.devices,
        strategy=strategy,
        deterministic=False,
        benchmark=True,
        callbacks=callbacks,
        plugins=plugins,
        gradient_clip_val=gradient_clip_val
        )
        
    if not args.dont_train:
        import torch._dynamo
        torch._dynamo.config.verbose = True

        model_compiled = torch.compile(model, disable=True)
        trainer.fit(
            model=model_compiled,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=ckpt_path,
        )
    if args.wandb:
        logger.unwatch(model)
