import os
import random
import time
from os.path import basename, exists, join

import hydra
import pytorch_lightning as pl
import torch
from data_module import DataModule
from lightning_module import CodecLightningModule
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, FSDPStrategy
from torch.utils.data import DataLoader

seed = 1024
seed_everything(seed)


@hydra.main(config_path='config', config_name='default', version_base=None)
def train(cfg):
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.log_dir,
                                          save_top_k=-1,
                                          save_last=True,
                                          every_n_train_steps=20000,
                                          monitor='mel_loss',
                                          mode='min')

    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks = [checkpoint_callback, lr_monitor]

    datamodule = DataModule(cfg)
    lightning_module = CodecLightningModule(cfg)
    log_dir_name = os.path.basename(os.path.normpath(cfg.log_dir))
    wandb_logger = WandbLogger(
        project='xcodec2',  # 替换为您的项目名称
        name=log_dir_name,  # 替换为您的运行名称
        config=OmegaConf.to_container(cfg, resolve=True)  # 将 Hydra 配置转换为字典并传递
    )

    ckpt_path = None
    last_ckpt = os.path.join(cfg.log_dir, 'last.ckpt')
    if os.path.exists(last_ckpt):
        ckpt_path = last_ckpt
        print(f"Resuming from checkpoint: {ckpt_path}")
    else:
        print("No checkpoint found, starting training from scratch.")

    trainer = pl.Trainer(
        **cfg.train.trainer,
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=callbacks,
        logger=wandb_logger,
        profiler="simple",  # 启用 Profiler
        limit_train_batches=1.0 if not cfg.debug else 0.001)
    torch.backends.cudnn.benchmark = True
    # lightning_module.strict_loading = False
    # LightningModule.strict_loading = True
    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=ckpt_path)
    print(
        f'Training ends, best score: {checkpoint_callback.best_model_score}, ckpt path: {checkpoint_callback.best_model_path}'
    )


if __name__ == '__main__':
    train()
