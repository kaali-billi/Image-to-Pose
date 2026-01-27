import os.path as osp
import warnings
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Subset
from DataLoader_DC import DC_dataset

try:
    from mmcv import Config
except ImportError:
    from mmengine import Config

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
import torch
from dataloaders import DATAMODULES
from models import MODELS
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        'config_name',
        type=str,
        help='The name of configuration file.'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1
    )

    return parser.parse_args()


def main():
    warnings.filterwarnings(action='ignore')
    print(torch.cuda.is_available())  # Should be True
    import sys
    print(sys.path)
    # configurations
    arg = parse_args()
    cfg = Config.fromfile(osp.join('configs/', f'{arg.config_name}.yml'))

    # set random seed
    seed_everything(cfg.training.seed)
    dataset = DC_dataset(cfg, "train")
    total_size = len(dataset)
    indices = np.random.choice(total_size, size=750, replace=False)

    # Create subset
    subset = Subset(dataset, indices)

    # Create dataloader with subset
    data = DataLoader(subset, batch_size=cfg.training.batch_size, shuffle=True)

    print(f"Original dataset size: {total_size}")
    print(f"Subset size: {len(subset)}")
    #data = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)

    # model
    model = MODELS.build({
        'type': cfg.model.type,
        'cfg': cfg
    })
    print(f'Training with model {type(model).__name__}...')

    # get resume path
    resume_path = torch.load('Weights/LRO_4/LRO_ARP_NEW_DC.pth',map_location='cuda')   #cfg.training.resume_from
    weights = resume_path['model_state_dict']

    if resume_path is not None:
        print(f'The training is resumed from resume_path.')
    else:
        print(f'The training is from scratch.')
    # define checkpoint configurations'''
    work_dir = osp.join(cfg.training.work_dir, arg.config_name)
    save_dir = osp.join(cfg.training.save_dir, arg.config_name)
    save_path = osp.join(save_dir, f'{arg.config_name}_saved.yml')
    cfg.dump(save_path)

    checkpoint_callback = ModelCheckpoint(
        dirpath=work_dir,
        every_n_epochs=cfg.evaluation.every_n_epochs,
        monitor='val/rms',
        mode='min',
        save_weights_only=False,
        save_top_k=cfg.training.get('save_top_k', 3)
    )
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # define trainer
    trainer = Trainer(
        precision=cfg.training.precision,
        accelerator='gpu',
        default_root_dir=work_dir,
        devices=arg.gpus,
        max_epochs=cfg.training.max_epochs,
        check_val_every_n_epoch=cfg.evaluation.every_n_epochs,
        callbacks=[checkpoint_callback],
        sync_batchnorm=(arg.gpus > 1),
        num_nodes=1,
        gradient_clip_val=cfg.training.gradient_clip_val,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        amp_backend="native",
    )

    # training
    model.load_state_dict(weights)
    trainer.fit(model, data)
    torch.save({
        'model_state_dict': model.state_dict()}, osp.join(save_dir, "LRO_ARP_NEW_DC.pth"))  # training done.
    save_path = osp.join(save_dir, f'{arg.config_name}_saved.yml')
    cfg.dump(save_path)
    print('The training is done.')


if __name__ == '__main__':
    main()
