from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from easydict import EasyDict
import numpy as np
import os
import cv2
import torch
from mmengine.config import Config



class DC_dataset(Dataset):
    def __init__(self, cfg, mode):
        super().__init__()

        self.mode = mode
        self.cfg = cfg

        # prepare args
        args = {
            'dataset': 'Dream_Chaser',
            'use_right': False,
            'img_path': self.cfg.dataset.data_path,
            'mask_path': self.cfg.dataset.mask_path,
            'gt_path': self.cfg.dataset.gt_path,
            'img_path_eval': self.cfg.evaluation.data_path,
            'mask_path_eval': self.cfg.evaluation.mask_path,
            'gt_path_eval': self.cfg.evaluation.gt_path,
            'do_kb_crop': self.cfg.evaluation.do_kb_crop,
            'input_height': self.cfg.dataset.input_height,
            'input_width': self.cfg.dataset.input_width,
            'do_random_rotate': True,
            'degree': 1.0,
            'max_translation_x': 8
        }
        self.args = EasyDict(args)

        if self.mode == 'Val':
            self.img_list = sorted(os.listdir(self.args.img_path_eval))
            self.gt_list = sorted(os.listdir(self.args.gt_path_eval))
            self.mask_list = sorted(os.listdir(self.args.mask_path_eval))
        else:
            self.img_list = sorted(os.listdir(self.args.img_path))
            self.gt_list = sorted(os.listdir(self.args.gt_path))
            self.mask_list = sorted(os.listdir(self.args.mask_path))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.args.img_path, self.img_list[idx])
        mask_path = os.path.join(self.args.mask_path, self.mask_list[idx])
        gt_path = os.path.join(self.args.gt_path, self.gt_list[idx])

        image = torch.tensor(cv2.imread(img_path))
        image = image.permute(2, 0, 1)
        depth = torch.from_numpy(np.load(gt_path)) # / 256
        mask = torch.tensor(cv2.imread(mask_path))
        segm = mask[:, :, 0].unsqueeze(0) / 255  # Shape becomes (1, 1000, 1000)

        depth = depth.unsqueeze(0)

        sample = {'image': image.float(), 'mask': segm.float(), 'depth': depth.float()}

        return sample


'''cfg = Config.fromfile("configs/DC_PFF.yml")
print(cfg)
dataset = DC_dataset(cfg, "train")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for idx, sample in enumerate(dataloader):
    print(idx)
    img = sample['image']
    depth = sample['depth']
    mask = sample['mask']

    print(img.shape, mask.shape, depth.shape)
'''