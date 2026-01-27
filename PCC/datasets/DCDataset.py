import torch.utils.data as data
import numpy as np
import os, sys
import data_transforms as data_transforms
from .IO import IO
import random
import json
from .build import DATASETS
from utils.logger import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

@DATASETS.register_module()
class DC_DATA(data.Dataset):
    def __init__(self, config):
        self.partial_points_path = config.PARTIAL_POINTS_PATH
        self.complete_points_path = config.COMPLETE_POINTS_PATH
        self.category_file = config.CATEGORY_FILE_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
        self.file_list = self._get_file_list(self.subset)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        if subset == 'train':
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'RandomMirrorPoints',
                'objects': ['partial', 'gt']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])
        else:
            return data_transforms.Compose([{
                'callback': 'RandomSamplePoints',
                'parameters': {
                    'n_points': 2048
                },
                'objects': ['partial']
            }, {
                'callback': 'ToTensor',
                'objects': ['partial', 'gt']
            }])

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = (pc - centroid) / 5.73
        return pc, centroid

    def pc_norm_gt(self, pc, cen):
        pc = (pc - cen) / 5.73
        return pc


    def _get_file_list(self, subset):
        """Prepare file list for the dataset"""
        file_list = []

        for dc in self.dataset_categories:
            print_log('Collecting files of Taxonomy [ID=%s, Name=%s]' % (dc['taxonomy_id'], dc['taxonomy_name']),
                      logger='DCDATASET')
            samples = dc[subset]
            # Collecting Samples of Input and Ground Truth paths
            for s in samples:
                file_list.append({
                    'taxonomy_id':
                        dc['taxonomy_id'],
                    'model_id':
                        s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s)
                    ],
                    'gt_path':
                        self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log('Complete collecting files of the dataset. Total files: %d' % len(file_list), logger='DCDATASET')
        return file_list

    def __getitem__(self, idx):
        sample = self.file_list[idx]
        data = {}
        for ri in ['partial', 'gt']:
            file_path = str(sample['%s_path' % ri])
            # Reading and normalizing Inc. Point cloud and GT for model Input
            if ri == 'partial':
                data[ri], cen = self.pc_norm(IO.get(file_path[2:len(file_path)-2]).astype(np.float32))
                data['cen_par'] = cen
            else:
                data[ri]= self.pc_norm_gt(IO.get(file_path).astype(np.float32), cen)
        assert data['gt'].shape[0] == self.npoints
        if self.transforms is not None:
            data = self.transforms(data)
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        return len(self.file_list)