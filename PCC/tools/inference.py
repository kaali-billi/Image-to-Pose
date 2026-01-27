##############################################################
# % Author: Castle
# % Date:14/01/2023
###############################################################
import argparse
import os
import numpy as np
import cv2
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../'))
import builder
from utils.config import cfg_from_yaml_file
from utils import misc
from datasets.IO import IO
from datasets.data_transforms import Compose
import time
import open3d as o3d


def pc_norm(pc, ret_cen=False):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / 3.1122
    if ret_cen:
        return pc,m,centroid
    else:
        return pc, m

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='',
        help = 'yaml config file')
    parser.add_argument(
        '--checkpt',
        help = 'pretrained weight',
        default='')
    parser.add_argument('--pc_root', type=str, default='', help='Pc root')
    parser.add_argument('--gt_root', type=str, default='', help='Pc_GT root')
    parser.add_argument('--pc', type=str, default='', help='Pc file')
    parser.add_argument(
        '--save_vis_img',
        action='store_true',
        default=False,
        help='whether to save img of complete point cloud')
    '''parser.add_argument(
        '--out_pc_root',
        type=str,
        default='',
        help='root of the output pc file. '
        'Default not saving the visualization images.')'''
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    #assert args.save_vis_img or (args.out_pc_root != '')
    assert args.model_config is not None
    assert args.checkpt is not None
    assert (args.pc != '') or (args.pc_root != '')

    return args

def inference_single(model, pc_path, args, config, root=None):
    #GT = cph.io.read_point_cloud('ISS_NORM_ICP.pcd')
    if root is not None:
        pc_file = os.path.join(root, pc_path)
    else:
        pc_file = pc_path
    # read single point cloud


    pc_ndarray = IO.get(pc_file).astype(np.float32)


    # normalize it to fit the model on ISS_DATA
    pc_ndarray, m, centroid = pc_norm(pc_ndarray, True)



    transform = Compose([{
        'callback': 'UpSamplePoints',
        'parameters': {
            'n_points': 2048
        },
        'objects': ['input']
    }, {
        'callback': 'ToTensor',
        'objects': ['input']
    }])

    pc_ndarray_normalized = transform({'input': pc_ndarray})

    start = time.time()
    ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    end = time.time()

    # denormalize it to adapt for the original input
    dense_points = dense_points * 3.1122
    dense_points = dense_points + centroid
    print(end-start)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense_points))
    o3d.visualization.draw_geometries([pcd])
    #save_path = 'TEST_set\RECONSTRUCTED'
    #os.makedirs(save_path, exist_ok=True)
    #o3d.io.write_point_cloud(os.path.join(save_path, f'ISS_0000.pcd'), pcd)
    return

def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    print("Model Modules : ",base_model.modules())
    builder.load_model(base_model, args.checkpt)
    base_model.to(args.device.lower())
    base_model.eval()
    if args.pc_root != '':
        pc_file_list = os.listdir(args.pc_root)
        for pc_file in pc_file_list :
            inference_single(base_model, pc_file, args, config, root=args.pc_root)
    else:
        inference_single(base_model, args.pc, args, config)
if __name__ == '__main__':
    main()