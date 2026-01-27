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
import time
from utils import misc
from datasets.IO import IO
from datasets.data_transforms import Compose
from tqdm import tqdm
import torch
from scipy.spatial.transform import Rotation
import open3d as o3d
import cupoch as cph
import numpy as np
from project_pointcloud_image import plt, project_pointcloud_to_image


def pc_norm(pc, sf, ret_cen=False):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    if ret_cen:
        return pc, m, centroid
    else:
        return pc, m


def play_motion(list_of_rec: [], list_of_ort: [], save_dir):
    play_motion.vis = o3d.visualization.Visualizer()
    play_motion.index = 0
    os.makedirs(save_dir, exist_ok=True)
    # Save all point clouds first
    print("Saving point clouds...")
    for i, rec_pcd in enumerate(list_of_rec):
        o3d.io.write_point_cloud(os.path.join(save_dir, 'REC_{}.pcd'.format(str(i).zfill(4))), rec_pcd)
    print(f"Saved {len(list_of_rec)} frames to {save_dir}/")
    def forward(vis):
        pm = play_motion
        if pm.index < len(list_of_rec) - 1:
            # print(pm.index)
            pm.index += 1
            rec.points = list_of_rec[pm.index].points
            ort.points = list_of_ort[pm.index].points
            ort.paint_uniform_color([0, 1, 0])
            rec.paint_uniform_color([1, 0, 0])
            vis.update_geometry(ort)
            vis.update_geometry(rec)
            vis.poll_events()
            time.sleep(0.05)
            vis.update_renderer()
        return False

    # Geometry of the initial frame
    ort = list_of_ort[0]
    rec = list_of_rec[0]
    # Initialize Visualizer and start animation callback
    vis = play_motion.vis
    vis.create_window(window_name='IP/RECON/ORT')
    vis.set_full_screen(True)
    #ctr = vis.get_view_control()
    # ctr.rotate(0, -50)
    vis.add_geometry(mesh_coord_frame)
    vis.add_geometry(ort)
    vis.add_geometry(rec)
    vis.register_animation_callback(forward)
    vis.run()
    vis.destroy_window()


mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default='',
                        help='yaml config file')
    parser.add_argument(
        '--checkpt',
        help='pretrained weight')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert args.model_config is not None
    assert args.checkpt is not None

    return args


def pcd_block(depth):
    scale_x = 0.64
    scale_y = 0.48
    pcd = []
    cx = (999 / 2)  # * scale_x
    cy = (999 / 2)  # * scale_y
    fx = 500.0  # * scale_x
    fy = 500.0  # * scale_y
    height, width = depth.shape
    for i in range(height):
        for j in range(width):
            z = depth[i][j]
            x = (j - cx) * z / fx
            y = (i - cy) * z / fy

            pcd.append([x, y, z])
    pc_EXT = o3d.geometry.PointCloud()  # create point cloud object
    pc_EXT.points = o3d.utility.Vector3dVector(pcd)
    R = pc_EXT.get_rotation_matrix_from_xyz((1 * np.pi, 0, 0))
    pc_EXT = pc_EXT.rotate(R, center=(0, 0, 0))
    abc = []
    for coord in np.array(pc_EXT.points):
        x, y, z = coord
        if z < 0:
            abc.append(coord)
    pc = o3d.geometry.PointCloud()  # create point cloud object
    pc.points = o3d.utility.Vector3dVector(abc)
    if len(pc.points) > 2048:
        pc = pc.farthest_point_down_sample(2048)
    return pc


def deploy(model, args, config):
    path_par = f'../SIM_DC_TEST/FT_PC/'
    save_path = f'../SIM_DC_TEST/FT_REC/'
    print(os.path.exists(path_par), os.path.exists(save_path))
    sf = config.scaling_factor
    rec = []
    ip = []
    #print(sf)
    tt = []
    for f in tqdm(os.listdir(path_par)):
        pc_file = os.path.join(path_par, f)
        IP = o3d.io.read_point_cloud(pc_file)
        #IP = pcd_block(np.load(pc_file))
        IP.paint_uniform_color([0, 0, 0])
        #pc_ndarray = IO.get(pc_file).astype(np.float32)
        pc_ndarray, m, centroid = pc_norm(np.array(IP.points), sf, True)
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
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Start timer
        start_event.record()
        pc_ndarray_normalized = transform({'input': pc_ndarray})
        ret = model(pc_ndarray_normalized['input'].unsqueeze(0).to(args.device.lower()))
        dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
        end_event.record()

        # Wait for everything to finish running
        torch.cuda.synchronize()
        tt.append(start_event.elapsed_time(end_event))

        # denormalize it to adapt for the original input
        dense_points = dense_points * sf
        dense_points = dense_points + centroid
        recon_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense_points))
        recon_pc = recon_pc.farthest_point_down_sample(2048)
        #print(len(recon_pc.points))
        recon_pc.paint_uniform_color([1, 0, 0])
        rec.append(recon_pc)
        ip.append(IP)



    tt = np.array(tt)
    play_motion(rec, ip, save_path)


    print("Average, Mean Time taken", np.average(tt), np.mean(tt))

    return


from fvcore.nn import FlopCountAnalysis, parameter_count


def main():
    args = get_args()

    # init config
    config = cfg_from_yaml_file(args.model_config)
    # build model
    base_model = builder.model_builder(config.model)
    print("Model Modules : ", base_model.modules())
    builder.load_model(base_model, args.checkpt)
    base_model.to(args.device.lower())
    base_model.eval()

    '''input_shape = (1, 2048, 3)  # (batch, num_points, xyz)
    dummy_input = torch.randn(input_shape).cuda()

    # Calculate FLOPs
    flops = FlopCountAnalysis(base_model, dummy_input)
    print(f"Total FLOPs: {flops.total():,}")
    print(f"FLOPs in GFLOPs: {flops.total() / 1e9:.2f}")

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")'''
    deploy(base_model, args, config)


def get_model(model_config, wts):
    config = cfg_from_yaml_file(model_config)
    # build model
    base_model = builder.model_builder(config.model)
    print("Model Modules : ", base_model.modules())
    builder.load_model(base_model, wts)
    base_model.to('cuda')
    base_model.eval()
    return base_model


if __name__ == '__main__':
    main()
