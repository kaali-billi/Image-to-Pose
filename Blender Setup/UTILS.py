""""
DESCRIPTION : Functions for Point Cloud Processing
            - exr2numpy() : Converts EXR file to Depth Map
            - pcd_block() : Converts Depth map to Point cloud with down-sampling
            - extract_data_Quat() : gives back Orientation in Quaternions from specific NPZ file
            - extract_data_RT() :  gives back Orientation in DCM from specific NPZ file
            - orient() : gives back down-sampled GT point Cloud using the extracted orientations
                         from quaternions, simple changes can be made to work with DCM
            - Plot_traj_compare() / Plot_traj() : Plotting 3D Trajectory
            - extract_cent_Quat() : gives back orientation respective of ISS Centroid
            - GPU_ICP_Q() : Performs GPU-ICP on Normalized IP and Reference Point Cloud
            - pc_norm() : Normalizes and returns point cloud and centroid optional
            - orient() : Given Quaternion and translation returns orientated reference point cloud
            - create_test() : given NPZ and EXR files, creates partial IP point cloud and GT complete point cloud
            - get_metrics() : given 2 Point clouds calculates EMD, ChamferL1 and ChamferL2
            - Euler_dif() : gives angular difference between 2 Quaternions (GT and Predicted)
            - Pick_points() : takes in pcd to allow interactive visualization
            - Trajectory() : Takes in relative translation and orientation to give back world trajectory
"""
## Scipy-Rotation Quaternion representation : [Qx,Qy,Qz,Qo] v/s GT_Quaternion Representations : [Qo, Qx,Qy,Qz]

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
import OpenEXR
import Imath
import array
import torch
import torch.nn as nn
import cupoch as cph
from extensions.emd import emd
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from time import time
import pyquaternion as pyq
import math
import plotly.graph_objects as go
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

# Initialize Loss Functions
EMD = emd().cuda()
CDL1 = ChamferDistanceL1().cuda()
CDL2 = ChamferDistanceL2().cuda()

# Origin Coordinate frame
def frame (size):
    mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=[0, 0, 0])
    return mesh_coord_frame

def Plot_traj_compare(x, y, z, x1, y1, z1):
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, name='GT Trajectory',
        marker=dict(
            size=6,
            color=z,
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=5
        )
    ), go.Scatter3d(
        x=x1, y=y1, z=z1, name='Reconstructed Trajectory',
        marker=dict(
            size=6,
            color=z,
            colorscale='inferno',
        ),
        line=dict(
            color='ivory',
            width=5
        )
    )])

    fig.update_layout(
        width=1920,
        height=1080,
        autosize=False,
        scene=dict(
            camera=dict(
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                eye=dict(
                    x=0,
                    y=1.0707,
                    z=1,
                )

            ),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube',
            xaxis=dict(nticks=4, range=[-500, 50], ),
            yaxis=dict(nticks=4, range=[-500, 50], )

        )
    )
    #fig.write_html('PLuM_Python/DC/Trajectory_Compare.html')
    return fig.show(renderer="browser")


def Plot_traj(x, y, z):
    fig = go.Figure(data=go.Scatter3d(
        x=x, y=y, z=z,
        marker=dict(
            size=4,
            color=z,
            colorscale='Viridis',
        ),
        line=dict(
            color='darkblue',
            width=2
        )
    ))

    fig.update_layout(
        width=1920,
        height=1080,
        autosize=False,
        scene=dict(
            camera=dict(
                up=dict(
                    x=0,
                    y=0,
                    z=1
                ),
                eye=dict(
                    x=0,
                    y=1.0707,
                    z=1,
                )

            ),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube',
            xaxis=dict(nticks=4, range=[-1000, 1000], ),
            yaxis=dict(nticks=4, range=[-1000, 1000], )

        )
    )

    return fig.show(renderer="browser")


def pc_norm(pc, ret_cen=False):
    """ pc: NxC, return NxC """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    if ret_cen:
        return pc, m, centroid
    else:
        return pc, m


def Trajectory(Q, T):
    q = pyq.Quaternion(Q)
    t = [0, T[0], T[1], T[2]]
    W = q * t * q.inverse
    return W[1], W[2], W[3]


def exr2numpy(exr, maxvalue=1.):
    """ converts 1-channel exr-data to 2D numpy arrays """
    file = OpenEXR.InputFile(exr)

    # Compute the size
    dw = file.header()['dataWindow']
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    (R) = [array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in "R"]

    # create numpy 2D-array
    img = np.zeros((sz[1], sz[0], 3), np.float64)

    # normalize
    data = np.array(R)
    data[data > maxvalue] = -1
    img = np.array(data).reshape(img.shape[0], -1)

    return img


def pcd_block(depth):
    pcd = []
    cx = 999 / 2
    cy = 999 / 2
    fx = 875.0
    fy = 875.0
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
    pc = pc.farthest_point_down_sample(2048)
    return pc


def extract_data_RT(path_RT):
    angles = []
    translation = []
    data = np.load(path_RT)
    if 'object_poses' in data.files:
        obj_poses = data['object_poses']

        for obj in obj_poses:
            obj_name = obj['name']
            obj_p = obj['pose']

            if obj_name == 'ISS_final_data':
                r = Rotation.from_matrix(obj_p[:3, :3])
                temp = r.as_euler("xyz", degrees=False)
                translation = obj_p[:, 3]
                angles = temp
    return angles, translation


def extract_data_Quat(path_quat):
    x = pyq.Quaternion([0, -1, 0, 0])
    obj_q = []
    translation = []
    data_q = np.load(path_quat)
    if 'location' in data_q.files:
        translations = data_q['location']
        for obj in translations:
            obj_name = obj['name']
            obj_loc = obj['location']
            if obj_name == 'JWST':
                translation = pyq.Quaternion([0, obj_loc[0], obj_loc[1], obj_loc[2]])
                T = x * translation * x.conjugate
                translation = np.array([T.x, T.y, T.z])

    if 'object_poses' in data_q.files:
        obj_poses = data_q['object_poses']
        for obj in obj_poses:
            obj_name = obj['name']
            obj_p = obj['pose']
            if obj_name == 'JWST':
                obj_q = pyq.Quaternion(obj_p)
                Q = x * obj_q
                obj_q = np.array([Q.w, Q.x, Q.y, Q.z])
    return obj_q, translation


def extract_data_PTR(path_quat, object):
    x = pyq.Quaternion([0, -1, 0, 0])
    obj_q = []
    translation = []
    data_q = np.load(path_quat)
    if 'location' in data_q.files:
        translations = data_q['location']
        for obj in translations:
            obj_name = obj['name']
            obj_loc = obj['location']
            if obj_name == object:
                translation = pyq.Quaternion([0, obj_loc[0], obj_loc[1], obj_loc[2]])
                T = x * translation * x.conjugate
                translation = np.array([T.x, T.y, T.z])

    if 'object_poses' in data_q.files:
        obj_poses = data_q['object_poses']
        for obj in obj_poses:
            obj_name = obj['name']
            obj_p = obj['pose']
            if obj_name == object:
                obj_q = pyq.Quaternion(obj_p)
                Q = x*obj_q
                obj_q = np.array([Q.w, Q.x, Q.y, Q.z])

    return obj_q, translation


def extract_cent_Quat(path_quat):
    x = pyq.Quaternion([0, -1, 0, 0])
    obj_q = []
    translation = []
    data_q = np.load(path_quat)
    if 'location' in data_q.files:
        translations = data_q['location']
        for obj in translations:
            obj_name = obj['name']
            obj_loc = obj['location']
            if obj_name == 'Dream Chaser':
                translation = pyq.Quaternion([0, obj_loc[0], obj_loc[1], obj_loc[2]])
                T = x * translation * x.conjugate
                translation = np.array([T.x, T.y, T.z])

    if 'object_poses' in data_q.files:
        obj_poses = data_q['object_poses']
        for obj in obj_poses:
            obj_name = obj['name']
            obj_p = obj['pose']
            if obj_name == 'Dream Chaser':
                obj_q = pyq.Quaternion(obj_p)
                Q = x * obj_q
                obj_q = np.array([Q.w, Q.x, Q.y, Q.z])

    return obj_q, translation


def orient(ort, translation, d_sample, pcd):
    #ISS = o3d.io.read_point_cloud(pcd)
    ISS = pcd
    T = ISS.get_rotation_matrix_from_quaternion(ort)
    ISS = ISS.translate([translation[0], #-3.42,  # X
                         translation[1], #-0.63,  # Y
                         translation[2]])#+1.87])  # Z
    ISS = ISS.rotate(T, center=translation)
    if len(ISS.points) != d_sample:
        ISS = ISS.farthest_point_down_sample(d_sample)
    ISS.paint_uniform_color([1, 0, 0])
    return ISS


def create_test_train(path_exr, path_npz, N, pcd, object):
    d_map = exr2numpy(path_exr, 1000)
    pc = pcd_block(d_map)
    ort, T = extract_data_PTR(path_npz, object)
    RSS = orient(ort, T, 13300, pcd)
    assert len(RSS.points) != 0
    # o3d.visualization.draw_geometries([RSS,pc,mesh_coord_frame])
    o3d.io.write_point_cloud('PLuM_Python/DC/GT/GT_{}.pcd'.format(str(N).zfill(4)), RSS)
    o3d.io.write_point_cloud('PLuM_Python/DC/TEST/{}.pcd'.format(str(N).zfill(4)), pc)
    return ort


def create_test_test(path_exr, path_npz, N, pcd, object):
    d_map = exr2numpy(path_exr, 1000)
    pc = pcd_block(d_map)
    ort, T = extract_data_PTR(path_npz, object)
    RSS = orient(ort, T, 13300, pcd)
    assert len(RSS.points) != 0
    # o3d.visualization.draw_geometries([RSS,pc,mesh_coord_frame])
    o3d.io.write_point_cloud('EUCLID_DATA/test_ada/GT/GT_{}.pcd'.format(str(N).zfill(4)), RSS)
    o3d.io.write_point_cloud('EUCLID_DATA/test_ada/ip/{}.pcd'.format(str(N).zfill(4)), pc)
    return ort


'''def create_test_test(path_exr, path_npz, N):
    d_map = exr2numpy(path_exr, 1000)
    pc = pcd_block(d_map, True)
    pts = np.array(pc.points)
    new_pts = []
    # Remove points behind the origin
    for coord in pts:
        x, y, z = coord
        if z != 1:
            new_pts.append(coord)

    N_pts = np.array(new_pts)
    pc.points = o3d.utility.Vector3dVector(N_pts)
    assert len(pc.points) != 0
    pc.paint_uniform_color([0, 0, 1])
    ort, T = extract_cent_Quat(path_npz)
    RSS = orient(ort, T, 13300)
    assert len(RSS.points) != 0
    o3d.io.write_point_cloud('JWST_TEST_030gt.pcd'.format(str(N).zfill(4)), RSS)
    o3d.io.write_point_cloud('JWST_TEST_030.pcd'.format(str(N).zfill(4)), pc)
'''


def create_test_par(Q, T, N):
    RSS = orient(Q, T, 2048)
    o3d.io.write_point_cloud('Space_Shuttle_data/PTR DATA/eval/{}.pcd'.format(str(N).zfill(4)), RSS)


def correct_scale(target, GT, cor_trans=False):
    Pt, St = pc_norm(target)
    if cor_trans and GT is not None:
        Pgt, Sgt = pc_norm(GT)
        target = Pt * Sgt + np.mean(GT, axis=0)
    else:
        Sgt = 68.74836
        target = Pt * Sgt + np.mean(target, axis=0)
    return target


def get_metrics(recon_file, gt_file):
    GT = o3d.io.read_point_cloud(gt_file)
    RE = o3d.io.read_point_cloud(recon_file)
    T_op = torch.from_numpy(np.array(RE.points)).type(torch.float32).cuda().unsqueeze(0)
    T_gt = torch.from_numpy(np.array(GT.points)).type(torch.float32).cuda().unsqueeze(0)
    EMD_loss = EMD(T_gt, T_op)
    L1_loss = CDL1(T_gt, T_op)
    L2_loss = CDL2(T_gt, T_op)
    return torch.sqrt(EMD_loss).detach().cpu().numpy(), L1_loss.detach().cpu().numpy(), L2_loss.detach().cpu().numpy()


def get_metrics_1(recon_points, gt_points):
    T_op = torch.from_numpy(recon_points).type(torch.float32).cuda()
    T_gt = torch.from_numpy(gt_points).type(torch.float32).cuda()
    EMD_loss = EMD(T_gt.unsqueeze(0), T_op.unsqueeze(0))
    L1_loss = CDL1(T_gt.unsqueeze(0), T_op.unsqueeze(0))
    L2_loss = CDL2(T_gt.unsqueeze(0), T_op.unsqueeze(0))
    return torch.sqrt(EMD_loss).detach().cpu().numpy(), L1_loss.detach().cpu().numpy(), L2_loss.detach().cpu().numpy()


def GPU_ICP_Q(target_gpu, init, i=0):
    source_gpu = cph.io.read_point_cloud('JWST_ICP_90.pcd')
    threshold = 1000
    if i != 0:
        trans_init = init
    else:
        trans_init = np.identity(4)
    start = time()
    reg_p2p = cph.registration.registration_icp(
        source_gpu,
        target_gpu,
        threshold,
        trans_init.astype(np.float32),
        cph.registration.TransformationEstimationPointToPoint(),
        cph.registration.ICPConvergenceCriteria(max_iteration=400)
    )
    time_taken = time() - start
    return reg_p2p.transformation, time_taken


def CPU_ICP_Q(target_cpu, source_cpu):
    threshold = 0.1
    trans_init = np.identity(4)
    start = time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_cpu,
        target_cpu,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    time_taken = time() - start
    return np.copy(reg_p2p.transformation), time_taken


def Euler_dif(Q, ort, flag=1):
    Q_icp = pyq.Quaternion([Q[3], Q[0], Q[1], Q[2]])
    if flag:
        Q_gt = pyq.Quaternion(ort)
    else:
        Q_gt = pyq.Quaternion(matrix=Rotation.from_euler("XYZ", ort,True).as_matrix())
    qd = Q_icp.conjugate * Q_gt
    phi = math.atan2(2 * (qd.w * qd.x + qd.y * qd.z), 1 - 2 * (qd.x ** 2 + qd.y ** 2))
    theta = math.asin(2 * (qd.w * qd.y - qd.z * qd.x))
    psi = math.atan2(2 * (qd.w * qd.z + qd.x * qd.y), 1 - 2 * (qd.y ** 2 + qd.z ** 2))
    return  np.rad2deg([phi,theta,psi])

def Euler_dif_deg(Q, ort):
    Q_icp = pyq.Quaternion([Q[3], Q[0], Q[1], Q[2]])
    Q_gt = pyq.Quaternion(ort)
    qd = Q_icp.conjugate * Q_gt
    return qd.degrees

def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("Press [shift + right click] to undo point picking")
    print("2) After picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def simplePCA(arr):
    '''

    :param arr: input array of shape shape[N,M]
    :return:
        mean - center of the multidimensional data,
        eigenvalues - scale,
        eigenvectors - direction
    '''

    # calculate mean
    m = np.mean(arr, axis=0)
    # center data
    arrm = arr - m

    # calculate the covariance, decompose eigenvectors and eigenvalues
    # M * vect = eigenval * vect
    # cov = M*M.T
    Cov = np.cov(arrm.T)
    eigval, eigvect = np.linalg.eig(Cov.T)

    # return mean, eigenvalues, eigenvectors
    return m, eigval, eigvect


def save_args_to_file(args, filename):
    with open(filename, 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")


def Model_size(net):
    param_size = 0
    for param in net.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in net.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print('model size: {:.3f}MB'.format(size_all_mb))


def play_motion(list_of_rec: [], list_of_ort: []):
    play_motion.vis = o3d.visualization.Visualizer()
    play_motion.index = 0

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
            time.sleep(0.1)
            vis.update_renderer()
        return False

    # Geometry of the initial frame
    ort = list_of_ort[0]
    rec = list_of_rec[0]
    # Initialize Visualizer and start animation callback
    vis = play_motion.vis
    vis.create_window(window_name='IP/RECON/ORT')
    vis.set_full_screen(True)
    # ctr = vis.get_view_control()
    # ctr.rotate(0, -50)
    vis.add_geometry(mesh_coord_frame)
    vis.add_geometry(ort)
    vis.add_geometry(rec)
    vis.register_animation_callback(forward)
    vis.run()
    vis.destroy_window()


class kabsch_torch(nn.Module):
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD.
    :param P: A Nx3 matrix of points
    :param Q: A Nx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """

    def __int__(self):
        super(kabsch_torch, self).__int__()

    def forward(self, P, Q):
        # Compute centroids
        centroid_P = torch.mean(P, dim=0)
        centroid_Q = torch.mean(Q, dim=0)
        # Optimal translation
        t = centroid_Q - centroid_P
        # Center the points
        p = P - centroid_P
        q = Q - centroid_Q

        # Compute the covariance matrix

        H = torch.matmul(p.transpose(0, 1).cuda(), q.cuda())
        # SVD
        U, S, Vt = torch.linalg.svd(H)
        # Validate right-handed coordinate system
        if torch.det(torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))) < 0.0:
            Vt[:, -1] *= -1.0
        # Optimal rotation
        R = torch.matmul(Vt.transpose(0, 1), U.transpose(0, 1))

        # RMSD
        # rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(0, 1)) - q)) / P.shape[0])

        # t = centroid_Q
        return R.detach().cpu().numpy(), t.detach().cpu().numpy()

def generate_gpu_voronoi(image_size=(1000, 1000), num_points=30):
    H, W = image_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate random seed points
    seed_points = torch.randint(0, min(H, W), (num_points, 2), device=device).float()

    # Create grid of all image coordinates
    y = torch.arange(H, device=device).view(-1, 1).repeat(1, W)
    x = torch.arange(W, device=device).repeat(H, 1)
    coords = torch.stack([x, y], dim=2).view(-1, 2).float()  # [H*W, 2]

    # Compute distances to each seed point [num_points, H*W]
    dists = torch.cdist(seed_points.unsqueeze(0), coords.unsqueeze(0)).squeeze(0)  # [num_points, H*W]

    # Assign each pixel to the nearest seed point
    labels = torch.argmin(dists, dim=0).view(H, W)  # [H, W]
    # Visualize
    labels, points = labels.cpu().numpy(), seed_points.cpu().numpy()
    plt.imshow(labels, cmap='tab20')
    plt.scatter(points[:, 0], points[:, 1], c='black', s=10)
    plt.title("GPU-accelerated Voronoi")
    plt.axis('off')
    plt.show()

