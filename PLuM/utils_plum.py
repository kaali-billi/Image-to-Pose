import numpy as np
import math
import open3d as o3d
import torch
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation
import time
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import pyquaternion as pyq
import plotly.graph_objects as go
from scipy.spatial import distance_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ = "XYZ"


def Trajectory(Q, T):
    q = pyq.Quaternion(Q)
    t = [0, T[0], T[1], T[2]]
    W = q * t * q.inverse
    return W[1], W[2], W[3]


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
    fig.write_html('PLuM_Python/LRO_test/Trajectory_Compare.html')
    return fig.show(renderer="browser")


def read_params(file):
    parameters = {}

    # Open and read the file
    with open(file, 'r') as file:
        for line in file:
            # Split the line into key and value
            key, value = line.split(':', 1)
            key = key.strip()  # Remove leading/trailing spaces
            value = value.strip()  # Remove leading/trailing spaces

            # Handle different types of values
            if ',' in value:  # If the value contains a comma, assume it's a list of floats
                parameters[key] = np.array([float(x) for x in value.split(',')])
            elif value.replace('.', '', 1).isdigit():  # Single numeric value
                parameters[key] = float(value)
            else:  # Assume it's a string
                parameters[key] = value
    return parameters


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


def compute_diameter_open3d_accurate(point_cloud):
    """
    Computes the diameter of a 3D object with high accuracy using Open3D and NumPy/Scipy.
    :param point_cloud: Open3D PointCloud object.
    :return: Diameter (float)
    """
    # Convert Open3D PointCloud to NumPy array
    points = np.asarray(point_cloud.points)

    # Compute pairwise distances using a distance matrix
    dist_matrix = distance_matrix(points, points)

    # Find the maximum distance (diameter)
    diameter = np.max(dist_matrix)
    return diameter


def homogeneous_intrinsic(ROLL, PITCH, YAW, X, Y, Z):
    # EXTRINSIC ROTATION "xyz"
    # Open3d.get_rotation_matrix_from_xyz() : INTRINSIC ROTATION "XYZ" for QUAT as well
    T = np.eye(4, 4)
    T[:3, :3] = Rotation.from_euler(SEQ, [ROLL, PITCH, YAW], degrees=True).as_matrix()
    T[:, 3] = [X, Y, Z, 1]
    # T = torch.tensor(T,device=device)
    return T


def R_Intrinsic(roll, pitch, yaw):
    g = Rotation.from_euler(SEQ, [roll, pitch, yaw], degrees=True).as_matrix()
    return g


def hom2eul(T):
    roll = math.atan2(T[2, 1], T[2, 2])
    pitch = math.asin(-T[2, 0])
    yaw = math.atan2(T[1, 0], T[0, 0])
    x = T[0, 3]
    y = T[1, 3]
    z = T[2, 3]

    return [roll, pitch, yaw, x, y, z]


def rescale_pc(pts, cen):
    pts = pts + cen
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    return pc


def vis_PF_SA(TT, iteration, g, hyp, maxElement, cen, pcd_file, gt, ret):

    pose = homogeneous_intrinsic(hyp[0], hyp[1], hyp[2], cen[0], cen[1], cen[2])
    ort = o3d.io.read_point_cloud(gt)
    pcd = pcd_file
    if len(ort.points) > len(pcd.points):
        ort = ort.farthest_point_down_sample(len(pcd.points))
    ort = ort.transform(pose)

    ort.paint_uniform_color([0, 0, 0])
    pcd.paint_uniform_color([1, 0, 0])
    #pdists = np.array(pcd.compute_point_cloud_distance(ort))
    if ret:
        return ort
    else:
        print("Time Taken : ", TT)
        print("Iter, Counter : ", iteration, g)
        print("roll(deg), pitch(deg), yaw(deg), evidence, Rev/point")
        print(f"{hyp},"
              f"{maxElement},"
              f"{maxElement / len(pcd_file.points)},"
              )
        o3d.visualization.draw_geometries([ort, pcd, mesh_coord_frame])
        return None


def save_args_to_file(args, filename):
    with open(filename, 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")


def read_pc(file, n=1024, sf = 0):
    """
    LRO_SCALE = 2.2568
    DC_SCALE = 5.73 somehwo working better without rescaling , maybe add a threshold close to 95% or something ?
    """
    t = o3d.io.read_point_cloud(file)
    if len(t.points) > n:
        t = t.farthest_point_down_sample(n)
    pts = np.asarray(t.points)
    cen = np.mean(pts, axis=0)
    m1 = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    pts = (pts - cen)
    m = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    pts = (pts/m) * sf
    perf =abs(m-sf)*100/sf
    print(m, m1)
    return pts, cen, perf


def generate_hypotheses(args):
    seed = args.SEARCH_SEED
    min_deviation = args.SEARCH_MINDEV
    max_deviation = args.SEARCH_MAXDEV
    step = args.SEARCH_STEP_SIZE

    num_roll = np.round((max_deviation[0] - min_deviation[0]) / step[0] + 1)
    num_pitch = np.round((max_deviation[1] - min_deviation[1]) / step[1] + 1)
    num_yaw = np.round((max_deviation[2] - min_deviation[2]) / step[2] + 1)

    # Initialize the hypotheses matrix
    hypotheses = np.zeros((int(num_roll * num_pitch * num_yaw), 3))

    hyp_index = 0

    for roll in range(int(num_roll)):
        hyp_0 = seed[0] + min_deviation[0] + step[0] * roll
        for pitch in range(int(num_pitch)):
            hyp_1 = seed[1] + min_deviation[1] + step[1] * pitch
            for yaw in range(int(num_yaw)):
                hyp_2 = seed[2] + min_deviation[2] + step[2] * yaw
                # Save each hypothesis to the hypotheses list
                hypotheses[hyp_index, :] = [hyp_0, hyp_1, hyp_2]
                hyp_index += 1
    return hypotheses  # torch.tensor(hypotheses,device=device)


def fibonacci_sphere(samples=2000):
    """
    Generate points on a unit sphere using the Fibonacci lattice.

    Args:
        samples (int): Number of points to generate on the sphere.

    Returns:
        np.ndarray: Array of 3D points on the sphere.
    """
    indices = np.arange(0, samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / samples)  # Elevation angle
    theta = np.pi * (1 + 5 ** 0.5) * indices  # Golden angle

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack([x, y, z], axis=-1)


def generate_rotations_from_fibonacci(samples=1000, seed=51):
    """
    Generate unique rotations based on points from a Fibonacci sphere.

    Args:
        samples (int): Number of rotations to generate.

    Returns:
        list: A list of quaternions representing rotations.
        :param samples: no. of orientations
        :param seed: numpy random seed
    """
    points = fibonacci_sphere(samples)  # Points on the sphere
    angs = []
    np.random.seed(seed)
    for point in points:
        axis = point / np.linalg.norm(point)  # Normalize the point to get axis
        angle = np.random.uniform(-np.pi, np.pi)  # Random angle in radians
        rotation = Rotation.from_rotvec(axis * angle)  # Convert axis-angle to rotation
        angs.append(rotation.as_euler('XYZ', degrees=True))  # Save as quaternion

    return np.array(angs)


def calculate_evidence(lookup_table, hypotheses, point_cloud, max_xyz, num_xyz, points_per_meter, T):
    def process_hypothesis(i):
        # Transform the point cloud measurements to the lookup frame
        sensor_to_model = R_Intrinsic(hypotheses[i, 0], hypotheses[i, 1], hypotheses[i, 2])
        stm_inv = np.linalg.inv(sensor_to_model)
        # stm_inv = sensor_to_model
        t = np.expand_dims(np.array(T), axis=0)
        pointcloud_lookup = (stm_inv @ point_cloud.T) + t.T
        pointcloud_lookup = pointcloud_lookup.T
        # Extra Translation added to make all points x,y,z positive for the lookup index calculation
        # True translation kept as completed Ptcloud centroid

        # Initialize the evidence to zero
        evidence = 0

        # Iterate through the sensor measurements and sum the evidence
        for k in range(point_cloud.shape[0]):
            # Compute the lookup table indices
            x, y, z = pointcloud_lookup[k]

            if 0 <= x <= max_xyz[0] and 0 <= y <= max_xyz[1] and 0 <= z <= max_xyz[2]:
                x_index = round(x * points_per_meter)
                y_index = round(y * points_per_meter)
                z_index = round(z * points_per_meter)
                index = z_index + (y_index * num_xyz[2]) + (x_index * num_xyz[1] * num_xyz[2])
                evidence += lookup_table[index]
        return evidence

    # Parallel processing using joblib
    hyp_evidence = Parallel(n_jobs=-1)(delayed(process_hypothesis)(i) for i in range(hypotheses.shape[0]))

    return hyp_evidence


def readLookupTable(lookup_file):
    f = np.loadtxt(lookup_file, delimiter=',')
    # f = torch.from_numpy(np.array(f)).cuda()
    return f


mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])


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
            ort.paint_uniform_color([1, 0, 0])
            rec.paint_uniform_color([0, 0, 0])
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
    #ctr = vis.get_view_control()
    # ctr.rotate(0, -50)
    vis.add_geometry(mesh_coord_frame)
    vis.add_geometry(ort)
    vis.add_geometry(rec)
    vis.register_animation_callback(forward)
    vis.run()
    vis.destroy_window()


def compute_adds_score(pred_pcd, gt_pcd, diameter, percentage=0.05):
    kdt = KDTree(gt_pcd, metric='euclidean')
    distance, _ = kdt.query(pred_pcd, k=1)
    mean_distances = np.mean(distance)
    threshold = diameter * percentage
    score = (mean_distances < threshold).sum()
    return score


def add_gaussian_noise(pcd, mean=0.0, stddev=0.01):
    # Convert point cloud to numpy array
    points = np.asarray(pcd.points)

    # Generate Gaussian noise
    noise = np.random.normal(mean, stddev, points.shape)

    # Add the noise to the points
    noisy_points = points + noise

    # Update the point cloud with noisy points
    pcd.points = o3d.utility.Vector3dVector(noisy_points)

    return pcd


def generate_random_rotation():
    # Generate a random rotation matrix
    np.random.seed(51)
    random_rotation = o3d.geometry.get_rotation_matrix_from_xyz(
        np.random.uniform(low=-np.pi, high=np.pi, size=3)
    )
    return random_rotation


def gen_hypotheses_M():
    seed = [0, 0, 0]
    min_deviation = [-180, -90, -180]
    max_deviation = [180, 90, 180]
    step = [15, 15, 15]

    num_roll = np.round((max_deviation[0] - min_deviation[0]) / step[0] + 1)
    num_pitch = np.round((max_deviation[1] - min_deviation[1]) / step[1] + 1)
    num_yaw = np.round((max_deviation[2] - min_deviation[2]) / step[2] + 1)

    # Initialize the hypotheses matrix
    hypotheses = np.zeros((int(num_roll * num_pitch * num_yaw), 3))

    hyp_index = 0

    for roll in range(int(num_roll)):
        hyp_0 = seed[0] + min_deviation[0] + step[0] * roll
        for pitch in range(int(num_pitch)):
            hyp_1 = seed[1] + min_deviation[1] + step[1] * pitch
            for yaw in range(int(num_yaw)):
                hyp_2 = seed[2] + min_deviation[2] + step[2] * yaw
                # Save each hypothesis to the hypotheses list
                hypotheses[hyp_index, :] = [hyp_0, hyp_1, hyp_2]
                hyp_index += 1
    return hypotheses


def create_thick_vector(origin, shaft_radius=0.005, cone_radius=0.007, length=1.0):
    """
    Create a straight vector as an arrow in Open3D with adjustable thickness.

    :param origin: The start point of the vector (3D coordinates)
    :param direction: The direction of the vector (3D coordinates)
    :param shaft_radius: Radius of the shaft (controls thickness)
    :param cone_radius: Radius of the cone (controls thickness at the tip)
    :param length: Length of the arrow (vector)
    :return: Transformed arrow object representing the vector
    """
    # Create an arrow along the Z-axis, which we will rotate and translate
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=shaft_radius,
                                                   cone_radius=cone_radius,
                                                   cylinder_height=length - 0.2 * length,  # Adjust height of shaft
                                                   cone_height=0.2 * length)  # Adjust height of cone

    # Translate the arrow to start from the origin
    arrow.translate(origin)

    # Set the color of the arrow
    arrow.paint_uniform_color([0.5, 0, 1])
    return arrow


def vis_reward_map(hyp, evd):
    origin = np.array([0, 0, 0])  # Start at the origin (0, 0, 0)
    evd = evd / max(evd)
    viridis = plt.get_cmap('viridis')
    # Create the vector as an arrow with adjustable thickness
    vectors = []
    for i in range(len(hyp)):
        hp = hyp[i]
        r = R_Intrinsic(hp[0], hp[1], hp[2])
        thick_vector = create_thick_vector(origin, length=5 * evd[i])
        thick_vector.rotate(r, center=[0, 0, 0])
        cmap = viridis(evd[i])
        thick_vector.paint_uniform_color([cmap[0], cmap[1], cmap[2]])
        vectors.append(thick_vector)

    vectors.append(mesh_coord_frame)
    o3d.visualization.draw_geometries(vectors, window_name="Hypotheses Evidence Relation")


def uniform_orientations(samples=1000):
    """
    Generate uniform orientations using Fibonacci sphere points.

    Args:
        samples (int): Number of orientations to generate.
        angle (float): Fixed rotation angle in radians.

    Returns:
        np.ndarray: Array of quaternions representing uniform rotations.
    """
    points = fibonacci_sphere(samples)  # Points on the sphere
    orientations = []
    angle = np.pi
    for point in points:
        axis = point / np.linalg.norm(point)  # Normalize the axis
        rotation = Rotation.from_rotvec(axis * angle)  # Axis-angle to rotation
        orientations.append(rotation.as_euler('XYZ', degrees=True))  # Quaternion representation

    return np.array(orientations)
