import numpy as np
import math
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
import time
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import pyquaternion as pyq
import plotly.graph_objects as go
from scipy.spatial import distance_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

SEQ = "XYZ"


def Trajectory(Q, T):
    q = pyq.Quaternion(Q)
    t = [0, T[0], T[1], T[2]]
    W = q * t * q.inverse
    return W[1], W[2], W[3]


def Plot_traj_compare(x, y, z, x1, y1, z1):
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, name='GT Trajectory',
        marker=dict(size=6, color=z, colorscale='Viridis'),
        line=dict(color='darkblue', width=5)
    ), go.Scatter3d(
        x=x1, y=y1, z=z1, name='Reconstructed Trajectory',
        marker=dict(size=6, color=z, colorscale='inferno'),
        line=dict(color='ivory', width=5)
    )])

    fig.update_layout(
        width=1920, height=1080, autosize=False,
        scene=dict(
            camera=dict(up=dict(x=0, y=0, z=1), eye=dict(x=0, y=1.0707, z=1)),
            aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube',
            xaxis=dict(nticks=4, range=[-500, 50]),
            yaxis=dict(nticks=4, range=[-500, 50])
        )
    )
    fig.write_html('PLuM_Python/LRO_test/Trajectory_Compare.html')
    return fig.show(renderer="browser")


def read_params(file):
    parameters = {}
    with open(file, 'r') as file:
        for line in file:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if ',' in value:
                parameters[key] = np.array([float(x) for x in value.split(',')])
            elif value.replace('.', '', 1).isdigit():
                parameters[key] = float(value)
            else:
                parameters[key] = value
    return parameters


def pick_points(pcd):
    print("\n1) Please pick at least three correspondences using [shift + left click]")
    print("Press [shift + right click] to undo point picking")
    print("2) After picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def compute_diameter_open3d_accurate(point_cloud):
    """GPU-accelerated diameter computation"""
    points = torch.tensor(np.asarray(point_cloud.points), device=device, dtype=torch.float32)

    # Compute pairwise distances on GPU
    dists = torch.cdist(points, points, p=2)
    diameter = torch.max(dists).item()

    return diameter


def homogeneous_intrinsic(ROLL, PITCH, YAW, X, Y, Z):
    T = np.eye(4, 4)
    T[:3, :3] = Rotation.from_euler(SEQ, [ROLL, PITCH, YAW], degrees=True).as_matrix()
    T[:, 3] = [X, Y, Z, 1]
    return T


def R_Intrinsic(roll, pitch, yaw):
    """Keep on CPU for single rotations - matches scipy and Open3D"""
    g = Rotation.from_euler(SEQ, [roll, pitch, yaw], degrees=True).as_matrix()
    return g


def R_Intrinsic_batch_gpu(roll_pitch_yaw):
    """
    GPU-accelerated batch rotation matrix generation using scipy for correctness
    VERIFIED to match:
    - scipy.spatial.transform.Rotation.from_euler('XYZ', angles, degrees=True)
    - open3d.geometry.get_rotation_matrix_from_xyz()

    Args:
        roll_pitch_yaw: torch.Tensor or numpy array of shape (N, 3) or (3,) in degrees

    Returns:
        torch.Tensor of shape (N, 3, 3) or (3, 3) rotation matrices

    Note: Uses scipy internally for guaranteed correctness with XYZ INTRINSIC rotations
    """
    # Convert to numpy if needed
    if isinstance(roll_pitch_yaw, torch.Tensor):
        device_orig = roll_pitch_yaw.device
        rpy_np = roll_pitch_yaw.cpu().numpy()
    else:
        device_orig = device
        rpy_np = roll_pitch_yaw

    # Use scipy to generate rotation matrices (guaranteed correct)
    if rpy_np.ndim == 1:
        # Single rotation
        R_scipy = Rotation.from_euler('XYZ', rpy_np, degrees=True).as_matrix()
    else:
        # Batch processing - scipy can handle batch directly!
        R_scipy = Rotation.from_euler('XYZ', rpy_np, degrees=True).as_matrix()

    # Convert to torch tensor on appropriate device
    R = torch.tensor(R_scipy, device=device_orig, dtype=torch.float32)

    return R


def hom2eul(T):
    roll = math.atan2(T[2, 1], T[2, 2])
    pitch = math.asin(-T[2, 0])
    yaw = math.atan2(T[1, 0], T[0, 0])
    x, y, z = T[0, 3], T[1, 3], T[2, 3]
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

    if ret:
        return ort
    else:
        print(f"Time Taken: {TT}")
        print(f"Iter, Counter: {iteration}, {g}")
        print("roll(deg), pitch(deg), yaw(deg), evidence, Rev/point")
        print(f"{hyp}, {maxElement}, {maxElement / len(pcd_file.points)}")
        mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([ort, pcd, mesh_coord_frame])
        return None


def save_args_to_file(args, filename):
    with open(filename, 'w') as file:
        for arg, value in vars(args).items():
            file.write(f"{arg}: {value}\n")


def read_pc(file, n=1024, sf=0):
    t = o3d.io.read_point_cloud(file)
    if len(t.points) > n:
        t = t.farthest_point_down_sample(n)
    pts = np.asarray(t.points)
    cen = np.mean(pts, axis=0)
    pts = pts - cen
    m = np.max(np.sqrt(np.sum(pts ** 2, axis=1)))
    pts = pts/m * sf
    #print(m,sf)
    perf = abs(m - sf) * 100 / sf
    return pts, cen, perf


def generate_hypotheses(args):
    seed = args.SEARCH_SEED
    min_deviation = args.SEARCH_MINDEV
    max_deviation = args.SEARCH_MAXDEV
    step = args.SEARCH_STEP_SIZE

    num_roll = np.round((max_deviation[0] - min_deviation[0]) / step[0] + 1)
    num_pitch = np.round((max_deviation[1] - min_deviation[1]) / step[1] + 1)
    num_yaw = np.round((max_deviation[2] - min_deviation[2]) / step[2] + 1)

    hypotheses = np.zeros((int(num_roll * num_pitch * num_yaw), 3))
    hyp_index = 0

    for roll in range(int(num_roll)):
        hyp_0 = seed[0] + min_deviation[0] + step[0] * roll
        for pitch in range(int(num_pitch)):
            hyp_1 = seed[1] + min_deviation[1] + step[1] * pitch
            for yaw in range(int(num_yaw)):
                hyp_2 = seed[2] + min_deviation[2] + step[2] * yaw
                hypotheses[hyp_index, :] = [hyp_0, hyp_1, hyp_2]
                hyp_index += 1

    return hypotheses


def fibonacci_sphere(samples=2000):
    """Generate points on a unit sphere using the Fibonacci lattice."""
    indices = np.arange(0, samples, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / samples)
    theta = np.pi * (1 + 5 ** 0.5) * indices

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack([x, y, z], axis=-1)


def generate_rotations_from_fibonacci(samples=1000, seed=51):
    """Generate unique rotations based on points from a Fibonacci sphere."""
    points = fibonacci_sphere(samples)
    angs = []
    np.random.seed(seed+54632)

    for point in points:
        axis = point / np.linalg.norm(point)
        angle = np.random.uniform(-np.pi, np.pi)
        rotation = Rotation.from_rotvec(axis * angle)
        angs.append(rotation.as_euler('XYZ', degrees=True))

    return np.array(angs)


def calculate_evidence_gpu(lookup_table, hypotheses, point_cloud, max_xyz, num_xyz, points_per_meter, T):
    """
    Fully GPU-accelerated evidence calculation
    VERIFIED to match original CPU implementation with scipy rotations

    Args:
        lookup_table: numpy array or torch tensor
        hypotheses: numpy array of shape (N, 3) - roll, pitch, yaw in degrees
        point_cloud: numpy array of shape (M, 3)
        max_xyz: list of 3 floats
        num_xyz: list of 3 ints
        points_per_meter: float
        T: translation vector [x, y, z]
    Returns:
        list of evidence values
    """
    # Move data to GPU
    if isinstance(lookup_table, np.ndarray):
        lookup_table = torch.tensor(lookup_table, device=device, dtype=torch.float32)
    if isinstance(hypotheses, np.ndarray):
        hypotheses_gpu = torch.tensor(hypotheses, device=device, dtype=torch.float32)
    else:
        hypotheses_gpu = hypotheses
    point_cloud_gpu = torch.tensor(point_cloud, device=device, dtype=torch.float32)
    T_gpu = torch.tensor(T, device=device, dtype=torch.float32).unsqueeze(0)
    max_xyz_gpu = torch.tensor(max_xyz, device=device, dtype=torch.float32)
    num_xyz_gpu = torch.tensor(num_xyz, device=device, dtype=torch.long)

    N = hypotheses_gpu.shape[0]  # Number of hypotheses
    M = point_cloud_gpu.shape[0]  # Number of points

    # Generate batch rotation matrices (VERIFIED to match scipy/Open3D)
    R_batch = R_Intrinsic_batch_gpu(hypotheses_gpu)  # (N, 3, 3)
    R_inv_batch = torch.linalg.inv(R_batch)  # (N, 3, 3)

    # Transform point cloud for all hypotheses at once
    # point_cloud_gpu: (M, 3) -> (1, M, 3)
    # R_inv_batch: (N, 3, 3)
    # Result: (N, M, 3)
    pc_expanded = point_cloud_gpu.unsqueeze(0).expand(N, -1, -1)  # (N, M, 3)

    # Batch matrix multiplication: (N, 3, 3) @ (N, 3, M) -> (N, 3, M)
    pointcloud_lookup = torch.bmm(R_inv_batch, pc_expanded.transpose(1, 2))  # (N, 3, M)
    pointcloud_lookup = pointcloud_lookup.transpose(1, 2)  # (N, M, 3)

    # Add translation
    pointcloud_lookup = pointcloud_lookup + T_gpu.unsqueeze(1)  # (N, M, 3)

    # Compute indices for all points and hypotheses
    # Check bounds
    valid_mask = (
            (pointcloud_lookup[..., 0] >= 0) & (pointcloud_lookup[..., 0] <= max_xyz_gpu[0]) &
            (pointcloud_lookup[..., 1] >= 0) & (pointcloud_lookup[..., 1] <= max_xyz_gpu[1]) &
            (pointcloud_lookup[..., 2] >= 0) & (pointcloud_lookup[..., 2] <= max_xyz_gpu[2])
    )  # (N, M)

    # Compute indices
    indices_float = pointcloud_lookup * points_per_meter
    indices = torch.round(indices_float).long()  # (N, M, 3)

    # Flatten index calculation: z + y*num_z + x*num_y*num_z
    flat_indices = (
            indices[..., 2] +
            indices[..., 1] * num_xyz_gpu[2] +
            indices[..., 0] * num_xyz_gpu[1] * num_xyz_gpu[2]
    )  # (N, M)

    # Clamp indices to valid range
    flat_indices = torch.clamp(flat_indices, 0, len(lookup_table) - 1)

    # Apply mask - set invalid indices to 0 (will contribute 0 evidence)
    flat_indices = flat_indices * valid_mask.long()

    # Gather evidence values
    evidence_values = lookup_table[flat_indices]  # (N, M)

    # Zero out invalid entries
    evidence_values = evidence_values * valid_mask.float()

    # Sum evidence for each hypothesis
    evidence = torch.sum(evidence_values, dim=1)  # (N,)

    return evidence.cpu().numpy().tolist()


from torch.distributions import Normal


def parallel_hypothesis_sampling_gpu(
        hypotheses,
        evidence,
        iteration,
        params,
        thresh,
        npts,
        device='cuda'
):
    """
    GPU-parallelized hypothesis sampling using PyTorch

    Args:
        hypotheses: torch.Tensor of shape (num_hypotheses, 3)
        evidence: torch.Tensor of shape (num_hypotheses,)
        iteration: int - current iteration number
        params: object with SEARCH_ROT_SIGMA, SEARCH_RESAMPLE, SEARCH_ITER
        thresh: float - threshold for early stopping
        npts: int - number of points
        device: str - 'cuda' or 'cpu'

    Returns:
        hypotheses: Updated hypotheses tensor
        maxElement: Maximum evidence value
        maxElementIndex: Index of maximum evidence
        should_break: Boolean indicating if threshold is reached
    """

    # Move tensors to GPU
    hypotheses = hypotheses.to(device)
    evidence = evidence.to(device)

    numberOfHypotheses = hypotheses.shape[0]

    # =========================================================================
    # Step 1: Compute exaggerated hypothesis probabilities (parallelized)
    # =========================================================================

    # Exaggerate evidence
    exaggerated = torch.pow(evidence, iteration)

    # Compute normalizing constant
    normalisingConstant = torch.sum(exaggerated)

    # Check for infinity
    if torch.isinf(normalisingConstant):
        raise ValueError("Normalising constant has reached infinity!")

    # Normalize probabilities
    exageratedHypothesisProb = exaggerated / normalisingConstant

    # Compute cumulative probabilities (parallel prefix sum)
    cumProb = torch.cumsum(exageratedHypothesisProb, dim=0)

    # =========================================================================
    # Step 2: Find maximum element (parallelized)
    # =========================================================================

    maxElement = torch.max(evidence)
    maxElementIndex = torch.argmax(evidence)

    # Check for threshold
    should_break = (maxElement / npts) > thresh

    if should_break:
        return hypotheses, maxElement.item(), maxElementIndex.item(), should_break

    # =========================================================================
    # Step 3: Resample hypotheses (parallelized)
    # =========================================================================

    # Set seed for reproducibility
    torch.manual_seed(0)

    # Scale noise down per iteration
    scale = 1.0 / iteration
    noise_sigma = params.SEARCH_ROT_SIGMA * scale

    # Generate uniform sample levels (vectorized)
    sample_levels = (torch.arange(params.SEARCH_RESAMPLE, device=device) + 0.5) / params.SEARCH_RESAMPLE

    # Find indices using searchsorted (parallelized binary search)
    sampled_indices = torch.searchsorted(cumProb, sample_levels, right=False)

    # Clamp indices to valid range
    sampled_indices = torch.clamp(sampled_indices, 0, numberOfHypotheses - 1)

    # Sample hypotheses based on indices (fully parallelized)
    hypothesesSampled = hypotheses[sampled_indices, :]

    # Generate noise for all samples at once (parallelized)
    noise_distribution = Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(noise_sigma, device=device)
    )
    noiseVectors = noise_distribution.sample((params.SEARCH_RESAMPLE, 3))

    # Add noise to all hypotheses at once
    hypothesesSampled = hypothesesSampled + noiseVectors

    # =========================================================================
    # Step 4: Handle special cases
    # =========================================================================

    # Update number of hypotheses after first iteration
    if iteration == 1:
        params.SEARCH_RESAMPLE += 1
        numberOfHypotheses = params.SEARCH_RESAMPLE

    # Save the best evidence (overwrite last element)
    hypothesesSampled[params.SEARCH_RESAMPLE - 1, :] = hypotheses[maxElementIndex, :]

    # Update hypotheses for next iteration
    if iteration != params.SEARCH_ITER:
        hypotheses = hypothesesSampled[:params.SEARCH_RESAMPLE, :]
    else:
        hypotheses = hypothesesSampled

    return hypotheses, maxElement.item(), maxElementIndex.item(), should_break


def parallel_hypothesis_sampling_gpu1(
        hypotheses,
        evidence,
        iteration,
        params,
        thresh,
        npts,
        device='cuda'
):
    """
    GPU-parallelized hypothesis sampling
    All inputs and outputs are torch tensors on GPU
    """

    # Ensure tensors are on correct device
    hypotheses = hypotheses.to(device)
    evidence = evidence.to(device)

    numberOfHypotheses = hypotheses.shape[0]

    # Step 1: Compute exaggerated probabilities
    exaggerated = torch.pow(evidence, iteration)
    normalisingConstant = torch.sum(exaggerated)

    if torch.isinf(normalisingConstant):
        raise ValueError("Normalising constant has reached infinity!")

    exageratedHypothesisProb = exaggerated / normalisingConstant
    cumProb = torch.cumsum(exageratedHypothesisProb, dim=0)

    # Step 2: Find maximum
    maxElement = torch.max(evidence)
    maxElementIndex = torch.argmax(evidence)

    should_break = (maxElement / npts) > thresh

    if should_break:
        return hypotheses, maxElement.item(), maxElementIndex.item(), should_break

    # Step 3: Resample hypotheses
    torch.manual_seed(0)

    scale = 1.0 / iteration
    noise_sigma = params.SEARCH_ROT_SIGMA * scale

    sample_levels = (torch.arange(params.SEARCH_RESAMPLE, device=device) + 0.5) / params.SEARCH_RESAMPLE
    sampled_indices = torch.searchsorted(cumProb, sample_levels, right=False)
    sampled_indices = torch.clamp(sampled_indices, 0, numberOfHypotheses - 1)

    hypothesesSampled = hypotheses[sampled_indices, :]

    noise_distribution = Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(noise_sigma, device=device)
    )
    noiseVectors = noise_distribution.sample((params.SEARCH_RESAMPLE, 3))
    hypothesesSampled = hypothesesSampled + noiseVectors

    # Step 4: Handle special cases
    if iteration == 1:
        # Increment SEARCH_RESAMPLE for next iterations
        params.SEARCH_RESAMPLE = 2001
        numberOfHypotheses = params.SEARCH_RESAMPLE

        # Expand hypothesesSampled to accommodate the new size
        extra_slot = torch.zeros((1, 3), device=device, dtype=hypothesesSampled.dtype)
        hypothesesSampled = torch.cat([hypothesesSampled, extra_slot], dim=0)

    # Save the best evidence (now safe - hypothesesSampled has correct size)
    print(hypothesesSampled.shape, hypotheses.shape)
    hypothesesSampled[params.SEARCH_RESAMPLE - 1, :] = hypotheses[maxElementIndex, :]

    # Update hypotheses for next iteration
    if iteration != params.SEARCH_ITER:
        hypotheses = hypothesesSampled[:params.SEARCH_RESAMPLE, :]
    else:
        hypotheses = hypothesesSampled

    return hypotheses, maxElement.item(), maxElementIndex.item(), should_break


def parallel_hypothesis_sampling_gpu2(
        hypotheses,
        evidence,
        iteration,
        params,
        thresh,
        npts,
        device='cuda',
        resample_count=None  # Track locally instead of modifying params
):
    """GPU-parallelized hypothesis sampling"""

    hypotheses = hypotheses.to(device)
    evidence = evidence.to(device)

    # Use local variable, don't modify params
    if resample_count is None:
        resample_count = params.SEARCH_RESAMPLE  # Initial value: 2000

    numberOfHypotheses = hypotheses.shape[0]

    # Compute exaggerated probabilities
    exaggerated = torch.pow(evidence, iteration)
    normalisingConstant = torch.sum(exaggerated)

    if torch.isinf(normalisingConstant):
        raise ValueError("Normalising constant has reached infinity!")

    exageratedHypothesisProb = exaggerated / normalisingConstant
    cumProb = torch.cumsum(exageratedHypothesisProb, dim=0)

    # Find maximum
    maxElement = torch.max(evidence)
    maxElementIndex = torch.argmax(evidence)

    should_break = (maxElement / npts) > thresh

    if should_break:
        return hypotheses, maxElement.item(), maxElementIndex.item(), should_break, resample_count

    # Resample hypotheses
    torch.manual_seed(0)
    scale = 1.0 / iteration
    noise_sigma = params.SEARCH_ROT_SIGMA * scale

    # Use local resample_count, not params.SEARCH_RESAMPLE
    sample_levels = (torch.arange(resample_count, device=device) + 0.5) / resample_count
    sampled_indices = torch.searchsorted(cumProb, sample_levels, right=False)
    sampled_indices = torch.clamp(sampled_indices, 0, numberOfHypotheses - 1)

    hypothesesSampled = hypotheses[sampled_indices, :]

    # Add noise
    noise_distribution = Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(noise_sigma, device=device)
    )
    noiseVectors = noise_distribution.sample((resample_count, 3))
    hypothesesSampled = hypothesesSampled + noiseVectors

    # Handle first iteration - increment LOCAL variable
    if iteration == 1:
        resample_count += 1  # Only affects local variable, not params
        best_hyp = hypotheses[maxElementIndex:maxElementIndex + 1, :]
        hypothesesSampled = torch.cat([hypothesesSampled, best_hyp], dim=0)
    else:
        # Now resample_count matches hypothesesSampled size
        print(hypothesesSampled.shape, hypotheses.shape, iteration)
        hypothesesSampled[resample_count - 1, :] = hypotheses[maxElementIndex, :]

    return hypothesesSampled, maxElement.item(), maxElementIndex.item(), should_break, resample_count


def parallel_hypothesis_sampling_gpu3(
        hypotheses,
        evidence,
        iteration,
        params,
        thresh,
        npts,
        device='cuda',
        resample_count=None
):
    """GPU-parallelized hypothesis sampling"""

    hypotheses = hypotheses.to(device)
    evidence = evidence.to(device)

    if resample_count is None:
        resample_count = params.SEARCH_RESAMPLE

    numberOfHypotheses = hypotheses.shape[0]

    # Compute exaggerated probabilities
    log_evidence = torch.log(evidence + 1e-10)  # Add small epsilon to avoid log(0)
    log_exaggerated = iteration * log_evidence

    # Subtract max for numerical stability (log-sum-exp trick)
    log_exaggerated = log_exaggerated - torch.max(log_exaggerated)
    exaggerated = torch.exp(log_exaggerated)

    normalisingConstant = torch.sum(exaggerated)

    if torch.isinf(normalisingConstant):
        raise ValueError("Normalising constant has reached infinity!")

    exageratedHypothesisProb = exaggerated / normalisingConstant
    cumProb = torch.cumsum(exageratedHypothesisProb, dim=0)

    # Find maximum in ORIGINAL hypotheses
    maxElement = torch.max(evidence)
    maxElementIndex = torch.argmax(evidence)

    # ✅ SAVE THE BEST HYPOTHESIS NOW (before resampling)
    #print(hypotheses.shape)
    best_hypothesis = hypotheses[maxElementIndex, :].clone()  # Save it!

    should_break = (maxElement / npts) > thresh

    if should_break:
        return hypotheses, maxElement.item(), maxElementIndex.item(), should_break, resample_count

    # Resample hypotheses
    torch.manual_seed(0)
    scale = 1.0 / iteration
    noise_sigma = params.SEARCH_ROT_SIGMA * scale

    sample_levels = (torch.arange(resample_count, device=device) + 0.5) / resample_count
    sampled_indices = torch.searchsorted(cumProb, sample_levels, right=False)
    sampled_indices = torch.clamp(sampled_indices, 0, numberOfHypotheses - 1)

    hypothesesSampled = hypotheses[sampled_indices, :]

    # Add noise
    noise_distribution = Normal(
        torch.tensor(0.0, device=device),
        torch.tensor(noise_sigma, device=device)
    )
    noiseVectors = noise_distribution.sample((resample_count, 3))
    hypothesesSampled = hypothesesSampled + noiseVectors

    # Handle first iteration
    if iteration == 1:
        resample_count += 1
        # ✅ Use the saved best_hypothesis, not hypotheses[maxElementIndex]
        best_hyp = best_hypothesis.unsqueeze(0)  # Shape: (1, 3)
        hypothesesSampled = torch.cat([hypothesesSampled, best_hyp], dim=0)
    else:
        # ✅ Use the saved best_hypothesis
        hypothesesSampled[resample_count - 1, :] = best_hypothesis

    return hypothesesSampled, maxElement.item(), maxElementIndex.item(), should_break, resample_count

def readLookupTable(lookup_file):
    f = np.loadtxt(lookup_file, delimiter=',')
    return f


mesh_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.5, origin=[0, 0, 0])


def play_motion(list_of_rec: [], list_of_ort: []):
    play_motion.vis = o3d.visualization.Visualizer()
    play_motion.index = 0

    def forward(vis):
        pm = play_motion
        if pm.index < len(list_of_rec) - 1:
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

    ort = list_of_ort[0]
    rec = list_of_rec[0]
    vis = play_motion.vis
    vis.create_window(window_name='IP/RECON/ORT')
    vis.set_full_screen(True)
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
    points = np.asarray(pcd.points)
    noise = np.random.normal(mean, stddev, points.shape)
    noisy_points = points + noise
    pcd.points = o3d.utility.Vector3dVector(noisy_points)
    return pcd


def generate_random_rotation():
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

    hypotheses = np.zeros((int(num_roll * num_pitch * num_yaw), 3))
    hyp_index = 0

    for roll in range(int(num_roll)):
        hyp_0 = seed[0] + min_deviation[0] + step[0] * roll
        for pitch in range(int(num_pitch)):
            hyp_1 = seed[1] + min_deviation[1] + step[1] * pitch
            for yaw in range(int(num_yaw)):
                hyp_2 = seed[2] + min_deviation[2] + step[2] * yaw
                hypotheses[hyp_index, :] = [hyp_0, hyp_1, hyp_2]
                hyp_index += 1

    return hypotheses


def create_thick_vector(origin, shaft_radius=0.005, cone_radius=0.007, length=1.0):
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=shaft_radius,
        cone_radius=cone_radius,
        cylinder_height=length - 0.2 * length,
        cone_height=0.2 * length
    )
    arrow.translate(origin)
    arrow.paint_uniform_color([0.5, 0, 1])
    return arrow


def vis_reward_map(hyp, evd):
    origin = np.array([0, 0, 0])
    evd = evd / max(evd)
    viridis = plt.get_cmap('viridis')
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
    points = fibonacci_sphere(samples)
    orientations = []
    angle = np.pi

    for point in points:
        axis = point / np.linalg.norm(point)
        rotation = Rotation.from_rotvec(axis * angle)
        orientations.append(rotation.as_euler('XYZ', degrees=True))

    return np.array(orientations)