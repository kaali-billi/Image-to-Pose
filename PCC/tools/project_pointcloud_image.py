import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt


def project_pointcloud_to_image(pcd, image, fx, fy, cx, cy, 
                                max_depth=None, point_size=2):
    """
    Project 3D point cloud onto 2D image using camera intrinsics
    
    Args:
        pcd: Open3D point cloud object
        image: RGB image (H, W, 3) numpy array
        fx, fy: Focal lengths in pixels
        cx, cy: Principal point coordinates
        max_depth: Optional max depth filter
        point_size: Size of projected points
    
    Returns:
        projected_image: Image with projected points overlaid
    """
    # Get points and colors from point cloud
    points_3d = np.asarray(pcd.points)  # (N, 3) - X, Y, Z
    
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)  # (N, 3) - RGB [0, 1]
        colors = (colors * 255).astype(np.uint8)
    else:
        # Default to cyan if no colors
        colors = np.full((len(points_3d), 3), [0, 255, 255], dtype=np.uint8)
    
    # Create output image
    H, W = image.shape[:2]
    projected_image = image.copy()
    
    # Filter by depth if specified
    if max_depth is not None:
        valid_depth = points_3d[:, 2] < max_depth
        points_3d = points_3d[valid_depth]
        colors = colors[valid_depth]
    
    # Project 3D points to 2D using pinhole camera model
    # u = fx * X/Z + cx
    # v = fy * Y/Z + cy
    
    Z = points_3d[:, 2]
    valid_z = Z < 0  # Only project points in front of camera
    
    u = (fx * points_3d[:, 0] / Z + cx).astype(int)
    v = (fy * points_3d[:, 1] / Z + cy).astype(int)
    
    # Filter points within image bounds
    valid_u = (u >= 0) & (u < W)
    valid_v = (v >= 0) & (v < H)
    valid = valid_z & valid_u & valid_v
    
    u = u[valid]
    v = v[valid]
    colors = colors[valid]
    depths = Z[valid]
    
    # Sort by depth (far to near) for proper occlusion
    depth_order = np.argsort(-depths)  # Descending order
    u = u[depth_order]
    v = v[depth_order]
    colors = colors[depth_order]
    
    # Draw points on image
    for i in range(len(u)):
        cv2.circle(projected_image, (u[i], v[i]), point_size, 
                  colors[i].tolist(), -1)
    
    print(f"Projected {len(u)} / {len(points_3d)} points")
    
    return projected_image


def project_with_depth_colormap(pcd, image, fx, fy, cx, cy, 
                                max_depth=50, point_size=2):
    """
    Project point cloud with depth-based coloring
    
    Args:
        pcd: Open3D point cloud
        image: RGB image
        fx, fy, cx, cy: Camera intrinsics
        max_depth: Maximum depth for color mapping
        point_size: Point size
    
    Returns:
        projected_image: Image with depth-colored points
    """
    points_3d = np.asarray(pcd.points)
    H, W = image.shape[:2]
    projected_image = image.copy()
    
    # Project to 2D
    Z = points_3d[:, 2]
    valid_z = Z > 0
    
    u = (fx * points_3d[:, 0] / Z + cx).astype(int)
    v = (fy * points_3d[:, 1] / Z + cy).astype(int)
    
    valid_u = (u >= 0) & (u < W)
    valid_v = (v >= 0) & (v < H)
    valid = valid_z & valid_u & valid_v
    
    u = u[valid]
    v = v[valid]
    depths = Z[valid]
    
    # Normalize depths for colormap
    depths_norm = np.clip(depths / max_depth, 0, 1)
    
    # Apply colormap (plasma, jet, turbo, etc.)
    colors = plt.cm.plasma(depths_norm)[:, :3]  # RGB
    colors = (colors * 255).astype(np.uint8)
    
    # Sort by depth
    depth_order = np.argsort(-depths)
    u = u[depth_order]
    v = v[depth_order]
    colors = colors[depth_order]
    
    # Draw points
    for i in range(len(u)):
        cv2.circle(projected_image, (u[i], v[i]), point_size,
                  colors[i].tolist(), -1)
    
    return projected_image


def create_depth_image_from_pointcloud(pcd, image_size, fx, fy, cx, cy):
    """
    Create a depth image from point cloud projection
    
    Args:
        pcd: Open3D point cloud
        image_size: (width, height) tuple
        fx, fy, cx, cy: Camera intrinsics
    
    Returns:
        depth_image: (H, W) depth map
    """
    points_3d = np.asarray(pcd.points)
    W, H = image_size
    
    # Initialize depth image
    depth_image = np.zeros((H, W), dtype=np.float32)
    
    # Project points
    Z = points_3d[:, 2]
    valid_z = Z > 0
    
    u = (fx * points_3d[:, 0] / Z + cx).astype(int)
    v = (fy * points_3d[:, 1] / Z + cy).astype(int)
    
    valid_u = (u >= 0) & (u < W)
    valid_v = (v >= 0) & (v < H)
    valid = valid_z & valid_u & valid_v
    
    u = u[valid]
    v = v[valid]
    depths = Z[valid]
    
    # Fill depth image (keep closest depth per pixel)
    for i in range(len(u)):
        current_depth = depth_image[v[i], u[i]]
        if current_depth == 0 or depths[i] < current_depth:
            depth_image[v[i], u[i]] = depths[i]
    
    return depth_image


# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    
    # Example 1: Load point cloud and project onto image
    # --------------------------------------------------
    
    # Load point cloud
    pcd = o3d.io.read_point_cloud("pointcloud.pcd")
    
    # Load RGB image
    image = cv2.imread("image.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Camera intrinsics for 1000x1000 image
    fx = 500.0
    fy = 500.0
    cx = 499.5
    cy = 499.5
    
    # Project point cloud onto image
    projected = project_pointcloud_to_image(
        pcd, image, fx, fy, cx, cy,
        max_depth=50,  # Filter far points
        point_size=2
    )
    
    # Display
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(projected)
    plt.title("Projected Point Cloud")
    plt.axis('off')
    
    # Show with depth coloring
    projected_depth = project_with_depth_colormap(
        pcd, image, fx, fy, cx, cy, max_depth=50
    )
    
    plt.subplot(133)
    plt.imshow(projected_depth)
    plt.title("Depth-Colored Projection")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("projection_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    
    # Example 2: Create point cloud from depth and project back
    # ----------------------------------------------------------
    
    def depth_to_pointcloud(depth_map, rgb_image, fx, fy, cx, cy):
        """Convert depth map to point cloud"""
        H, W = depth_map.shape
        
        u = np.arange(W)
        v = np.arange(H)
        u, v = np.meshgrid(u, v)
        
        Z = depth_map
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        
        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        colors = rgb_image.reshape(-1, 3) / 255.0
        
        # Filter valid points
        valid = Z.flatten() > 0
        points = points[valid]
        colors = colors[valid]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    # Load depth map
    depth = cv2.imread("depth.png", cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.0  # Convert to meters
    
    # Create point cloud from depth
    pcd_from_depth = depth_to_pointcloud(depth, image, fx, fy, cx, cy)
    
    # Project back onto image
    reprojected = project_pointcloud_to_image(
        pcd_from_depth, image, fx, fy, cx, cy
    )
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(image)
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(reprojected)
    plt.title("Reprojected")
    plt.axis('off')
    
    plt.show()
