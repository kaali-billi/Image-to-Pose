import math
import os.path
from typing import Dict
import torchvision.transforms.functional as TF

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


def restore_crops(padded_imgs, crop_boxes, paddings, final_size=(1000, 1000), invalid_val=-1):
    """
    Restores each padded and resized crop back to the original image size.

    Args:
        padded_imgs: (B, C, H_resized, W_resized) tensor
        crop_boxes: list of [ymin, xmin, ymax, xmax] for each image
        paddings: list of [left, right, top, bottom] for each image from resize_with_aspect_ratio_and_pad
        final_size: (H_final, W_final)
        invalid_val: optional fill value

    Returns:
        Restored images: (B, C, H_final, W_final)
    """
    B, C, H_resized, W_resized = padded_imgs.shape
    H_final, W_final = final_size
    output = []

    for b in range(B):
        ymin, xmin, ymax, xmax = crop_boxes[b]
        left, right, top, bottom = paddings[b]

        img = padded_imgs[b]  # (C, H_resized, W_resized)

        # Crop out the padding to get the true resized crop
        cropped_img = img[:, top:H_resized - bottom, left:W_resized - right]

        # Resize back to original crop size
        crop_h = ymax - ymin
        crop_w = xmax - xmin
        restored_crop = F.interpolate(
            cropped_img.unsqueeze(0),  # [1, C, H_crop_resized, W_crop_resized]
            size=(crop_h, crop_w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # [C, crop_h, crop_w]

        # Create empty canvas
        if invalid_val is not None:
            restored_img = torch.full((C, H_final, W_final), invalid_val, dtype=img.dtype, device=img.device)
        else:
            restored_img = torch.zeros((C, H_final, W_final), dtype=img.dtype, device=img.device)

        # Place restored crop back to its original ROI
        restored_img[:, ymin:ymax, xmin:xmax] = restored_crop
        output.append(restored_img)

    return torch.stack(output)


def restore_val(padded_imgs, crop_boxes, final_size=(1000, 1000), invalid_val=-1):
    """
    Restores a batch of cropped grayscale images (B, 1, H, W) into a canvas of final_size.

    Args:
        padded_imgs (Tensor): shape (B, 1, H_resized, W_resized)
        crop_boxes: list of tuples: [(ymin, xmin, ymax, xmax), ...] of length B
        final_size: (H_final, W_final) of restored canvas
        invalid_val: value to fill outside the crop

    Returns:
        Tensor: (B, 1, H_final, W_final)
    """
    B, C, H_resized, W_resized = padded_imgs.shape
    H_final, W_final = final_size
    output = []

    for b in range(B):
        ymin, xmin, ymax, xmax = crop_boxes[b]
        crop_h = ymax - ymin
        crop_w = xmax - xmin
        crop_aspect = crop_w / crop_h
        resize_aspect = W_resized / H_resized

        img = padded_imgs[b]  # (1, H_resized, W_resized)

        if crop_aspect >= resize_aspect:
            scale = W_resized / crop_w
            new_h = int(crop_h * scale)
            pad_y = (H_resized - new_h) // 2
            crop_resized = img[:, pad_y:pad_y + new_h, :]
            restored_crop = F.interpolate(
                crop_resized.unsqueeze(0),
                size=(crop_h, crop_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        else:
            scale = H_resized / crop_h
            new_w = int(crop_w * scale)
            pad_x = (W_resized - new_w) // 2
            crop_resized = img[:, :, pad_x:pad_x + new_w]
            restored_crop = F.interpolate(
                crop_resized.unsqueeze(0),
                size=(crop_h, crop_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        if invalid_val is not None:
            restored_img = torch.full((1, H_final, W_final), invalid_val, dtype=img.dtype, device=img.device)
        else:
            restored_img = torch.zeros((1, H_final, W_final), dtype=img.dtype, device=img.device)

        restored_img[:, ymin:ymax, xmin:xmax] = restored_crop
        output.append(restored_img)

    return torch.stack(output)


def smooth_normalized_depth(depth_norm, median_kernel=3, bilateral_d=5,
                            bilateral_sigma_color=0.1, bilateral_sigma_space=0.5):
    """
    Apply median and bilateral filtering to a normalized depth map in [0, 1].

    Args:
        depth_norm: (B, 1, H, W) torch.Tensor, normalized depth in [0,1]
        median_kernel: int, kernel size for median filter
        bilateral_d: int, diameter for bilateral filter (pixel neighborhood)
        bilateral_sigma_color: float, sigma for color space (should be ~0.05–0.1 for normalized)
        bilateral_sigma_space: float, sigma for spatial domain

    Returns:
        Smoothed normalized depth (B, 1, H, W) tensor
    """
    B, _, H, W = depth_norm.shape
    smoothed = []

    for i in range(B):
        # 1. Median filtering in torch
        d = depth_norm[i:i + 1]  # shape: (1, 1, H, W)
        pad = median_kernel // 2
        unfolded = F.unfold(d, kernel_size=median_kernel, padding=pad)  # (1, k*k, H*W)
        unfolded = unfolded.transpose(1, 2)  # (1, H*W, k*k)
        med = unfolded.median(dim=-1).values.view(1, 1, H, W)

        # 2. Bilateral filtering in OpenCV
        med_np = med.squeeze().cpu().numpy().astype(np.float32)  # shape: (H, W)
        bilateral = cv2.bilateralFilter(
            med_np,
            d=bilateral_d,
            sigmaColor=bilateral_sigma_color,
            sigmaSpace=bilateral_sigma_space
        )

        # Convert back to torch tensor
        bilateral_tensor = torch.from_numpy(bilateral).to(depth_norm.device).clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)
        smoothed.append(bilateral_tensor)

    return torch.cat(smoothed, dim=0)


def restore_crop_to_original(padded_imgs, crop_boxes, final_size=(1000, 1000), invalid_val=-1):
    """
    Restores a batch of N cropped grayscale images per sample into 1000x1000 canvas.

    Args:
        padded_imgs (Tensor): shape (B, N, 1, H, W)
        crop_boxes_batch: list of list of tuples: [[(ymin, xmin, ymax, xmax), ...], ...] of shape (B, N)
        resize_size: (H, W) of padded crop (default: 224x224)
        final_size: (H_final, W_final) of restored canvas (default: 1000x1000)
        invalid_val: if not None, fills outside with this value (e.g., -1 for depth maps)

    Returns:
        Tensor: (B, N, 1, 1000, 1000)
    """
    B, N, C, H_resized, W_resized = padded_imgs.shape
    H_final, W_final = final_size
    output = []
    for idx, (crop_box) in enumerate(crop_boxes):
        # for b in range(B):
        restored_samples = []
        ymin, xmin, ymax, xmax = crop_box
        crop_h = ymax - ymin
        crop_w = xmax - xmin
        crop_aspect = crop_w / crop_h
        resize_aspect = W_resized / H_resized

        for n in range(N):
            img = padded_imgs[idx, n]  # (1, H, W)

            if crop_aspect >= resize_aspect:
                scale = W_resized / crop_w
                new_h = int(crop_h * scale)
                pad_y = (H_resized - new_h) // 2
                crop_resized = img[:, pad_y:pad_y + new_h, :]
                restored_crop = F.interpolate(
                    crop_resized.unsqueeze(0),
                    size=(crop_h, crop_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                scale = H_resized / crop_h
                new_w = int(crop_w * scale)
                pad_x = (W_resized - new_w) // 2
                crop_resized = img[:, :, pad_x:pad_x + new_w]
                restored_crop = F.interpolate(
                    crop_resized.unsqueeze(0),
                    size=(crop_h, crop_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            if invalid_val is not None:
                restored_img = torch.full((1, H_final, W_final), invalid_val, dtype=img.dtype, device=img.device)
            else:
                restored_img = torch.zeros((1, H_final, W_final), dtype=img.dtype, device=img.device)

            restored_img[:, ymin:ymax, xmin:xmax] = restored_crop
            restored_samples.append(restored_img)

        # Stack N crops: (N, 1, 1000, 1000)
        output.append(torch.stack(restored_samples))

    return torch.stack(output)


def resize_with_aspect_ratio_and_pad(img: torch.Tensor, target_size=224):
    """
    Resize [C, H, W] tensor while preserving aspect ratio, then pad to (target_size, target_size)
    """
    C, H, W = img.shape

    # Determine new size
    if H > W:
        new_h = target_size
        new_w = int(W * target_size / H)
    else:
        new_w = target_size
        new_h = int(H * target_size / W)

    img_resized = TF.resize(img, [new_h, new_w])

    # Compute padding to center the image
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    padding = [
        pad_w // 2, pad_w - pad_w // 2,  # left, right
        pad_h // 2, pad_h - pad_h // 2  # top, bottom
    ]
    img_padded = F.pad(img_resized, padding, mode='constant', value=0)

    return img_padded, padding


class MetricTool:
    def __init__(self, work_dir: str):
        self.metrics = []
        self.work_dir = work_dir

    def add(self, metrics: Dict[str, torch.Tensor]):
        self.metrics.append(
            {k: v.cpu().numpy() for k, v in metrics.items()}
        )

    def clear(self):
        self.metrics.clear()

    def summary(self, postfix: str = ''):
        df = pd.DataFrame(self.metrics, dtype=np.float32)

        avg = df.mean()
        print('-' * 32)
        print(f'Test samples:{len(self.metrics)}.')
        print(avg)
        print('-' * 32)

        # save to csv
        if postfix is None:
            postfix = ''

        if len(postfix) > 0:
            postfix = '_' + postfix

        # create work dir
        os.makedirs(self.work_dir, exist_ok=True)

        avg.to_csv(
            os.path.join(self.work_dir, f'result_avg{postfix}.csv'),
            header=False,
        )
        df.to_csv(
            os.path.join(self.work_dir, f'result{postfix}.csv')
        )


class FrequencySparseRegularity(nn.Module):
    def __init__(self, patch_size: int):
        super().__init__()

        self.patch_size = patch_size
        meshgrid = torch.meshgrid(
            torch.arange(self.patch_size),
            torch.arange(self.patch_size),
            indexing='ij'
        )
        weight = meshgrid[0] + meshgrid[1]
        weight = 1.2 ** weight - 1.0
        weight = weight / weight.sum()
        self.register_buffer('_weight', weight.flatten(), persistent=True)  # (P * P)

    def forward(self, x: torch.Tensor):
        """
        Compute regularity
        :param x: (..., P, P)
        :return:
        """
        assert x.shape[-2:] == (self.patch_size, self.patch_size)

        loss = (x.flatten(-2).abs() * self._weight).sum(-1)
        return loss.mean()


class SmoothRegularity(nn.Module):
    def __init__(self):
        super().__init__()

        self.huber = nn.HuberLoss('none', math.log(1.01))

    def forward(self, depth_log: torch.Tensor, image: torch.Tensor):
        """
        Compute smooth loss
        :param depth_log: (B, 1, H, W), depth in log space
        :param image: (B, 3, H, W)
        :return:
        """
        grad_depth_x = (depth_log[:, :, :, :-1] - depth_log[:, :, :, 1:]).abs()
        grad_depth_y = (depth_log[:, :, :-1, :] - depth_log[:, :, 1:, :]).abs()
        # # using huber loss
        # grad_depth_x = self.huber(grad_depth_x, torch.zeros_like(grad_depth_x))
        # grad_depth_y = self.huber(grad_depth_y, torch.zeros_like(grad_depth_y))

        image = image.mean(dim=1, keepdim=True)
        grad_img_x = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        grad_img_y = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()


def shift_image(x: torch.Tensor, shift: int):
    B, C, H, W = x.shape
    if shift >= 0:
        return F.pad(x, (shift, 0, shift, 0), 'constant', 0.)[:, :, : H, : W]
    else:
        return F.pad(x, (0, -shift, 0, -shift), 'constant', 0.)[:, :, -H:, -W:]


import open3d as o3d
def pcd_block(depth):
    pcd = []
    cx = 999 / 2
    cy = 999 / 2
    fx = 500.0
    fy = 500.0
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
    pts = torch.from_numpy(np.array(pc.points).type(torch.float32).cuda().unsqueeze(0))
    # pc = pc.farthest_point_down_sample(2048)
    return pts

