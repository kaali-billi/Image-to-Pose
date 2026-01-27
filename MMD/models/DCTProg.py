import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.DCDepth import DCDepth
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from torchvision import transforms

from utils import post_process_depth, flip_lr, compute_errors_pth, colormap, inv_normalize, colormap_magma
from .registry import MODELS
from .utils import SmoothRegularity, resize_with_aspect_ratio_and_pad, restore_crops
from thop import profile
from extensions.emd import emd
import open3d as o3d
EMD = emd().cuda()


def pcd_block_pred(depth, fx=500, fy=500, cx=999 / 2, cy=999 / 2,
                   crop_box=None, original_size=(1000, 1000), resized_size=(224, 224)):
    """
    Project a cropped and resized depth image to 3D point cloud.

    Args:
        depth: Depth map of shape (224, 224) or (resized_size)
        fx, fy: Focal lengths in pixels for original image
        cx, cy: Principal point for original image
        crop_box: Tuple of (x_min, y_min, x_max, y_max) in original image coordinates
        original_size: (height, width) of original image before crop
        resized_size: (height, width) after resize (should match depth.shape)
    """

    # If no crop box provided, assume full image was used
    if crop_box is None:
        crop_box = (0, 0, original_size[1], original_size[0])  # (x_min, y_min, x_max, y_max)

    for b in range(1):
        y_min, x_min, y_max, x_max = crop_box[b]

    crop_width = x_max - x_min
    crop_height = y_max - y_min

    # Calculate scale factors from crop to resized image
    scale_x = resized_size[1] / crop_width  # width scale
    scale_y = resized_size[0] / crop_height  # height scale

    # Adjust intrinsics for the resized crop
    # First, shift principal point to crop coordinates
    cx_crop = cx - x_min
    cy_crop = cy - y_min

    # Then scale everything to resized dimensions
    fx_scaled = fx * scale_x
    fy_scaled = fy * scale_y
    cx_scaled = cx_crop * scale_x
    cy_scaled = cy_crop * scale_y

    height, width = depth.shape
    pcd = []

    for i in range(height):
        for j in range(width):
            z = depth[i][j]
            if z <= 0:  # Skip invalid depth values
                continue

            x = (j - cx_scaled) * z / fx_scaled
            y = (i - cy_scaled) * z / fy_scaled
            pcd.append([x, y, z])

    if len(pcd) == 0:
        # Return empty point cloud if no valid points
        return o3d.geometry.PointCloud()

    pc_EXT = o3d.geometry.PointCloud()
    pc_EXT.points = o3d.utility.Vector3dVector(pcd)

    # Rotate 180 degrees around X axis
    R = pc_EXT.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pc_EXT = pc_EXT.rotate(R, center=(0, 0, 0))

    # Filter points with z < 0 (after rotation)
    abc = []
    for coord in np.array(pc_EXT.points):
        x, y, z = coord
        if z < 0:
            abc.append(coord)

    if len(abc) < 20:  # Not enough points for filtering
        return o3d.geometry.PointCloud()

    pc_unfil = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(abc))

    # Statistical outlier removal
    cl, ind = pc_unfil.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)

    # Apply mask to filter points
    inlier_mask = np.zeros(len(pc_unfil.points), dtype=bool)
    inlier_mask[ind] = True
    abc = np.array(abc)
    filtered_pc = abc[inlier_mask]

    if len(filtered_pc) < 10:  # Not enough points after filtering
        return o3d.geometry.PointCloud()

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(filtered_pc))

    # Resample to exactly 2048 points
    if len(pc.points) < 2048:
        # Upsample if too few points
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(upsample_pc(filtered_pc, 2048)))
    else:
        # Downsample if too many points
        pcd = pc.farthest_point_down_sample(2048)

    pcd.paint_uniform_color([1, 0, 0])
    return pcd

class SILogLossInstance(nn.Module):
    def __init__(self, variance_focus: float, patch_size: int = 8, min_valid_pixels: int = 4, square_root: bool = True):
        super().__init__()

        assert 0 <= variance_focus <= 1.
        self.variance_focus = variance_focus

        self.patch_size = patch_size
        self.min_valid_pixels = min_valid_pixels
        self.square_root = square_root
        self.register_buffer(
            '_weight',
            torch.ones(1, 1, self.patch_size, self.patch_size, dtype=torch.float32)
        )

    def forward(self, depth_log: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, **kwargs):
        """
        Compute the silog loss
        :param depth_log: depth prediction in log space
        :param depth_gt: depth ground truth in metric
        :param mask: valid mask, binary
        :return:
        """
        mask = mask.float()
        assert depth_log.shape == mask.shape

        # filter mask
        if self.min_valid_pixels > 0:
            patch_mask = F.conv2d(mask, self._weight, stride=self.patch_size)
            patch_mask = (patch_mask >= self.min_valid_pixels).float()
            patch_mask = patch_mask.repeat_interleave(self.patch_size, dim=-1).repeat_interleave(self.patch_size,
                                                                                                 dim=-2)
            mask = mask * patch_mask

        B, _, H, W = depth_log.shape
        # convert gt to log space
        depth_gt = torch.log(depth_gt.clamp_min(1.0e-4))
        # flatten
        depth_log = depth_log.flatten(1)
        depth_gt = depth_gt.flatten(1)
        mask = mask.flatten(1)
        # compute difference
        diff = (depth_log - depth_gt) * mask
        # compute silog loss for each sample
        num = mask.sum(1)
        loss = diff.square().sum(1) / num - self.variance_focus * (diff.sum(1) / num).square()  # (B,)

        if self.square_root:
            loss = loss.sqrt()
        loss = 10. * loss
        # compute weight
        loss = loss.mean()

        return loss


@MODELS.register_module()
class DCTProg(LightningModule):
    """
    Bisection depth model
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.patch_size = 8
        self.max_depth = self.cfg.dataset.max_depth
        self.min_depth = self.cfg.dataset.min_depth
        self.output_space = self.cfg.model.output_space

        if cfg.training.norm:
            self.scale = 1
            print("Normalized Depth Training Scale")
        else:
            self.scale = (math.log(self.max_depth) if self.output_space == 'log' else self.max_depth)
        # output space, (metric or log)
        assert self.output_space in ['metric', 'log']

        # model
        self.model = DCDepth(
            self.cfg.model.encoder,
            self.cfg.model.pretrain,
            scale=self.scale,
            img_size=(self.cfg.dataset.input_height, self.cfg.dataset.input_width),
            ape=self.cfg.model.ape,
            drop_path_rate=self.cfg.model.drop_path_rate,
            drop_path_rate_crf=self.cfg.model.drop_path_rate_crf,
            seq_dropout_rate=self.cfg.model.seq_dropout_rate
        )

        # loss
        self.si_log = SILogLossInstance(self.cfg.loss.variance_focus, self.patch_size,
                                        self.cfg.loss.min_valid_pixels, self.cfg.loss.square_root)

        self.smooth_regularity = SmoothRegularity()
        self.beta = self.cfg.loss.beta
        if self.beta is not None:
            assert 0.5 <= self.beta <= 1.5
        self.epsilon = 1e-6
        self.delta = 0.2
        self.total_steps = None
        print(f'Output Space={self.output_space}.')

    def output2metric(self, out: torch.Tensor):
        """
        Convert output in metric or log space to metric depth
        :param out:
        :return:
        """
        if self.output_space == 'log':
            return out.exp()
        elif self.output_space == 'metric':
            return out
        else:
            raise NotImplementedError

    def output2log(self, out: torch.Tensor):
        """
        Convert output in metric or log space to log space
        :param out:
        :return:
        """
        if self.output_space == 'log':
            return out
        elif self.output_space == 'metric':
            return out.clamp_min(1.0e-4).log()
        else:
            raise NotImplementedError

    def configure_optimizers(self):
        self.total_steps = self.trainer.estimated_stepping_batches

        optimizer = AdamW(
            [
                {
                    'params': self.model.parameters_5x(),
                    'lr': self.cfg.optimization.max_lr,
                    'weight_decay': 0.
                },
                {
                    'params': self.model.parameters_1x(),
                    'lr': self.cfg.optimization.max_lr / self.cfg.optimization.lr_ratio,
                    'weight_decay': self.cfg.optimization.weight_decay
                },
            ]
        )
        # scheduler
        lrs = [group['lr'] for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, lrs, self.total_steps, div_factor=self.cfg.optimization.div_factor,
            final_div_factor=self.cfg.optimization.final_div_factor, pct_start=self.cfg.optimization.pct_start,
            anneal_strategy=self.cfg.optimization.anneal_strategy
        )

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    @torch.no_grad()
    def log_images(self, image, image_aug, depth_preds, depth_gt):
        writer = self.logger.experiment

        depth_gt = torch.where(depth_gt < 1e-3, depth_gt * 0 + 1e-3, depth_gt)
        global_step = self.global_step

        # visualize rgb
        writer.add_image(f'train/image', inv_normalize(image[0, :, :, :]), global_step)
        writer.add_image(f'train/image_aug', inv_normalize(image_aug[0, :, :, :]), global_step)

        # visualize depth
        n_pred = len(depth_preds)
        if self.cfg.dataset.name in ['nyu', 'tofdc', 'Dream_Chaser', 'DC_Close','LRO_1']:
            writer.add_image(f'train/depth_gt', colormap(depth_gt[0, :, :, :]), global_step)

            for idx in range(n_pred):
                depth_pred = self.output2metric(depth_preds[idx].detach()[0])
                writer.add_image(f'train/depth_pred_{idx}', colormap(depth_pred), global_step)

        else:
            writer.add_image(f'train/depth_gt', colormap_magma(torch.log10(depth_gt[0, :, :, :])), global_step)

            for idx in range(n_pred):
                depth_pred = self.output2metric(depth_preds[idx].detach()[0])
                writer.add_image(f'train/depth_pred_{idx}', colormap_magma(torch.log10(depth_pred)), global_step)

    def get_exponential_weights(self, n: int):
        xs = np.arange(n)
        ys = self.beta ** xs
        ys = ys / ys.sum()
        weights = ys.tolist()
        return list(reversed(weights))

    def on_train_epoch_start(self) -> None:
        torch.cuda.empty_cache()

    @staticmethod
    def crop_roi_from_segmentation(segmentation_map, image=None, padding=0, target_size=224):
        """
        Crop the region of interest (ROI) from the segmentation map.
        Args:
            segmentation_map: (B, 1, H, W) or (1, H, W) tensor with binary mask
            image: (Optional) (B, C, H, W) tensor of corresponding image
            padding: extra pixels to add around the bounding box
            target_size: Resized / Padded target
        Returns:
            Cropped segmentation map, and cropped image if provided
        """
        image = image * segmentation_map
        if segmentation_map.dim() == 4:
            segmentation_map = segmentation_map.squeeze(1)  # [B, H, W]
        B, H, W = segmentation_map.shape
        cropped_images = []
        crop_boxes = []
        pads = []
        for b in range(B):
            mask = segmentation_map[b]  # [H, W]
            nonzero = torch.nonzero(mask, as_tuple=False)

            if nonzero.size(0) == 0:
                # No ROI found, return black image
                c = image.shape[1] if image is not None else 3
                cropped_images.append(torch.zeros((c, target_size, target_size), dtype=torch.float32))
                continue

            y_min, x_min = nonzero.min(0)[0]
            y_max, x_max = nonzero.max(0)[0]

            # Apply padding
            y_min = max(y_min.item() - padding, 0)
            y_max = min(y_max.item() + padding, H)
            x_min = max(x_min.item() - padding, 0)
            x_max = min(x_max.item() + padding, W)

            if image is not None:
                img = image[b]  # [C, H, W]
                crop_img = img[:, y_min:y_max, x_min:x_max]  # [C, h, w]

                # Apply aspect-ratio resize and pad
                crop_img, padd = resize_with_aspect_ratio_and_pad(crop_img, target_size=target_size)
                cropped_images.append(crop_img)
                pads.append(padd)

            crop_boxes.append([y_min, x_min, y_max, x_max])

        return torch.stack(cropped_images, dim=0), crop_boxes, pads  # x_min, y_min, x_max, y_max,

    @staticmethod
    def normalize_depth_map(depth, invalid_val=-1.0):
        """
        Normalize each depth map in a batch to [0, 1], ignoring invalid values.

        Args:
            depth: Tensor (B, 1, H, W)
            invalid_val: float, value representing invalid depth (e.g., -1)

        Returns:
            normalized_depth: Tensor (B, 1, H, W)
            min_vals: Tensor (B,) – min valid depth per image
            max_vals: Tensor (B,) – max valid depth per image
        """
        B, _, H, W = depth.shape
        depth_flat = depth.view(B, -1)
        valid_mask = depth_flat > invalid_val

        normalized = torch.full_like(depth_flat, invalid_val)
        min_vals = torch.full((B,), float('nan'), device=depth.device)
        max_vals = torch.full((B,), float('nan'), device=depth.device)

        for b in range(B):
            valid_vals = depth_flat[b][valid_mask[b]]
            if valid_vals.numel() > 0:
                min_d = valid_vals.min()
                max_d = valid_vals.max()
                range_d = max_d - min_d

                if range_d > 0:
                    normalized[b][valid_mask[b]] = (valid_vals - min_d) / range_d
                else:
                    normalized[b][valid_mask[b]] = 0.0  # all valid pixels same

                min_vals[b] = min_d
                max_vals[b] = max_d

        return normalized.view(B, 1, H, W), min_vals, max_vals

    @staticmethod
    def unnormalize_depth_map(norm_depth, min_depth, max_depth):
        """
        Unnormalize a batch of normalized depth maps using per-image min and max.

        Args:
            norm_depth: (B, 1, H, W) tensor with values in [0, 1]
            min_depth: (B,) or (B, 1, 1, 1) broadcastable tensor
            max_depth: (B,) or (B, 1, 1, 1) broadcastable tensor

        Returns:
            depth: (B, 1, H, W) unnormalized depth maps
        """
        # Ensure shapes are broadcastable
        if min_depth.ndim == 1:
            min_depth = min_depth.view(-1, 1, 1, 1)
        if max_depth.ndim == 1:
            max_depth = max_depth.view(-1, 1, 1, 1)

        norm_depth = norm_depth.clamp(0.0, 1.0)

        depth = norm_depth * (max_depth - min_depth) + min_depth
        return depth

    def depth_to_pointcloud(self, depth_batch):
        """
        Convert a depth map to a point cloud and normalize it by subtracting the centroid.

        Args:
            depth_batch: (B, 1, H, W) tensor with depth values. The second dimension
                         is assumed to be 1 for a single channel depth map.
            fx, fy: focal lengths (can be a single value or a batch of values if different per batch item)
            cx, cy: principal points (can be a single value or a batch of values if different per batch item)

        Returns:
            normalized_filtered_pointclouds: A list of (N_i, 3) tensors, where N_i is the number of valid points
                                             for the i-th batch item, and each point cloud is normalized
                                             by subtracting its centroid.
        """

        # Assuming these are constant for now, but the ensure_tensor function handles batching if they were
        # passed in as tensors of shape (B,) or (B,1)
        cx_val = 999 / 2
        cy_val = 999 / 2
        fx_val = 500.0
        fy_val = 500.0

        B, _, H, W = depth_batch.shape
        device = depth_batch.device

        # Create mesh grid of pixel coordinates (u, v)
        u = torch.arange(0, W, device=device).float()
        v = torch.arange(0, H, device=device).float()
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')  # each (H, W)

        # Flatten to (N,), where N = H * W
        grid_u = grid_u.reshape(-1)  # (N,)
        grid_v = grid_v.reshape(-1)  # (N,)
        N = H * W

        # Expand to (B, N)
        grid_u = grid_u.unsqueeze(0).expand(B, N)
        grid_v = grid_v.unsqueeze(0).expand(B, N)

        # Flatten depth to (B, N)
        depth = depth_batch.reshape(B, N)

        # Broadcast intrinsics
        def ensure_tensor(x, batch_size, device_):
            if torch.is_tensor(x):
                if x.numel() == 1:
                    return x.expand(batch_size).view(batch_size, 1).to(device_)
                return x.view(batch_size, 1).to(device_)
            else:
                return torch.full((batch_size, 1), float(x), device=device_)

        fx = ensure_tensor(fx_val, B, device).view(B, 1)
        fy = ensure_tensor(fy_val, B, device).view(B, 1)
        cx = ensure_tensor(cx_val, B, device).view(B, 1)
        cy = ensure_tensor(cy_val, B, device).view(B, 1)

        # Project to 3D
        X = (grid_u - cx) * depth / fx
        Y = (grid_v - cy) * depth / fy
        Z = depth

        # Stack into (B, N, 3)
        pointclouds = torch.stack((X, Y, Z), dim=2)  # (B, N, 3)
        valid_mask = (depth != -1)  # (B, N)

        normalized_filtered_pointclouds = []
        for i in range(B):
            # Filter out invalid points for the current batch item
            current_pc = pointclouds[i][valid_mask[i]]  # (N_i, 3)

            if current_pc.numel() > 0:  # Ensure there are valid points
                # Calculate centroid
                centroid = torch.mean(current_pc, dim=0, keepdim=True)  # (1, 3)

                # Subtract centroid for normalization
                normalized_pc = current_pc - centroid
                normalized_filtered_pointclouds.append(normalized_pc)
            else:
                # If no valid points, append an empty tensor with the correct shape
                normalized_filtered_pointclouds.append(torch.empty(0, 3, device=device))

        return normalized_filtered_pointclouds

    @staticmethod
    def sample_valid_points_batched(pointcloud_list, num_samples=2048, replace=False):
        """
        Sample a fixed number of points from a list of [N_i, 3] tensors.

        Returns:
            (B, num_samples, 3) tensor
        """
        B = len(pointcloud_list)
        device = pointcloud_list[0].device
        sampled = []

        for pc in pointcloud_list:
            N = pc.shape[0]
            if N == 0:
                sampled_pc = torch.zeros((num_samples, 3), device=device)
            elif N < num_samples and not replace:
                idx = torch.randint(0, N, (num_samples,), device=device)
                sampled_pc = pc[idx]
            else:
                idx = torch.randperm(N, device=device)[:num_samples] if not replace else \
                    torch.randint(0, N, (num_samples,), device=device)
                sampled_pc = pc[idx]
            sampled.append(sampled_pc)

        return torch.stack(sampled)

    def training_step(self, batch, batch_idx):
        # torch.autograd.set_detect_anomaly(True)
        # === Unpack Batch ===
        image = batch['image']  # [B, 3, H, W]
        mask = batch['mask']
        GT = batch['depth']  # [B, 1, H, W]

        # === Depth Ground Truth Normalization ===
        depth_gt, min_d, max_d = self.normalize_depth_map(GT)
        # === ROI Cropping ===
        roi, crop_boxes, pads = self.crop_roi_from_segmentation(mask, image)
        depth_roi, _, _ = self.crop_roi_from_segmentation(mask, depth_gt)
        # === Debug NaN/Inf checks ===
        if torch.isnan(depth_roi).any() or torch.isinf(depth_roi).any():
            print("NaN or Inf detected in cropped depth_roi")
            raise ValueError("Invalid values in depth_roi")

        # === Forward ===
        depths, freq_regs, min_p, max_p = self.model(image, roi)

        # Earth Movers Distance Loss for Structural learning
        depth_emd = depths[-1]
        depth_emd = restore_crops(depth_emd, crop_boxes, pads, invalid_val=-1)
        depth_emd[mask == 0] = -1
        gts = self.depth_to_pointcloud(depth_gt)
        gts = self.sample_valid_points_batched(gts)
        pred = self.depth_to_pointcloud(depth_emd)
        preds = self.sample_valid_points_batched(pred)
        emd_loss = EMD(gts, preds)

        # === Valid Mask ===
        valid_mask = depth_roi >= self.min_depth

        # === Smooth L1 Loss : Huber Loss at Beta=0.05 (5cm accuracy)
        D_loss = F.smooth_l1_loss(min_p, min_d, beta=self.delta) + F.smooth_l1_loss(max_p, max_d, beta=self.delta)
        abs_diff = torch.mean(torch.abs(min_d - min_p)) + torch.mean(torch.abs(max_d - max_p))
        # === Loss Accumulator ===

        total_loss = 0

        # === Depth Pyramid Processing ===
        weight_func = self.get_exponential_weights
        for idx, (depth, freq_reg, weight) in enumerate(zip(depths, freq_regs, weight_func(len(depths)))):
            # depth = depth.detach()

            # === Convert to log-space safely ===
            depth_log = depth.clamp_min(1.0e-4).log() # OUTPUT of DC_DEPTH.Depth_update is in Log Space

            assert depth_log.shape == depth_roi.shape, ("SHAPE ERROR", GT.shape, depth_log.shape, len(depths))

            # === SI-Log Loss ===
            si_log = self.si_log(depth_log, depth_roi, valid_mask)
            # === Smoothness Regularity ===
            if idx > 3:
                smooth_reg = self.smooth_regularity(depth_log, roi)
            else:
                smooth_reg = torch.zeros(1, dtype=si_log.dtype, device=si_log.device)

            # === Log Individual Losses ===
            self.log(f'loss/si_log_{idx}', si_log.item(), on_step=True, on_epoch=False)
            self.log(f'loss/freq_reg_{idx}', freq_reg.item(), on_step=True, on_epoch=False)
            self.log(f'loss/smooth_reg_{idx}', smooth_reg.item(), on_step=True, on_epoch=False)

            # === Accumulate Weighted Loss ===
            total_loss += weight * (
                    si_log +
                    self.cfg.loss.freq_reg_weight * freq_reg +
                    self.cfg.loss.smooth_reg_weight * smooth_reg
            )
        self.log(f'loss/D_LOSS', D_loss.item(), on_step=True, on_epoch=False)
        self.log(f'loss/EMD', emd_loss.item(), on_step=True, on_epoch=False)

        total_loss += D_loss + 0.5 * emd_loss
        # === Final NaN Check ===
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(D_loss)
            print("NaN or Inf detected in total loss!",si_log,freq_reg,emd_loss)
            print(f"D_gt: {max_d}")
            print(f"max_d: {max_p}")
            print(f"depth sample stats: min={depths[0].min()}, max={depths[0].max()}")
            raise ValueError("NaN/Inf in total_loss")

        # === Log Total Loss ===
        loss_dict = {
            'MAE': round(abs_diff.item(), 3),
            'siloss': round(si_log.item(), 3),
            'D_loss': round(D_loss.item(), 3),
            'EMD': round(emd_loss.item(), 3)
        }
        self.log('loss/total', loss_dict, on_step=True, on_epoch=False, prog_bar=True)

        # === Log Images ===
        if self.global_step % self.cfg.training.log_freq == 0:
            depths_P = [torch.tensor(d) for d in depths]
            self.log_images(image, roi, depths_P, depth_roi)

        # === Log LR ===
        optim = self.optimizers()
        for idx, group in enumerate(optim.optimizer.param_groups):
            self.log(f'learning_rate/group_{idx}', group['lr'])

        return total_loss

    def evaluate_depth(self, batch, vis):
        post_process = True
        # fetch data
        self.model.eval()
        image = batch['image'].to('cuda')  # [B, 3, H, W]
        mask = batch['mask'].to('cuda')
        GT = batch['depth'].to('cuda')  # [B, 1, H, W]
        # === Depth Ground Truth Normalization ===
        depth_gt, min_d, max_d = self.normalize_depth_map(GT)

        # === ROI Cropping ===
        roi, crop_boxes, pads = self.crop_roi_from_segmentation(mask, image)
        crroi,_,_ = self.crop_roi_from_segmentation(mask, mask)
        depth_roi, _, _ = self.crop_roi_from_segmentation(mask, depth_gt)
        # === Debug NaN/Inf checks ===
        if torch.isnan(depth_roi).any() or torch.isinf(depth_roi).any():
            print("NaN or Inf detected in cropped depth_roi")
            raise ValueError("Invalid values in depth_roi")

        # === Forward ===
        # print(image.is_cuda, roi.is_cuda)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Start timer
        start_event.record()
        from thop import clever_format

        # Run the model
        depths, min_p, max_p = self.model(image.float(), roi.float())  # max_p

        # End timer
        end_event.record()

        # Wait for everything to finish running
        torch.cuda.synchronize()

        # Calculate elapsed time (in milliseconds)
        elapsed_time_ms = start_event.elapsed_time(end_event)
        # === Valid Mask ===

        valid_mask = GT >= self.min_depth
        depths = restore_crops(depths, crop_boxes, pads, invalid_val=-1)
        pred_norm = depths * mask
        depths = self.unnormalize_depth_map(pred_norm, min_p, max_p)
        depths = depths * mask
        min_abs = (torch.abs(min_d - min_p))
        max_abs = (torch.abs(max_d - max_p))
        measures = compute_errors_pth(GT[valid_mask], depths[valid_mask])
        return measures, min_abs, max_abs, elapsed_time_ms, max_d.cpu().numpy(), depths.squeeze(0).cpu()


    def validation_step(self, batch, batch_idx, writer):
        self.evaluate_depth(batch, batch_idx, writer)


def crop2_roi_from_segmentation(segmentation_map, image=None, padding=0):
    """
    Crop the region of interest (ROI) from the segmentation map.
    Args:
        segmentation_map: (B, 1, H, W) or (1, H, W) tensor with binary mask
        image: (Optional) (B, C, H, W) tensor of corresponding image
        padding: extra pixels to add around the bounding box
    Returns:
        Cropped segmentation map, and cropped image if provided
    """

    if segmentation_map.dim() == 4:
        segmentation_map = segmentation_map.squeeze(1)  # remove channel dimension (B, H, W)

    B, H, W = segmentation_map.shape
    crops = []
    crops_img = []
    T = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        # transforms.ToTensor(),  # Convert to tensor
    ])
    for b in range(B):
        mask = segmentation_map[b]  # (H, W)
        nonzero = torch.nonzero(mask, as_tuple=False)

        if nonzero.size(0) == 0:
            # No object found
            crops.append(None)
            crops_img.append(None)
            continue

        y_min, x_min = nonzero.min(0)[0]
        y_max, x_max = nonzero.max(0)[0]

        # Apply padding
        y_min = max(y_min.item() - padding, 0)
        y_max = min(y_max.item() + padding, H)
        x_min = max(x_min.item() - padding, 0)
        x_max = min(x_max.item() + padding, W)

        crop = mask[y_min:y_max, x_min:x_max]
        crops.append(crop)

        if image is not None:
            img = image[b]
            crop_img = img[:, y_min:y_max, x_min:x_max]
            #crop_img = T(crop_img)
            crops_img.append(crop_img)

    if image is not None:
        return torch.stack(crops_img, dim=0)
    else:
        return crops