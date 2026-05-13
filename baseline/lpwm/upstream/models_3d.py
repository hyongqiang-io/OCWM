"""
3D-LPWM: Extension of DLP with 3D particle representation.

Minimal modification strategy:
- z_p (2D image coords) + z_depth (1D) are reinterpreted as 3D position
- Decoder applies perspective scale: z_scale_proj = z_scale * ref_depth / softplus(z_depth)
- Cross-view reconstruction loss provides 3D learning signal
- Dynamics/Context modules remain unchanged
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import DLP
from utils.loss_functions import calc_reconstruction_loss
from modules.vision_modules import rgb_to_minusoneone, minusoneone_to_rgb


class DLP3D(DLP):
    """
    3D-LPWM: Extends DLP with perspective-aware decoding and cross-view loss.

    Key differences from DLP:
    - z_depth is treated as physical depth (camera Z), not compositing order
    - Decoder applies perspective scale adjustment (far objects -> smaller patches)
    - Cross-view reconstruction loss when n_views > 1
    - Camera intrinsics/extrinsics used for un-project/re-project
    """

    def __init__(self,
                 # 3D-specific parameters
                 use_3d_particles=True,
                 perspective_scale=True,
                 lambda_cross_view=1.0,
                 depth_ref=1.0,
                 camera_fx=64.0,
                 camera_fy=64.0,
                 camera_cx=64.0,
                 camera_cy=64.0,
                 **kwargs):
        """
        Args:
            use_3d_particles: Enable 3D particle mode
            perspective_scale: Apply perspective scale to patches (far=smaller)
            lambda_cross_view: Weight for cross-view reconstruction loss
            depth_ref: Reference depth for perspective scaling normalization
            camera_fx/fy/cx/cy: Camera intrinsic parameters (pixels)
        """
        super().__init__(**kwargs)

        self.use_3d_particles = use_3d_particles
        self.perspective_scale = perspective_scale
        self.lambda_cross_view = lambda_cross_view
        self.depth_ref = depth_ref

        self.register_buffer('camera_K', torch.tensor([
            [camera_fx, 0.0, camera_cx],
            [0.0, camera_fy, camera_cy],
            [0.0, 0.0, 1.0]
        ], dtype=torch.float32))

        self.register_buffer('camera_T_1to2', torch.eye(4, dtype=torch.float32))

    def set_camera_params(self, K=None, T_1to2=None):
        """Set camera parameters (call after init with dataset info)."""
        if K is not None:
            if not isinstance(K, torch.Tensor):
                K = torch.tensor(K, dtype=torch.float32)
            self.camera_K.copy_(K)
        if T_1to2 is not None:
            if not isinstance(T_1to2, torch.Tensor):
                T_1to2 = torch.tensor(T_1to2, dtype=torch.float32)
            self.camera_T_1to2.copy_(T_1to2)

    def _get_positive_depth(self, z_depth):
        """Convert raw z_depth to positive physical depth."""
        return F.softplus(z_depth) + 0.1

    def perspective_scale_adjustment(self, z_scale, z_depth):
        """
        Apply perspective foreshortening to patch scale.
        Objects farther from camera appear smaller.

        z_scale: [..., 2]
        z_depth: [..., 1]
        Returns: adjusted z_scale with same shape
        """
        if not self.perspective_scale:
            return z_scale
        depth_positive = self._get_positive_depth(z_depth)
        depth_factor = self.depth_ref / depth_positive
        return z_scale * depth_factor

    def unproject_to_3d(self, z_2d, z_depth, K=None):
        """
        Un-project 2D keypoints + depth to 3D camera coordinates.

        z_2d: [..., 2] in [-1, 1] (normalized image coords)
        z_depth: [..., 1] (raw depth from encoder)
        K: [3, 3] intrinsic matrix
        Returns: [..., 3] in camera frame
        """
        if K is None:
            K = self.camera_K
        depth = self._get_positive_depth(z_depth)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        img_size = float(self.image_size)

        # [-1, 1] -> pixel coords
        u = (z_2d[..., 0:1] + 1.0) * 0.5 * img_size
        v = (z_2d[..., 1:2] + 1.0) * 0.5 * img_size

        # pixel -> camera 3D
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth

        return torch.cat([x, y, z], dim=-1)

    def project_to_2d(self, z_3d, K=None):
        """
        Project 3D points to 2D normalized coords and depth.

        z_3d: [..., 3] in camera frame
        K: [3, 3] intrinsic matrix
        Returns: (z_2d [..., 2] in [-1, 1], depth [..., 1])
        """
        if K is None:
            K = self.camera_K
        img_size = float(self.image_size)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        x, y, z = z_3d[..., 0:1], z_3d[..., 1:2], z_3d[..., 2:3]
        z_safe = z.clamp(min=0.1)

        # 3D -> pixel
        u = fx * x / z_safe + cx
        v = fy * y / z_safe + cy

        # pixel -> [-1, 1]
        u_norm = u / (0.5 * img_size) - 1.0
        v_norm = v / (0.5 * img_size) - 1.0

        z_2d = torch.cat([u_norm, v_norm], dim=-1)
        return z_2d, z_safe

    def transform_to_view2(self, z_3d, T_1to2=None):
        """
        Transform 3D points from view1 camera frame to view2.

        z_3d: [..., 3]
        T_1to2: [4, 4] rigid transform
        Returns: [..., 3]
        """
        if T_1to2 is None:
            T_1to2 = self.camera_T_1to2
        R = T_1to2[:3, :3]
        t = T_1to2[:3, 3]
        return torch.einsum('ij,...j->...i', R, z_3d) + t

    def cross_view_render(self, z, z_depth, z_scale, z_features, z_obj_on,
                          z_bg_features, z_ctx=None, K1=None, K2=None, T_1to2=None):
        """
        Render particles encoded from view1 into view2's camera.

        Steps:
        1. Un-project (z_2d, z_depth) -> 3D in view1 frame
        2. Transform 3D -> view2 frame
        3. Project to view2 2D coords + new depth
        4. Decode with perspective-adjusted scale
        """
        if K1 is None:
            K1 = self.camera_K
        if K2 is None:
            K2 = self.camera_K

        # 1. Un-project to 3D
        z_3d = self.unproject_to_3d(z, z_depth, K=K1)
        # 2. Transform
        z_3d_v2 = self.transform_to_view2(z_3d, T_1to2=T_1to2)
        # 3. Project to view2
        z_v2, depth_v2 = self.project_to_2d(z_3d_v2, K=K2)

        # Convert positive depth back to raw form for the decoder
        # inverse of softplus: log(exp(x) - 1), but we use the depth directly for ordering
        z_depth_v2 = torch.log(torch.clamp(depth_v2 - 0.1, min=1e-6))

        # 4. Perspective scale in view2
        if self.perspective_scale:
            z_scale_v2 = self.perspective_scale_adjustment(z_scale, z_depth_v2)
        else:
            z_scale_v2 = z_scale

        # Decode
        dec_dict = self.decoder_module(z_v2, z_scale_v2, z_features, z_obj_on,
                                       z_depth_v2, z_bg_features, z_ctx)
        rec = dec_dict['rec']
        if self.normalize_rgb:
            dec_dict['rec_rgb'] = minusoneone_to_rgb(rec)
        else:
            dec_dict['rec_rgb'] = rec
        return dec_dict

    def compute_cross_view_loss(self, x_view2, z, z_depth, z_scale, z_features,
                                z_obj_on, z_bg_features, z_ctx=None,
                                recon_loss_type="mse", recon_loss_func=None):
        """
        Cross-view reconstruction loss:
        encode(view1) -> 3D particles -> render(view2) -> compare with GT view2.

        x_view2: [bs, T, C, H, W] ground truth view2 images
        Returns: scalar loss
        """
        cross_dec = self.cross_view_render(z, z_depth, z_scale, z_features,
                                           z_obj_on, z_bg_features, z_ctx)
        rec_v2 = cross_dec['rec']

        x_target = x_view2
        if self.normalize_rgb:
            x_target = rgb_to_minusoneone(x_target)

        # Flatten time dim for loss
        if len(x_target.shape) == 5:
            x_target = x_target.reshape(-1, *x_target.shape[2:])
        if len(rec_v2.shape) == 5:
            rec_v2 = rec_v2.reshape(-1, *rec_v2.shape[2:])

        if recon_loss_func is not None:
            loss = recon_loss_func(rec_v2, x_target).mean()
        else:
            loss = calc_reconstruction_loss(x_target, rec_v2, loss_type=recon_loss_type,
                                            reduction='mean')
        return loss

    def decode_all(self, z, z_scale, z_features, obj_on_sample, z_depth, z_bg_features,
                   z_ctx, warmup=False, filter_key=None):
        """Override: applies perspective scale adjustment before decoding."""
        if self.use_3d_particles and self.perspective_scale:
            z_scale = self.perspective_scale_adjustment(z_scale, z_depth)

        return super().decode_all(z, z_scale, z_features, obj_on_sample, z_depth,
                                  z_bg_features, z_ctx, warmup, filter_key)

    def forward(self, x, deterministic=False, warmup=False, with_loss=False, beta_kl=0.1,
                beta_dyn=0.1, beta_rec=1.0, kl_balance=0.001, dynamic_discount=None,
                recon_loss_type="mse", recon_loss_func=None, balance=0.5, beta_dyn_rec=1.0,
                num_static=None, actions=None, actions_mask=None, lang_embed=None,
                beta_obj=0.0, done_mask=None, x_goal=None,
                x_view2=None):
        """
        Extended forward pass with cross-view loss.

        x_view2: [bs, T, C, H, W] second view images (for cross-view loss)
        """
        output_dict = super().forward(
            x, deterministic=deterministic, warmup=warmup, with_loss=with_loss,
            beta_kl=beta_kl, beta_dyn=beta_dyn, beta_rec=beta_rec,
            kl_balance=kl_balance, dynamic_discount=dynamic_discount,
            recon_loss_type=recon_loss_type, recon_loss_func=recon_loss_func,
            balance=balance, beta_dyn_rec=beta_dyn_rec, num_static=num_static,
            actions=actions, actions_mask=actions_mask, lang_embed=lang_embed,
            beta_obj=beta_obj, done_mask=done_mask, x_goal=x_goal
        )

        # Cross-view loss: only when training with dual-view data
        if x_view2 is not None and self.use_3d_particles and with_loss:
            z = output_dict['z']
            z_depth = output_dict['z_depth']
            z_scale = output_dict['z_scale']
            z_features = output_dict['z_features']
            z_obj_on = output_dict['obj_on']
            z_bg_features = output_dict['z_bg_features']
            z_context = output_dict['z_context']

            loss_cross_view = self.compute_cross_view_loss(
                x_view2, z, z_depth, z_scale, z_features, z_obj_on,
                z_bg_features, z_ctx=z_context,
                recon_loss_type=recon_loss_type, recon_loss_func=recon_loss_func
            )

            output_dict['loss_cross_view'] = loss_cross_view

            # Add to total loss in loss_dict
            if output_dict.get('loss_dict') is not None:
                output_dict['loss_dict']['loss_cross_view'] = loss_cross_view
                output_dict['loss_dict']['loss'] = (
                    output_dict['loss_dict']['loss'] +
                    self.lambda_cross_view * loss_cross_view
                )
        else:
            output_dict['loss_cross_view'] = torch.tensor(0.0, device=x.device)

        return output_dict
