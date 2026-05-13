"""
Training script for 3D-LPWM.
Extends the standard LPWM training with cross-view reconstruction loss.

Key difference: when n_views=2, view1 is used for standard ELBO, and view2 is
passed as x_view2 for the cross-view loss (instead of flattening both views
into a single batch as in the original LPWM training).
"""
import numpy as np
import os
from tqdm import tqdm
import matplotlib
import argparse
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torch.optim as optim

from models_3d import DLP3D
from datasets.get_dataset import get_video_dataset
from utils.util_func import (plot_keypoints_on_image_batch, prepare_logdir, save_config,
                             log_line, get_config, format_epoch_summary,
                             plot_training_metrics, save_metrics_data, save_code_backup)
from utils.loss_functions import calc_reconstruction_loss, LossLPIPS
from eval.eval_model import evaluate_validation_elbo_dyn, animate_trajectory_lpwm
from eval.eval_gen_metrics import eval_lpwm_im_metric

matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def build_model_from_config(config, device):
    """Build DLP3D model from config dict."""
    # 3D-specific params
    use_3d = config.get('use_3d_particles', True)
    perspective_scale = config.get('perspective_scale', True)
    lambda_cross_view = config.get('lambda_cross_view', 1.0)
    depth_ref = config.get('depth_ref', 1.0)
    camera_fx = config.get('camera_fx', 64.0)
    camera_fy = config.get('camera_fy', 64.0)
    camera_cx = config.get('camera_cx', 64.0)
    camera_cy = config.get('camera_cy', 64.0)

    model = DLP3D(
        # 3D-specific
        use_3d_particles=use_3d,
        perspective_scale=perspective_scale,
        lambda_cross_view=lambda_cross_view,
        depth_ref=depth_ref,
        camera_fx=camera_fx,
        camera_fy=camera_fy,
        camera_cx=camera_cx,
        camera_cy=camera_cy,
        # Standard DLP params
        cdim=config['ch'],
        image_size=config['image_size'],
        normalize_rgb=config['normalize_rgb'],
        n_views=config.get('n_views', 1),
        n_kp_per_patch=config['n_kp_per_patch'],
        patch_size=config['patch_size'],
        anchor_s=config['anchor_s'],
        n_kp_enc=config['n_kp_enc'],
        n_kp_prior=config['n_kp_prior'],
        pad_mode=config['pad_mode'],
        dropout=config['dropout'],
        features_dist=config.get('features_dist', 'gauss'),
        learned_feature_dim=config['learned_feature_dim'],
        learned_bg_feature_dim=config.get('learned_bg_feature_dim', config['learned_feature_dim']),
        n_fg_categories=config.get('n_fg_categories', 8),
        n_fg_classes=config.get('n_fg_classes', 4),
        n_bg_categories=config.get('n_bg_categories', 4),
        n_bg_classes=config.get('n_bg_classes', 4),
        scale_std=config['scale_std'],
        offset_std=config['offset_std'],
        obj_on_alpha=config['obj_on_alpha'],
        obj_on_beta=config['obj_on_beta'],
        obj_res_from_fc=config['obj_res_from_fc'],
        obj_ch_mult_prior=config.get('obj_ch_mult_prior', config['obj_ch_mult']),
        obj_ch_mult=config['obj_ch_mult'],
        obj_base_ch=config['obj_base_ch'],
        obj_final_cnn_ch=config['obj_final_cnn_ch'],
        bg_res_from_fc=config['bg_res_from_fc'],
        bg_ch_mult=config['bg_ch_mult'],
        bg_base_ch=config['bg_base_ch'],
        bg_final_cnn_ch=config['bg_final_cnn_ch'],
        use_resblock=config['use_resblock'],
        num_res_blocks=config['num_res_blocks'],
        cnn_mid_blocks=config.get('cnn_mid_blocks', False),
        mlp_hidden_dim=config.get('mlp_hidden_dim', 256),
        pint_enc_layers=config['pint_enc_layers'],
        pint_enc_heads=config['pint_enc_heads'],
        timestep_horizon=config['timestep_horizon'],
        n_static_frames=config['num_static_frames'],
        predict_delta=config['predict_delta'],
        context_dim=config['context_dim'],
        ctx_dist=config.get('context_dist', 'gauss'),
        n_ctx_categories=config.get('n_ctx_categories', 8),
        n_ctx_classes=config.get('n_ctx_classes', 4),
        ctx_pool_mode=config.get('ctx_pool_mode', 'none'),
        pint_dyn_layers=config['pint_dyn_layers'],
        pint_dyn_heads=config['pint_dyn_heads'],
        pint_dim=config['pint_dim'],
        pint_ctx_layers=config['pint_ctx_layers'],
        pint_ctx_heads=config['pint_ctx_heads'],
        action_condition=config.get('action_condition', False),
        action_dim=config.get('action_dim', 0),
        null_action_embed=config.get('null_action_embed', False),
        random_action_condition=config.get('random_action_condition', False),
        random_action_dim=config.get('random_action_dim', 0),
        img_goal_condition=config.get('image_goal_condition', False),
    ).to(device)

    # Set camera extrinsics if provided
    T_1to2 = config.get('camera_T_1to2', None)
    if T_1to2 is not None:
        model.set_camera_params(T_1to2=T_1to2)

    return model


def train_3d_lpwm(config_path='./configs/panda_3d.json'):
    config = get_config(config_path)
    hparams = config
    device = config['device']
    if 'cuda' in device:
        device = torch.device(f'{device}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    # Key params
    ds = config['ds']
    n_views = config.get('n_views', 1)
    image_size = config['image_size']
    batch_size = config['batch_size']
    timestep_horizon = config['timestep_horizon']
    num_epochs = config['num_epochs']
    start_epoch = config.get('start_epoch', 0)
    warmup_epoch = config['warmup_epoch']
    eval_epoch_freq = config['eval_epoch_freq']
    run_prefix = config['run_prefix']

    # Loss params
    recon_loss_type = config['recon_loss_type']
    beta_kl = config['beta_kl']
    beta_dyn = config['beta_dyn']
    beta_rec = config['beta_rec']
    beta_dyn_rec = config['beta_dyn_rec']
    beta_obj = config.get('beta_obj', 0.0)
    kl_balance = config['kl_balance']
    num_static_frames = config['num_static_frames']
    img_goal_condition = config.get('image_goal_condition', False)
    use_ep_done_mask = config.get('use_ep_done_mask', False)

    # Dataset
    root = config['root']
    dataset = get_video_dataset(ds, root, seq_len=timestep_horizon + 1, mode='train',
                                image_size=image_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4,
                            pin_memory=True, drop_last=True)

    # Model
    model = build_model_from_config(config, device)
    print(f"[3D-LPWM] Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"[3D-LPWM] use_3d_particles={model.use_3d_particles}, "
          f"perspective_scale={model.perspective_scale}, "
          f"lambda_cross_view={model.lambda_cross_view}")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                           betas=tuple(config['adam_betas']),
                           eps=config['adam_eps'],
                           weight_decay=config['weight_decay'])

    if config['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1,
                                              gamma=config['scheduler_gamma'])
    else:
        scheduler = None

    # Logging
    logdir = prepare_logdir(run_prefix, config_path)
    save_config(hparams, logdir)
    save_code_backup(logdir)

    # Recon loss func
    recon_loss_func = LossLPIPS().to(device) if recon_loss_type == 'lpips' else None

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = {}
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch in pbar:
            x = batch[0].to(device)  # [bs, T, n_views, C, H, W] if multiview
            actions = batch[1].to(device) if config.get('action_condition', False) else None
            x_goal = batch[3].to(device) if img_goal_condition else None
            ep_done_mask = batch[4].to(device) if use_ep_done_mask else None

            warmup = (epoch < warmup_epoch)

            # Split views for 3D training
            x_view2 = None
            if n_views > 1 and len(x.shape) == 6:
                # x: [bs, T, n_views, C, H, W]
                x_view1 = x[:, :, 0]  # [bs, T, C, H, W]
                x_view2 = x[:, :, 1]  # [bs, T, C, H, W]
                x = x_view1

                if x_goal is not None and len(x_goal.shape) == 5:
                    # x_goal: [bs, n_views, C, H, W]
                    x_goal = x_goal[:, 0]

                if actions is not None and len(actions.shape) == 4:
                    # actions: [bs, T, n_views, action_dim]
                    actions = actions[:, :, 0]

                if ep_done_mask is not None and len(ep_done_mask.shape) == 3:
                    ep_done_mask = ep_done_mask[:, :, 0]

            elif n_views > 1:
                # Fallback: x is [bs, T, C, H, W] per view (already separated)
                x_view2 = None

            # Forward
            model_output = model(x, actions=actions, warmup=warmup, with_loss=True,
                                 beta_kl=beta_kl, beta_dyn=beta_dyn, beta_rec=beta_rec,
                                 kl_balance=kl_balance, recon_loss_type=recon_loss_type,
                                 recon_loss_func=recon_loss_func, beta_dyn_rec=beta_dyn_rec,
                                 beta_obj=beta_obj, done_mask=ep_done_mask, x_goal=x_goal,
                                 x_view2=x_view2)

            all_losses = model_output['loss_dict']
            loss = all_losses['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses
            for k, v in all_losses.items():
                if isinstance(v, torch.Tensor):
                    v = v.item()
                if k not in epoch_losses:
                    epoch_losses[k] = []
                epoch_losses[k].append(v)

            # Progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'cross_view': f"{model_output.get('loss_cross_view', torch.tensor(0.)).item():.4f}",
                'psnr': f"{all_losses.get('psnr', 0.):.2f}" if isinstance(
                    all_losses.get('psnr', 0.), float) else f"{all_losses.get('psnr', torch.tensor(0.)).item():.2f}"
            })

        if scheduler is not None:
            scheduler.step()

        # Epoch summary
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        log_msg = (f"Epoch {epoch} | loss: {avg_losses.get('loss', 0.):.4f} | "
                   f"psnr: {avg_losses.get('psnr', 0.):.2f} | "
                   f"cross_view: {avg_losses.get('loss_cross_view', 0.):.4f}")
        log_line(logdir, log_msg)
        print(log_msg)

        # Evaluation
        if (epoch + 1) % eval_epoch_freq == 0:
            model.eval()
            with torch.no_grad():
                # Save checkpoint
                ckpt_path = os.path.join(logdir, f"model_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                }, ckpt_path)

    # Final save
    final_path = os.path.join(logdir, "model_final.pth")
    torch.save({
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
    }, final_path)
    print(f"[3D-LPWM] Training complete. Final model saved to {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train 3D-LPWM')
    parser.add_argument('--config', type=str, default='./configs/panda_3d.json',
                        help='Path to config file')
    args = parser.parse_args()
    train_3d_lpwm(args.config)
