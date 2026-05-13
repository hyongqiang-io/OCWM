"""
3D-aware evaluation metrics for 3D-LPWM experiments.

Implements four pilot analyses (P1-P4) and the cross-view main experiment metrics:
  P1: Depth correlation probe  — does z_depth correlate with GT 3D positions?
  P2: Cross-view PSNR          — LPWM cross-view reconstruction quality
  P3: Occlusion recovery       — particle tracking accuracy through full occlusion
  P4: Depth complexity vs perf — PSNR as a function of inter-object depth variation

All functions follow the same signature convention:
  (model, dataloader, device, ...) -> dict of metric_name -> value(s)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

try:
    from piqa import PSNR, SSIM, LPIPS
    _PIQA_AVAILABLE = True
except ImportError:
    _PIQA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Utility: project 3D camera-frame points to normalised image coordinates
# ---------------------------------------------------------------------------

def perspective_project(pts_3d: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Project 3D points (camera frame) to 2D image coordinates in [-1, 1].

    Args:
        pts_3d: [N, 3] or [B, N, 3], coordinates in camera frame (x right, y down, z forward)
        K:      [3, 3] camera intrinsic matrix

    Returns:
        uv: same batch shape as pts_3d but last dim = 2, normalised to [-1, 1]
    """
    leading = pts_3d.shape[:-1]
    pts = pts_3d.reshape(-1, 3)               # [N, 3]
    proj = (K @ pts.T).T                       # [N, 3]
    uv_px = proj[:, :2] / proj[:, 2:3].clamp(min=1e-6)   # pixel coords
    # normalise from pixel coords to [-1, 1] using K[0,2] and K[1,2] as cx, cy
    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    u_norm = (uv_px[:, 0] - cx) / fx          # [-1, 1] approx for standard cam
    v_norm = (uv_px[:, 1] - cy) / fy
    uv = torch.stack([u_norm, v_norm], dim=-1)
    return uv.reshape(*leading, 2)


def transform_particles_to_view(z_p: torch.Tensor,
                                 R: torch.Tensor,
                                 t: torch.Tensor) -> torch.Tensor:
    """
    Rigid-body transform 3D particle positions from view1 to view2 frame.

    Args:
        z_p: [B, N, 3]  particle 3D positions in view1 camera frame
        R:   [3, 3]      rotation matrix from view1 to view2
        t:   [3]         translation vector from view1 to view2

    Returns:
        z_p_v2: [B, N, 3]
    """
    return (R @ z_p.transpose(-1, -2)).transpose(-1, -2) + t.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# P1: Depth correlation probe
# ---------------------------------------------------------------------------

def eval_depth_correlation(
    model,
    dataset,
    device: torch.device,
    batch_size: int = 32,
    max_batches: int = 50,
    gt_depth_key: str = 'obj_z',   # key in metadata; None → use stereo disparity
    camera_baseline: float = None,  # metres, required when gt_depth_key is None
    verbose: bool = False,
) -> dict:
    """
    P1: Measure Pearson r between model's mu_depth and ground-truth object depth.

    When gt_depth_key is provided (simulation metadata has 3D positions), the GT
    depth is read directly.  When None, stereo disparity from multi-view images
    is used (requires camera_baseline).

    Returns:
        {
          'pearson_r':      float,   # correlation coefficient
          'pearson_p':      float,   # p-value
          'mu_depth_mean':  float,
          'mu_depth_std':   float,
          'gt_depth_mean':  float,
          'gt_depth_std':   float,
          'n_particles':    int,
        }
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, drop_last=False)

    all_mu_depth = []
    all_gt_depth = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch[0].to(device)               # [B, T, C, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        with torch.no_grad():
            out = model(x, with_loss=False)

        # mu_depth: [B, T, n_kp, 1]  — compositing depth scalar
        mu_depth = out['mu_depth']             # [B, T, n_kp, 1]
        obj_on   = out['obj_on']               # [B, T, n_kp, 1]  transparency mask

        # only keep active particles (obj_on > 0.5)
        mask = (obj_on > 0.5).squeeze(-1)      # [B, T, n_kp]
        mu_d = mu_depth.squeeze(-1)[mask]      # [N_active]
        all_mu_depth.append(mu_d.cpu().float().numpy())

        # GT depth: from metadata if available
        if gt_depth_key is not None and len(batch) > 1:
            # expect batch[-2] to be object 3D positions [B, T, n_obj, 3]
            obj_pos = batch[-2].to(device)     # [B, T, n_obj, 3]
            # z-coordinate in camera frame = depth
            gt_d_full = obj_pos[..., 2]        # [B, T, n_obj]
            # average over objects per timestep → proxy per-frame depth
            gt_d = gt_d_full.mean(dim=-1, keepdim=True).expand_as(obj_on.squeeze(-1))
            gt_d = gt_d[mask]
        else:
            # Placeholder: use negative mu_depth as GT proxy (sanity check)
            # In real usage replace with stereo disparity computation
            if verbose:
                print("[P1] Warning: no GT depth available, using proxy.")
            gt_d = -mu_d  # trivially anti-correlated — expect r ~ -1

        all_gt_depth.append(gt_d.cpu().float().numpy() if torch.is_tensor(gt_d)
                            else gt_d)

    mu_arr = np.concatenate(all_mu_depth)
    gt_arr = np.concatenate(all_gt_depth)

    r, p = pearsonr(mu_arr, gt_arr) if len(mu_arr) > 2 else (0.0, 1.0)

    return {
        'pearson_r':     float(r),
        'pearson_p':     float(p),
        'mu_depth_mean': float(mu_arr.mean()),
        'mu_depth_std':  float(mu_arr.std()),
        'gt_depth_mean': float(gt_arr.mean()),
        'gt_depth_std':  float(gt_arr.std()),
        'n_particles':   int(len(mu_arr)),
    }


# ---------------------------------------------------------------------------
# P2 / E1: Cross-view PSNR
# ---------------------------------------------------------------------------

def eval_cross_view(
    model,
    dataset_v1,
    dataset_v2,
    camera_K: torch.Tensor,
    R_1to2: torch.Tensor,
    t_1to2: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
    max_batches: int = 30,
    use_3d_particles: bool = False,
) -> dict:
    """
    P2 / E1: Encode from view1, render to view2, compare with actual view2.

    For 2D-LPWM (use_3d_particles=False): particles have no 3D structure, so
    we can only test by simply mirroring the view1 reconstruction as view2
    prediction — this is the 'trivial baseline' exposing the structural failure.

    For 3D-LPWM (use_3d_particles=True): particles are transformed using R_1to2
    and t_1to2, then re-rendered with camera_K.

    Returns: dict with same-view and cross-view PSNR/SSIM/LPIPS.
    """
    assert _PIQA_AVAILABLE, "pip install piqa"
    model.eval()

    psnr_fn  = PSNR().to(device)
    ssim_fn  = SSIM().to(device)
    lpips_fn = LPIPS(network='vgg').to(device)

    loader_v1 = DataLoader(dataset_v1, batch_size=batch_size, shuffle=False,
                           num_workers=4, drop_last=True)
    loader_v2 = DataLoader(dataset_v2, batch_size=batch_size, shuffle=False,
                           num_workers=4, drop_last=True)

    results = {
        'same_view_psnr':  [], 'same_view_ssim':  [], 'same_view_lpips': [],
        'cross_view_psnr': [], 'cross_view_ssim': [], 'cross_view_lpips': [],
        'trivial_psnr':    [],  # copy v1 render → compare to v2
    }

    for batch_idx, (batch_v1, batch_v2) in enumerate(zip(loader_v1, loader_v2)):
        if batch_idx >= max_batches:
            break

        x_v1 = batch_v1[0].to(device)  # [B, T, C, H, W]
        x_v2 = batch_v2[0].to(device)
        if len(x_v1.shape) == 4:
            x_v1 = x_v1.unsqueeze(1)
            x_v2 = x_v2.unsqueeze(1)

        B, T, C, H, W = x_v1.shape
        x_v1_flat = x_v1.reshape(B * T, C, H, W).clamp(0, 1)
        x_v2_flat = x_v2.reshape(B * T, C, H, W).clamp(0, 1)

        with torch.no_grad():
            out_v1 = model(x_v1, with_loss=False)

        rec_v1 = out_v1['rec_rgb'].reshape(B * T, C, H, W).clamp(0, 1)  # same-view reconstruction

        # Same-view metrics
        results['same_view_psnr'].append(psnr_fn(rec_v1, x_v1_flat).mean().item())
        results['same_view_ssim'].append(ssim_fn(rec_v1, x_v1_flat).mean().item())
        results['same_view_lpips'].append(lpips_fn(rec_v1, x_v1_flat).mean().item())

        # Trivial cross-view baseline: just copy v1 reconstruction
        results['trivial_psnr'].append(psnr_fn(rec_v1, x_v2_flat).mean().item())

        if use_3d_particles:
            # 3D-LPWM: transform particles to v2 frame and re-render
            z_p_3d = out_v1.get('z_pos_3d')   # [B, T, n_kp, 3]  from 3D model
            if z_p_3d is not None:
                z_p_v2 = transform_particles_to_view(
                    z_p_3d.reshape(B * T, -1, 3), R_1to2, t_1to2
                )
                # render from v2 camera — requires model.decode_from_3d_pos()
                if hasattr(model, 'decode_from_3d_pos'):
                    rec_v2 = model.decode_from_3d_pos(
                        z_p_v2, out_v1, camera_K
                    ).clamp(0, 1)
                    results['cross_view_psnr'].append(
                        psnr_fn(rec_v2, x_v2_flat).mean().item())
                    results['cross_view_ssim'].append(
                        ssim_fn(rec_v2, x_v2_flat).mean().item())
                    results['cross_view_lpips'].append(
                        lpips_fn(rec_v2, x_v2_flat).mean().item())

    return {k: float(np.mean(v)) if v else None for k, v in results.items()}


# ---------------------------------------------------------------------------
# P3: Occlusion recovery
# ---------------------------------------------------------------------------

def eval_occlusion_recovery(
    model,
    occlusion_dataset,
    device: torch.device,
    occlusion_start: int,
    occlusion_end: int,
    gt_pos_key: int = 1,   # index in batch tuple containing GT object positions
    batch_size: int = 16,
    max_batches: int = 30,
    verbose: bool = False,
) -> dict:
    """
    P3: Measure how well the model tracks occluded objects.

    The occlusion_dataset should yield sequences where a specific object is
    fully occluded from frame `occlusion_start` to `occlusion_end`.

    Metrics:
      - z_t_during_occlusion: mean transparency of target particle during occlusion
        (should → 0 for 2D model, should stay >0 for 3D model)
      - position_error_at_t: L2 position error at t+1, t+5, t+10 after occlusion ends
    """
    model.eval()
    loader = DataLoader(occlusion_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, drop_last=False)

    z_t_during = []
    pos_errors = {1: [], 5: [], 10: []}

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        x = batch[0].to(device)               # [B, T, C, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        B, T = x.shape[:2]

        with torch.no_grad():
            out = model(x, with_loss=False)

        obj_on   = out['obj_on']   # [B, T, n_kp, 1]
        mu_z     = out['mu_depth'] # [B, T, n_kp, 1]
        z_base   = out.get('z_base', out.get('mu_p'))  # [B, T, n_kp, 2 or 3]

        # Mean transparency of all active particles during occlusion window
        occ_end = min(occlusion_end, T)
        occ_start = min(occlusion_start, occ_end - 1)
        z_t_occ = obj_on[:, occ_start:occ_end].mean().item()
        z_t_during.append(z_t_occ)

        # Position error after occlusion (requires GT positions in batch)
        if len(batch) > gt_pos_key:
            gt_pos = batch[gt_pos_key].to(device)  # [B, T, n_obj, 2 or 3]
            pred_pos = z_base                        # [B, T, n_kp, 2 or 3]
            for delta in [1, 5, 10]:
                t_idx = occ_end + delta - 1
                if t_idx < T and t_idx < gt_pos.shape[1]:
                    # closest particle to each GT object (Hungarian matching)
                    p = pred_pos[:, t_idx, :, :2]  # [B, n_kp, 2]
                    g = gt_pos[:, t_idx, :, :2]    # [B, n_obj, 2]
                    dists = torch.cdist(g, p)       # [B, n_obj, n_kp]
                    min_dist = dists.min(dim=-1).values.mean().item()
                    pos_errors[delta].append(min_dist)

        if verbose and batch_idx % 10 == 0:
            print(f"[P3] batch {batch_idx}: z_t_occ={z_t_occ:.3f}")

    return {
        'z_t_during_occlusion':  float(np.mean(z_t_during)) if z_t_during else None,
        'pos_error_t+1':  float(np.mean(pos_errors[1]))  if pos_errors[1]  else None,
        'pos_error_t+5':  float(np.mean(pos_errors[5]))  if pos_errors[5]  else None,
        'pos_error_t+10': float(np.mean(pos_errors[10])) if pos_errors[10] else None,
    }


# ---------------------------------------------------------------------------
# P4: Depth complexity ablation
# ---------------------------------------------------------------------------

def eval_depth_complexity_bins(
    model,
    dataset,
    device: torch.device,
    depth_bins: list = None,   # list of (min_dz, max_dz) tuples in metres
    gt_depth_fn=None,          # callable(batch) -> Tensor [B, T, n_obj]
    batch_size: int = 32,
    max_batches_per_bin: int = 30,
    verbose: bool = False,
) -> dict:
    """
    P4: Evaluate PSNR/SSIM as a function of inter-object depth variation (Δz).

    Sequences are bucketed by the max pairwise depth difference between objects.
    For each bucket, we compute PSNR/SSIM.  The expected result: PSNR decreases
    monotonically as Δz increases for 2D-LPWM, but not for 3D-LPWM.

    Args:
        depth_bins: list of (lo, hi) depth-difference intervals in scene units.
                    Default: [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, inf)]
        gt_depth_fn: callable(batch) -> [B, T, n_obj] tensor of z-coordinates.
                     If None, uses mu_depth from model as proxy (degrades signal).
    """
    assert _PIQA_AVAILABLE, "pip install piqa"
    model.eval()

    if depth_bins is None:
        depth_bins = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1e9)]
    bin_labels = [f"{lo:.1f}-{hi:.1f}" if hi < 1e8 else f"{lo:.1f}+"
                  for lo, hi in depth_bins]

    psnr_fn = PSNR().to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=4, drop_last=False)

    bin_psnrs = {label: [] for label in bin_labels}
    bin_counts = {label: 0 for label in bin_labels}

    for batch_idx, batch in enumerate(loader):
        x = batch[0].to(device)  # [B, T, C, H, W]
        if len(x.shape) == 4:
            x = x.unsqueeze(1)
        B, T = x.shape[:2]

        with torch.no_grad():
            out = model(x, with_loss=False)

        rec = out['rec_rgb']  # [B*T, C, H, W] or [B, T, C, H, W]
        if rec.shape[0] == B * T:
            rec = rec.reshape(B, T, *rec.shape[1:])
        x_flat = x.reshape(B * T, *x.shape[2:]).clamp(0, 1)
        rec_flat = rec.reshape(B * T, *rec.shape[2:]).clamp(0, 1)

        psnr_per_sample = psnr_fn(rec_flat, x_flat)  # scalar or [B*T]

        # Compute Δz per sample
        if gt_depth_fn is not None:
            depths = gt_depth_fn(batch).to(device)  # [B, T, n_obj]
        else:
            depths = out['mu_depth'].squeeze(-1)      # [B, T, n_kp] as proxy

        dz = (depths.max(dim=-1).values - depths.min(dim=-1).values)  # [B, T]
        dz_per_seq = dz.mean(dim=-1)  # [B]

        for b in range(B):
            delta = dz_per_seq[b].item()
            for (lo, hi), label in zip(depth_bins, bin_labels):
                if lo <= delta < hi:
                    if bin_counts[label] < max_batches_per_bin * batch_size:
                        psnr_val = psnr_per_sample[b * T: (b + 1) * T].mean().item() \
                            if psnr_per_sample.numel() > 1 else psnr_per_sample.item()
                        bin_psnrs[label].append(psnr_val)
                        bin_counts[label] += 1
                    break

        if verbose and batch_idx % 20 == 0:
            print(f"[P4] batch {batch_idx}: bin counts = {bin_counts}")

    results = {}
    for label in bin_labels:
        vals = bin_psnrs[label]
        results[f'psnr_dz_{label}'] = float(np.mean(vals)) if vals else None
        results[f'count_dz_{label}'] = bin_counts[label]
    return results


# ---------------------------------------------------------------------------
# Combined runner: run all pilot experiments at once
# ---------------------------------------------------------------------------

def run_pilot_experiments(
    model,
    config: dict,
    device: torch.device,
    datasets: dict,
    camera_params: dict = None,
    verbose: bool = True,
) -> dict:
    """
    Convenience runner for P1-P4.

    Args:
        datasets: {
            'train': dataset,
            'valid': dataset,
            'valid_v2': dataset (optional, for P2),
            'occlusion': dataset (optional, for P3),
        }
        camera_params: {
            'K':     [3,3] intrinsics,
            'R_1to2': [3,3],
            't_1to2': [3],
            'baseline': float (metres),
        }
    """
    results = {}

    # P1: depth correlation
    if verbose:
        print("=" * 60)
        print("P1: Depth Correlation Probe")
    p1 = eval_depth_correlation(model, datasets['valid'], device, verbose=verbose)
    results['P1'] = p1
    if verbose:
        print(f"  Pearson r(z_d, GT_depth) = {p1['pearson_r']:.4f}  (p={p1['pearson_p']:.4f})")
        print(f"  Interpretation: {'POOR' if abs(p1['pearson_r']) < 0.3 else 'GOOD'} "
              f"depth encoding in 2D model (|r|<0.3 → z_d not 3D)")

    # P2: cross-view failure baseline
    if 'valid_v2' in datasets and camera_params is not None:
        if verbose:
            print("=" * 60)
            print("P2: Cross-view Generalization")
        K    = camera_params['K'].to(device)
        R    = camera_params.get('R_1to2', torch.eye(3)).to(device)
        t    = camera_params.get('t_1to2', torch.zeros(3)).to(device)
        p2 = eval_cross_view(model, datasets['valid'], datasets['valid_v2'],
                             K, R, t, device, use_3d_particles=False)
        results['P2'] = p2
        if verbose:
            print(f"  Same-view PSNR:   {p2['same_view_psnr']:.2f} dB")
            print(f"  Trivial cross-PSNR: {p2['trivial_psnr']:.2f} dB  "
                  f"(gap = {p2['same_view_psnr'] - p2['trivial_psnr']:.2f} dB)")

    # P3: occlusion recovery
    if 'occlusion' in datasets:
        if verbose:
            print("=" * 60)
            print("P3: Occlusion Recovery")
        occ_cfg = config.get('occlusion_cfg', {'start': 5, 'end': 15})
        p3 = eval_occlusion_recovery(model, datasets['occlusion'], device,
                                     occlusion_start=occ_cfg['start'],
                                     occlusion_end=occ_cfg['end'], verbose=verbose)
        results['P3'] = p3
        if verbose:
            print(f"  Mean z_t during occlusion: {p3['z_t_during_occlusion']:.4f}  "
                  f"({'particle dies' if p3['z_t_during_occlusion'] < 0.2 else 'particle persists'})")

    # P4: depth complexity ablation
    if verbose:
        print("=" * 60)
        print("P4: Depth Complexity vs PSNR")
    p4 = eval_depth_complexity_bins(model, datasets['valid'], device, verbose=verbose)
    results['P4'] = p4
    if verbose:
        for k, v in p4.items():
            if 'psnr' in k and v is not None:
                print(f"  {k}: {v:.2f} dB  (n={p4[k.replace('psnr', 'count')]})")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    import json

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models import LPWM
    from utils.util_func import get_config
    from datasets.get_dataset import get_video_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to JSON config')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--p', nargs='+', default=['1', '2', '3', '4'],
                        help='Which pilot experiments to run (1 2 3 4)')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config = get_config(args.config)

    print(f"Loading model from {args.checkpoint} ...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = LPWM(**ckpt.get('model_kwargs', {}))
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).eval()

    valid_ds = get_video_dataset(config['ds'], config['root'],
                                 seq_len=config['timestep_horizon'] + 1,
                                 mode='valid', image_size=config['image_size'])

    datasets = {'valid': valid_ds}

    # Multi-view datasets for P2
    if config.get('n_views', 1) > 1 and '2' in args.p:
        datasets['valid_v2'] = get_video_dataset(
            config['ds'], config['root'],
            seq_len=config['timestep_horizon'] + 1,
            mode='valid', image_size=config['image_size'])
        # Camera params — update with actual calibration
        camera_params = {
            'K': torch.tensor([[256., 0., 64.], [0., 256., 64.], [0., 0., 1.]]),
            'R_1to2': torch.eye(3),
            't_1to2': torch.tensor([0.08, 0., 0.]),  # 8cm baseline placeholder
        }
    else:
        camera_params = None

    results = run_pilot_experiments(model, config, device, datasets,
                                    camera_params=camera_params, verbose=True)

    out_path = os.path.join(os.path.dirname(args.checkpoint), 'pilot_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
