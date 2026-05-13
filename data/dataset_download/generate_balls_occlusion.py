"""
Generate Balls Occlusion dataset in HDF5 format compatible with LPWM's balls_ds.py.

Simulates 2D elastic collisions of colored balls with frequent occlusion events.
Balls have varying sizes and can fully overlap (occlude) each other.

Output format (per split):
  {split}.hdf5:
    imgs:       [N, T, H, W, 3] uint8
    positions:  [N, T, K, 2]    float32  (normalized [0,1] ball centers)
    sizes:      [N, T, K]       float32  (ball radii in pixels)
    ids:        [N, T, K]       int32    (unique ball ID, persistent across time)
    in_camera:  [N, T, K]       int32    (1 if ball center is in frame, else 0)

Usage:
    python generate_balls_occlusion.py --output_dir /path/to/balls_occlusion
    python generate_balls_occlusion.py --output_dir /path/to/balls_occlusion --n_train 5000 --n_val 500 --n_test 500
"""

import argparse
import os
import numpy as np
import h5py
from pathlib import Path


def draw_circle(canvas, cx, cy, radius, color):
    """Draw a filled circle on canvas using anti-aliased rendering."""
    H, W = canvas.shape[:2]
    y_min = max(0, int(cy - radius - 1))
    y_max = min(H, int(cy + radius + 2))
    x_min = max(0, int(cx - radius - 1))
    x_max = min(W, int(cx + radius + 2))

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            if dist <= radius:
                alpha = min(1.0, radius - dist + 0.5)
                canvas[y, x] = (
                    alpha * np.array(color) + (1 - alpha) * canvas[y, x]
                ).astype(np.uint8)


def draw_circle_vectorized(canvas, cx, cy, radius, color):
    """Vectorized circle drawing for speed."""
    H, W = canvas.shape[:2]
    y_min = max(0, int(cy - radius - 1))
    y_max = min(H, int(cy + radius + 2))
    x_min = max(0, int(cx - radius - 1))
    x_max = min(W, int(cx + radius + 2))

    if y_min >= y_max or x_min >= x_max:
        return

    ys = np.arange(y_min, y_max)
    xs = np.arange(x_min, x_max)
    yy, xx = np.meshgrid(ys, xs, indexing='ij')

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    mask = dist <= radius
    alpha = np.clip(radius - dist + 0.5, 0.0, 1.0)

    color_arr = np.array(color, dtype=np.float32)
    for c in range(3):
        patch = canvas[y_min:y_max, x_min:x_max, c].astype(np.float32)
        patch = alpha * color_arr[c] + (1 - alpha) * patch
        canvas[y_min:y_max, x_min:x_max, c] = np.where(
            mask, patch.astype(np.uint8), canvas[y_min:y_max, x_min:x_max, c]
        )


COLORS = [
    (255, 0, 0),      # red
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
]

BG_COLOR = (32, 32, 32)


def simulate_episode(
    n_balls, ep_len, img_size=64, min_radius=4, max_radius=8,
    speed_range=(1.5, 4.0), occlusion_bias=True, rng=None
):
    """
    Simulate one episode of bouncing balls with occlusion.

    Args:
        n_balls: number of balls
        ep_len: number of frames
        img_size: image resolution (square)
        min_radius, max_radius: ball size range
        speed_range: initial speed magnitude range
        occlusion_bias: if True, bias initial positions to encourage occlusions
        rng: numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng()

    H = W = img_size

    # Initialize ball properties
    radii = rng.uniform(min_radius, max_radius, size=n_balls)
    colors = [COLORS[i % len(COLORS)] for i in range(n_balls)]

    # Initialize positions — bias toward center to encourage occlusions
    if occlusion_bias:
        margin = max_radius + 2
        center = img_size / 2
        spread = img_size * 0.3
        positions = rng.normal(center, spread, size=(n_balls, 2))
        positions = np.clip(positions, margin, img_size - margin)
    else:
        margin = max_radius + 2
        positions = rng.uniform(margin, img_size - margin, size=(n_balls, 2))

    # Initialize velocities
    speed_mag = rng.uniform(speed_range[0], speed_range[1], size=n_balls)
    angles = rng.uniform(0, 2 * np.pi, size=n_balls)
    velocities = np.stack([speed_mag * np.cos(angles), speed_mag * np.sin(angles)], axis=1)

    # Storage
    all_imgs = np.zeros((ep_len, H, W, 3), dtype=np.uint8)
    all_positions = np.zeros((ep_len, n_balls, 2), dtype=np.float32)
    all_sizes = np.zeros((ep_len, n_balls), dtype=np.float32)
    all_ids = np.zeros((ep_len, n_balls), dtype=np.int32)
    all_in_camera = np.ones((ep_len, n_balls), dtype=np.int32)

    dt = 1.0
    restitution = 0.98

    for t in range(ep_len):
        # Record state
        all_positions[t] = positions / img_size  # normalize to [0,1]
        all_sizes[t] = radii
        all_ids[t] = np.arange(n_balls)
        all_in_camera[t] = 1  # all balls always in frame for this dataset

        # Render frame (back-to-front by ball index — lower index drawn first, higher on top)
        canvas = np.full((H, W, 3), BG_COLOR, dtype=np.uint8)
        for i in range(n_balls):
            cx, cy = positions[i]
            draw_circle_vectorized(canvas, cx, cy, radii[i], colors[i])
        all_imgs[t] = canvas

        # Physics step
        positions = positions + velocities * dt

        # Wall collisions (elastic bounce)
        for i in range(n_balls):
            r = radii[i]
            for axis in range(2):
                if positions[i, axis] - r < 0:
                    positions[i, axis] = r
                    velocities[i, axis] = -velocities[i, axis] * restitution
                elif positions[i, axis] + r > img_size:
                    positions[i, axis] = img_size - r
                    velocities[i, axis] = -velocities[i, axis] * restitution

        # Ball-ball elastic collisions
        for i in range(n_balls):
            for j in range(i + 1, n_balls):
                dx = positions[j] - positions[i]
                dist = np.linalg.norm(dx)
                min_dist = radii[i] + radii[j]

                if dist < min_dist and dist > 1e-6:
                    # Normal vector
                    n = dx / dist
                    # Relative velocity
                    dv = velocities[i] - velocities[j]
                    # Relative velocity along normal
                    dvn = np.dot(dv, n)

                    if dvn > 0:  # approaching
                        # Mass proportional to radius^2
                        m_i = radii[i] ** 2
                        m_j = radii[j] ** 2
                        # Impulse
                        J = (2 * dvn) / (m_i + m_j)
                        velocities[i] -= J * m_j * n * restitution
                        velocities[j] += J * m_i * n * restitution

                    # Separate overlapping balls
                    overlap = min_dist - dist
                    separation = n * (overlap / 2 + 0.5)
                    positions[i] -= separation
                    positions[j] += separation

    return all_imgs, all_positions, all_sizes, all_ids, all_in_camera


def generate_split(
    output_path, n_episodes, ep_len, n_balls_range=(3, 6),
    img_size=64, seed=0, occlusion_bias=True
):
    """Generate one split (train/val/test) as HDF5."""
    rng = np.random.default_rng(seed)

    # Determine max balls for uniform array shape
    max_balls = n_balls_range[1]

    with h5py.File(output_path, 'w') as f:
        imgs_ds = f.create_dataset(
            'imgs', shape=(n_episodes, ep_len, img_size, img_size, 3),
            dtype=np.uint8, chunks=(1, ep_len, img_size, img_size, 3),
            compression='gzip', compression_opts=4
        )
        pos_ds = f.create_dataset(
            'positions', shape=(n_episodes, ep_len, max_balls, 2),
            dtype=np.float32
        )
        size_ds = f.create_dataset(
            'sizes', shape=(n_episodes, ep_len, max_balls),
            dtype=np.float32
        )
        ids_ds = f.create_dataset(
            'ids', shape=(n_episodes, ep_len, max_balls),
            dtype=np.int32
        )
        in_cam_ds = f.create_dataset(
            'in_camera', shape=(n_episodes, ep_len, max_balls),
            dtype=np.int32
        )

        for ep in range(n_episodes):
            n_balls = rng.integers(n_balls_range[0], n_balls_range[1] + 1)

            imgs, pos, sizes, ids, in_cam = simulate_episode(
                n_balls=n_balls,
                ep_len=ep_len,
                img_size=img_size,
                occlusion_bias=occlusion_bias,
                rng=rng,
            )

            imgs_ds[ep] = imgs

            # Pad to max_balls (unused slots get 0 position, 0 size, -1 id, 0 in_camera)
            pos_ds[ep, :, :n_balls] = pos
            size_ds[ep, :, :n_balls] = sizes
            ids_ds[ep, :, :n_balls] = ids
            in_cam_ds[ep, :, :n_balls] = in_cam

            # Mark unused slots
            if n_balls < max_balls:
                ids_ds[ep, :, n_balls:] = -1
                in_cam_ds[ep, :, n_balls:] = 0

            if (ep + 1) % 100 == 0 or ep == n_episodes - 1:
                print(f'  [{ep + 1}/{n_episodes}] generated')

    print(f'  Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)')


def main():
    parser = argparse.ArgumentParser(description='Generate Balls Occlusion dataset')
    parser.add_argument('--output_dir', type=str,
                        default='/home/hyq/OCWM/data/dataset/balls_occlusion')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_val', type=int, default=500)
    parser.add_argument('--n_test', type=int, default=500)
    parser.add_argument('--ep_len', type=int, default=100)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--min_balls', type=int, default=3)
    parser.add_argument('--max_balls', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    n_balls_range = (args.min_balls, args.max_balls)

    print(f'Generating Balls Occlusion dataset:')
    print(f'  Output: {args.output_dir}')
    print(f'  Image size: {args.img_size}x{args.img_size}')
    print(f'  Episode length: {args.ep_len}')
    print(f'  Balls per episode: {args.min_balls}-{args.max_balls}')
    print(f'  Episodes: train={args.n_train}, val={args.n_val}, test={args.n_test}')
    print()

    splits = [
        ('train', args.n_train, args.seed),
        ('val', args.n_val, args.seed + 10000),
        ('test', args.n_test, args.seed + 20000),
    ]

    for split_name, n_eps, seed in splits:
        print(f'Generating {split_name} split ({n_eps} episodes)...')
        output_path = os.path.join(args.output_dir, f'{split_name}.hdf5')
        generate_split(
            output_path=output_path,
            n_episodes=n_eps,
            ep_len=args.ep_len,
            n_balls_range=n_balls_range,
            img_size=args.img_size,
            seed=seed,
            occlusion_bias=True,
        )
        print()

    print('Done! Dataset ready at:', args.output_dir)
    print()
    print('Verify with:')
    print(f'  python -c "import h5py; f=h5py.File(\'{args.output_dir}/train.hdf5\',\'r\'); print({{k: v.shape for k,v in f.items()}})"')


if __name__ == '__main__':
    main()
