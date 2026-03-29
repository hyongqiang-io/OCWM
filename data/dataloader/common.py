from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.transforms.functional import InterpolationMode, pil_to_tensor, resize


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
SequenceSource = tuple[str, Path, str]


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing_root(root: Path, nested_candidates: Sequence[str]) -> Path:
    root = root.expanduser().resolve()
    if root.exists():
        return root

    for candidate in nested_candidates:
        candidate_path = root / candidate
        if candidate_path.exists():
            return candidate_path

    raise FileNotFoundError(f"Unable to locate dataset root under {root}.")


def resolve_existing_path(candidates: Iterable[Path]) -> Path:
    candidate_paths = list(candidates)
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    rendered = "\n".join(str(candidate) for candidate in candidate_paths)
    raise FileNotFoundError(f"Unable to locate any expected dataset path. Checked:\n{rendered}")


def maybe_resolve_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def list_image_files(root: Path, recursive: bool = False) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def list_video_files(root: Path, recursive: bool = False) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(files)


def directory_contains_files(root: Path, extensions: Sequence[str]) -> bool:
    if not root.exists() or not root.is_dir():
        return False

    normalized_extensions = {extension.lower() for extension in extensions}
    for path in root.iterdir():
        if path.is_file() and path.suffix.lower() in normalized_extensions:
            return True
    return False


def list_sequence_dirs(root: Path, recursive: bool = False) -> list[Path]:
    if not root.exists():
        return []

    sequence_dirs: list[Path] = []
    if directory_contains_files(root, IMAGE_EXTENSIONS):
        sequence_dirs.append(root.resolve())

    iterator = root.rglob("*") if recursive else root.glob("*")
    for path in iterator:
        if path.is_dir() and directory_contains_files(path, IMAGE_EXTENSIONS):
            sequence_dirs.append(path.resolve())

    unique_dirs = {path for path in sequence_dirs}
    return sorted(unique_dirs, key=lambda path: str(path))


def _relative_sequence_name(path: Path, root: Path) -> str:
    if path == root:
        return path.name
    return path.relative_to(root).as_posix()


def discover_video_sources(root: Path, recursive: bool = True) -> list[SequenceSource]:
    frame_dirs = list_sequence_dirs(root, recursive=recursive)
    if frame_dirs:
        return [(_relative_sequence_name(path, root), path, "frames") for path in frame_dirs]

    video_files = list_video_files(root, recursive=recursive)
    return [
        (video_path.relative_to(root).with_suffix("").as_posix(), video_path, "video")
        for video_path in video_files
    ]


def maybe_limit(paths: list[Path], max_items: Optional[int]) -> list[Path]:
    if max_items is None:
        return paths
    return paths[:max_items]


def load_image_tensor(
    image_path: Path,
    image_size: int,
    normalize: bool = False,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> Tensor:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = resize(
            image,
            size=[image_size, image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = pil_to_tensor(image).float() / 255.0

    if normalize:
        mean_tensor = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
        std_tensor = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
        tensor = (tensor - mean_tensor) / std_tensor

    return tensor


def sample_frame_indices(
    num_frames: int,
    frames_per_clip: Optional[int],
    frame_stride: int,
    random_clip: bool,
) -> list[int]:
    if num_frames == 0:
        return []

    stride = max(frame_stride, 1)
    if frames_per_clip is None or frames_per_clip <= 0:
        return list(range(0, num_frames, stride))

    max_offset = max(num_frames - 1 - (frames_per_clip - 1) * stride, 0)
    start_index = random.randint(0, max_offset) if random_clip and max_offset > 0 else 0

    indices = []
    for clip_index in range(frames_per_clip):
        indices.append(min(start_index + clip_index * stride, num_frames - 1))
    return indices


def load_video_clip(
    frame_paths: Sequence[Path],
    image_size: int,
    frames_per_clip: Optional[int],
    frame_stride: int,
    random_clip: bool,
    normalize: bool = False,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> tuple[Tensor, Tensor, list[Path]]:
    sampled_indices = sample_frame_indices(
        num_frames=len(frame_paths),
        frames_per_clip=frames_per_clip,
        frame_stride=frame_stride,
        random_clip=random_clip,
    )
    sampled_paths = [frame_paths[index] for index in sampled_indices]
    clip = torch.stack(
        [
            load_image_tensor(
                image_path=frame_path,
                image_size=image_size,
                normalize=normalize,
                mean=mean,
                std=std,
            )
            for frame_path in sampled_paths
        ],
        dim=0,
    )
    return clip, torch.tensor(sampled_indices, dtype=torch.long), sampled_paths


def load_video_file_clip(
    video_path: Path,
    image_size: int,
    frames_per_clip: Optional[int],
    frame_stride: int,
    random_clip: bool,
    normalize: bool = False,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> tuple[Tensor, Tensor, list[str], int]:
    try:
        frames, _, _ = read_video(str(video_path), pts_unit="sec", output_format="TCHW")
    except Exception as exc:
        raise RuntimeError(
            f"Unable to decode video file {video_path}. Ensure torchvision video decoding is available."
        ) from exc

    if frames.shape[0] == 0:
        raise ValueError(f"No video frames decoded from {video_path}")

    frames = resize(
        frames.float() / 255.0,
        size=[image_size, image_size],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    sampled_indices = sample_frame_indices(
        num_frames=int(frames.shape[0]),
        frames_per_clip=frames_per_clip,
        frame_stride=frame_stride,
        random_clip=random_clip,
    )
    clip = frames[sampled_indices]

    if normalize:
        mean_tensor = torch.tensor(mean, dtype=clip.dtype).view(1, 3, 1, 1)
        std_tensor = torch.tensor(std, dtype=clip.dtype).view(1, 3, 1, 1)
        clip = (clip - mean_tensor) / std_tensor

    sampled_refs = [f"{video_path}#{index:06d}" for index in sampled_indices]
    return clip, torch.tensor(sampled_indices, dtype=torch.long), sampled_refs, int(frames.shape[0])


def load_sequence_source(
    source: SequenceSource,
    image_size: int,
    frames_per_clip: Optional[int],
    frame_stride: int,
    random_clip: bool,
    normalize: bool = False,
    mean: Sequence[float] = DEFAULT_MEAN,
    std: Sequence[float] = DEFAULT_STD,
) -> tuple[Tensor, Tensor, list[str], int]:
    _, source_path, source_type = source
    if source_type == "frames":
        frame_paths = list_image_files(source_path)
        if len(frame_paths) == 0:
            raise ValueError(f"No video frames found under {source_path}")
        clip, frame_indices, sampled_paths = load_video_clip(
            frame_paths=frame_paths,
            image_size=image_size,
            frames_per_clip=frames_per_clip,
            frame_stride=frame_stride,
            random_clip=random_clip,
            normalize=normalize,
            mean=mean,
            std=std,
        )
        return clip, frame_indices, [str(path) for path in sampled_paths], len(frame_paths)
    if source_type == "video":
        return load_video_file_clip(
            video_path=source_path,
            image_size=image_size,
            frames_per_clip=frames_per_clip,
            frame_stride=frame_stride,
            random_clip=random_clip,
            normalize=normalize,
            mean=mean,
            std=std,
        )
    raise ValueError(f"Unsupported sequence source type: {source_type}")


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    drop_last: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
