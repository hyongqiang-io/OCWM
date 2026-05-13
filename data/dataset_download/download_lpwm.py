from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import DEFAULT_DATA_ROOT, ensure_dir, extract_archive, stage_local_archive


HF_DATASET_REPOS = {
    "sketchy": "taldatech/sketchy_128",
    "bair": "taldatech/bair_256",
    "bridge": "taldatech/bridge_256",
    "panda": "taldatech/panda_ds",
    "ogbench": "taldatech/ogbench_data",
    "mario": "taldatech/mario_gameplay",
    "obj3d128": "taldatech/OBJ3D",
}

HF_PRIMARY_ARCHIVES = {
    "sketchy": "sketchy_data.zip",
    "bair": "bair_256_ours.tar.gz",
    "bridge": "bridge_ds.tar.gz",
    "panda": "panda_ds.tar.gz",
    "ogbench": "ogbench_ds.tar.gz",
    "mario": "smb_ep.zip",
    "obj3d128": "OBJ3D.zip",
}

GOOGLE_DRIVE_FILES = {
    "obj3d128": "https://drive.google.com/file/d/1XSLW3qBtcxxvV-5oiRruVTlDlQ_Yatzm/view",
}

MEGA_FILES = {
    "phyre": "https://mega.nz/file/lIkDnZZD#Ym4vlAqd3egCljoujya33KHaua7AwmusmbUw27OdIHE",
    "balls": "https://mega.nz/file/4cUR1b5a#RwFFzCiESeeQb8rYgt7PK2_D8b_69-K85RV3jlaphTo",
}

ALIAS_TO_CANONICAL = {
    "langtable_action": "langtable",
    "obj3d128_img": "obj3d128",
    "sketchy_action": "sketchy",
}

HF_DEFAULT_ALLOW_PATTERNS = {
    "sketchy": ["*.png", "*.pt", "*.json", "*.jsonl", "*.txt", "*.md"],
    "bair": ["*.png", "*.pkl", "*.json", "*.jsonl", "*.txt", "*.md"],
    "bridge": ["*.png", "*.pt", "*.json", "*.jsonl", "*.txt", "*.md"],
    "panda": ["*.png", "*.pt", "*.json", "*.jsonl", "*.txt", "*.md"],
    "ogbench": ["*.png", "*.pt", "*.json", "*.jsonl", "*.txt", "*.md"],
    "mario": ["*.png", "*.json", "*.jsonl", "*.txt", "*.md"],
    "obj3d128": ["*.png", "*.json", "*.jsonl", "*.txt", "*.md"],
}

AUTO_MODE_HELP = {
    "sketchy": "官方 HF 预处理版本，可直接自动下载。",
    "bair": "官方 HF 预处理版本，可直接自动下载。",
    "bair64": "上游配置包含该数据集，但 LPWM README 未提供单独公开来源；通常需要你自行准备 64x64 版本。",
    "balls_occlusion": "与 Balls 类似，通常需要你自行准备对应 occlusion 版本数据。",
    "bridge": "官方 HF 预处理版本，可直接自动下载。",
    "panda": "官方 HF 预处理版本，可直接自动下载。",
    "ogbench": "官方 HF 预处理版本，可直接自动下载。",
    "mario": "官方 HF 预处理版本，可直接自动下载。",
    "obj3d128": "优先走 HF；也支持手动提供 Google Drive 压缩包。",
    "langtable": "官方只提供预处理脚本，需要基于 Open-X / TFDS 自行准备。",
    "phyre": "官方给的是 MEGA 链接。脚本会提示你手动提供压缩包，或先安装 MEGA 下载工具后自行下载。",
    "balls": "官方给的是 MEGA 链接。脚本会提示你手动提供压缩包，或先安装 MEGA 下载工具后自行下载。",
    "shapes": "无需下载，训练时在线生成。",
    "traffic": "仓库里有 loader 和预处理逻辑，但 LPWM README 未提供公开下载源。",
}


def canonical_dataset_name(name: str) -> str:
    return ALIAS_TO_CANONICAL.get(name, name)


def dataset_root(root: Path, dataset: str) -> Path:
    return root / canonical_dataset_name(dataset)


def build_manifest(root: Path) -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for dataset in sorted(
        set(HF_DATASET_REPOS) | set(MEGA_FILES) | {"bair64", "balls_occlusion", "langtable", "shapes", "traffic"}
    ):
        canonical = canonical_dataset_name(dataset)
        target_root = dataset_root(root, canonical)
        source_kind = "manual"
        source_ref = ""
        if canonical in HF_DATASET_REPOS:
            source_kind = "huggingface"
            source_ref = HF_DATASET_REPOS[canonical]
        elif canonical in MEGA_FILES:
            source_kind = "mega"
            source_ref = MEGA_FILES[canonical]

        manifest[canonical] = {
            "dataset": canonical,
            "target_root": str(target_root),
            "source_kind": source_kind,
            "source_ref": source_ref,
            "notes": AUTO_MODE_HELP[canonical],
        }
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or prepare LPWM datasets under data/dataset.")
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=sorted(
            {
                "bair",
                "bair64",
                "balls",
                "balls_occlusion",
                "bridge",
                "langtable",
                "langtable_action",
                "mario",
                "obj3d128",
                "obj3d128_img",
                "ogbench",
                "panda",
                "phyre",
                "shapes",
                "sketchy",
                "sketchy_action",
                "traffic",
            }
        ),
        help="LPWM dataset name. Repeat to process multiple datasets.",
    )
    parser.add_argument("--all-auto", action="store_true", help="Process every dataset that can be auto-downloaded.")
    parser.add_argument("--list", action="store_true", help="Print the LPWM dataset manifest and exit.")
    parser.add_argument(
        "--hf-allow-pattern",
        action="append",
        default=[],
        help="Extra huggingface_hub allow pattern. Repeatable.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional Hugging Face endpoint override, for example https://hf-mirror.com .",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=None,
        help="Use a local archive instead of remote download. Valid for datasets that need manual staging.",
    )
    parser.add_argument(
        "--copy-archive",
        action="store_true",
        help="When using --archive, copy it into the dataset root before extracting.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only place archives or snapshots, do not extract local archives.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove the existing target dataset directory before downloading or extracting.",
    )
    return parser.parse_args()


def resolve_requested_datasets(args: argparse.Namespace) -> list[str]:
    if args.list:
        return []

    if args.all_auto:
        datasets = sorted(HF_DATASET_REPOS)
    elif args.dataset:
        datasets = [canonical_dataset_name(name) for name in args.dataset]
    else:
        raise SystemExit("Specify --dataset ... or use --all-auto.")

    deduped: list[str] = []
    for name in datasets:
        if name not in deduped:
            deduped.append(name)
    return deduped


def print_manifest(root: Path) -> None:
    print(json.dumps(build_manifest(root), indent=2, ensure_ascii=False))


def maybe_reset_target(target_root: Path, force: bool) -> None:
    if force and target_root.exists():
        shutil.rmtree(target_root)


def download_hf_dataset(dataset: str, target_root: Path, hf_endpoint: str | None, extract: bool) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Automatic LPWM HF downloads require huggingface_hub."
        ) from exc

    repo_id = HF_DATASET_REPOS[dataset]
    ensure_dir(target_root)
    archive_name = HF_PRIMARY_ARCHIVES.get(dataset)
    if archive_name is None:
        raise SystemExit(f"No primary HF archive configured for dataset: {dataset}")
    archive_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=archive_name,
            repo_type="dataset",
            local_dir=str(target_root),
            local_dir_use_symlinks=False,
            resume_download=True,
            endpoint=hf_endpoint,
        )
    )
    if extract:
        extract_archive(archive_path, target_root)


def stage_manual_archive(dataset: str, target_root: Path, archive_path: Path, extract: bool, copy_archive: bool) -> None:
    if not archive_path.exists():
        raise SystemExit(f"Archive does not exist: {archive_path}")
    ensure_dir(target_root)
    stage_local_archive(
        archive_path=archive_path,
        root=target_root,
        extract=extract,
        copy_archive=copy_archive,
    )
    print(f"dataset_root={target_root}")
    print(f"dataset={dataset}")


def explain_manual_requirement(dataset: str, target_root: Path) -> None:
    if dataset == "shapes":
        print(
            f"{dataset}: no download needed. LPWM generates this dataset on the fly; target_root is not used. "
            "You can train directly with --dataset shapes."
        )
        return

    if dataset == "langtable":
        print(
            f"{dataset}: automatic download is not implemented because upstream requires preprocessing with "
            "`baseline/lpwm/upstream/datasets/langtable_preparation.py` against Open-X TFDS data."
        )
        print(f"target_root={target_root}")
        return

    if dataset == "bair64":
        print(
            f"{dataset}: LPWM upstream includes a config and loader, but the README does not publish a separate download source."
        )
        print(f"target_root={target_root}")
        print("Prepare a BAIR 64x64 directory tree manually, or derive it from your BAIR dataset and pass --data-root during training.")
        return

    if dataset == "balls_occlusion":
        print(
            f"{dataset}: upstream expects a separate Balls occlusion dataset root, but only a generic Balls MEGA link is documented."
        )
        print(f"target_root={target_root}")
        print("Prepare the occlusion variant manually, then re-run with --archive if you have a packaged copy.")
        return

    if dataset == "traffic":
        print(
            f"{dataset}: no stable public source is referenced in the LPWM README. "
            "Prepare the `.npy` file expected by `baseline/lpwm/upstream/datasets/traffic_ds.py` manually."
        )
        print(f"target_root={target_root}")
        return

    if dataset in MEGA_FILES:
        print(
            f"{dataset}: upstream only provides a MEGA link, so this script cannot fetch it directly without extra tooling."
        )
        print(f"source={MEGA_FILES[dataset]}")
        print(f"target_root={target_root}")
        print("Use one of the following approaches:")
        print("1. Download the archive manually, then re-run this command with --archive /path/to/file.")
        print("2. Install a MEGA CLI tool yourself and place the extracted dataset under the target_root shown above.")
        return

    raise SystemExit(f"No download strategy registered for dataset: {dataset}")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if args.list:
        print_manifest(root)
        return

    datasets = resolve_requested_datasets(args)
    for dataset in datasets:
        target_root = dataset_root(root, dataset)
        maybe_reset_target(target_root, args.force)

        if dataset in HF_DATASET_REPOS:
            if args.archive is not None:
                stage_manual_archive(
                    dataset=dataset,
                    target_root=target_root,
                    archive_path=args.archive.resolve(),
                    extract=not args.no_extract,
                    copy_archive=args.copy_archive,
                )
                continue

            download_hf_dataset(
                dataset=dataset,
                target_root=target_root,
                hf_endpoint=args.hf_endpoint,
                extract=not args.no_extract,
            )
            print(f"dataset_root={target_root}")
            print(f"dataset={dataset}")
            continue

        if args.archive is not None:
            stage_manual_archive(
                dataset=dataset,
                target_root=target_root,
                archive_path=args.archive.resolve(),
                extract=not args.no_extract,
                copy_archive=args.copy_archive,
            )
            continue

        explain_manual_requirement(dataset=dataset, target_root=target_root)


if __name__ == "__main__":
    main()
