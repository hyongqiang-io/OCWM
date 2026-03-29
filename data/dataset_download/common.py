from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import requests
from requests import Response


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "dataset"
CHUNK_SIZE = 1024 * 1024


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_download_url(url: str) -> str:
    if "sharepoint.com" in url and "download=1" not in url:
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}download=1"
    return url


def infer_filename_from_url(url: str, fallback: str = "downloaded") -> str:
    parsed = urlparse(url)
    parsed_name = Path(parsed.path).name
    if parsed_name:
        return parsed_name

    query = parse_qs(parsed.query)
    for key in ("filename", "file", "id"):
        values = query.get(key)
        if values:
            return values[0]
    return fallback


def _is_valid_archive(path: Path) -> bool:
    return zipfile.is_zipfile(path) or tarfile.is_tarfile(path)


def _raise_for_html_response(response: Response, url: str) -> None:
    content_type = (response.headers.get("content-type") or "").lower()
    if "text/html" not in content_type:
        return

    preview = response.text[:2000]
    if "The link was set to expire" in preview:
        raise RuntimeError(f"Download link expired or requires refresh: {url}")
    raise RuntimeError(f"Download URL returned an HTML page instead of an archive: {url}")


def download_file(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    normalized_url = normalize_download_url(url)
    existing_size = destination.stat().st_size if destination.exists() else 0
    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"

    print(f"[download] {normalized_url}")
    print(f"[save] {destination}")
    if existing_size > 0:
        print(f"[resume] from byte {existing_size}")

    with requests.get(
        normalized_url,
        headers=headers,
        stream=True,
        timeout=(30, 300),
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        _raise_for_html_response(response, normalized_url)

        append_mode = existing_size > 0 and response.status_code == 206
        if existing_size > 0 and response.status_code == 200:
            existing_size = 0
        mode = "ab" if append_mode else "wb"

        with destination.open(mode) as output_file:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    output_file.write(chunk)

    return destination


def download_from_google_drive(url: str, destination: Path, folder: bool = False) -> Path:
    target_dir = destination if folder else destination.parent
    ensure_dir(target_dir)
    command = [sys.executable, "-m", "gdown", "--fuzzy", "--continue"]
    if folder:
        command.append("--folder")
    command.extend([url, "-O", str(destination)])
    print(f"[gdown] {' '.join(command)}")
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Google Drive download failed. Install gdown and verify the shared link is still accessible."
        )
    return destination


def download_resource(
    url: str,
    destination: Path,
    use_gdown: bool = False,
    folder: bool = False,
) -> Path:
    if use_gdown or "drive.google.com" in url:
        return download_from_google_drive(url, destination, folder=folder)
    return download_file(url, destination)


def extract_archive(archive_path: Path, destination: Path) -> Path:
    ensure_dir(destination)
    print(f"[extract] {archive_path} -> {destination}")

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zip_file:
            zip_file.extractall(destination)
        return destination

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tar_file:
            tar_file.extractall(destination)
        return destination

    raise ValueError(f"Unsupported archive format: {archive_path}")


def download_and_prepare(
    url: str,
    root: Path,
    extract: bool = True,
    remove_archive: bool = False,
    output_name: str | None = None,
    use_gdown: bool = False,
) -> Path:
    ensure_dir(root)
    archive_name = output_name or infer_filename_from_url(url)
    archive_path = root / archive_name
    download_resource(url, archive_path, use_gdown=use_gdown)

    if extract:
        if not _is_valid_archive(archive_path):
            raise RuntimeError(
                f"Downloaded file is not a complete archive yet: {archive_path}. Re-run the downloader to resume."
            )
        extract_archive(archive_path, root)

    if remove_archive and archive_path.exists():
        archive_path.unlink()

    return archive_path


def stage_local_archive(
    archive_path: Path,
    root: Path,
    extract: bool = True,
    copy_archive: bool = False,
) -> Path:
    ensure_dir(root)
    target_path = root / archive_path.name

    if copy_archive:
        print(f"[copy] {archive_path} -> {target_path}")
        shutil.copy2(archive_path, target_path)
    else:
        target_path = archive_path

    if extract:
        extract_archive(target_path, root)

    return target_path
