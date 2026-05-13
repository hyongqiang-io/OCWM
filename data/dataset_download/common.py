from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from requests import Response


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "dataset"
CHUNK_SIZE = 1024 * 1024
HTTP_DOWNLOAD_MAX_RETRIES = 5
GOOGLE_DRIVE_RANGE_SIZE = 64 * CHUNK_SIZE
GOOGLE_DRIVE_MAX_RETRIES = 5
GOOGLE_DRIVE_FILE_ID_PATTERNS = (
    re.compile(r"/file/d/([A-Za-z0-9_-]+)"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]+)"),
)
GOOGLE_DRIVE_CONFIRM_ACTION_PATTERN = re.compile(r'action="([^"]+)"')
GOOGLE_DRIVE_INPUT_PATTERN = re.compile(r'name="([^"]+)" value="([^"]*)"')


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


def extract_google_drive_file_id(url: str) -> str:
    for pattern in GOOGLE_DRIVE_FILE_ID_PATTERNS:
        match = pattern.search(url)
        if match is not None:
            return match.group(1)
    return url


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


def _raise_for_google_drive_html(html: str, url: str) -> None:
    if "Too many users have viewed or downloaded this file recently" in html:
        raise RuntimeError(f"Google Drive download quota exceeded: {url}")
    if "Google Drive can't scan this file for viruses." in html:
        raise RuntimeError(f"Google Drive returned an unconfirmed virus-scan warning page: {url}")
    if "Sign in" in html and "Google Drive" in html:
        raise RuntimeError(f"Google Drive file is not publicly accessible: {url}")
    raise RuntimeError(f"Google Drive returned an HTML page instead of a downloadable file: {url}")


def _parse_content_range_total(content_range: str | None) -> int | None:
    if content_range is None or "/" not in content_range:
        return None
    total = content_range.rsplit("/", 1)[1].strip()
    if not total.isdigit():
        return None
    return int(total)


def _parse_content_length(content_length: str | None) -> int | None:
    if content_length is None:
        return None
    content_length = content_length.strip()
    if not content_length.isdigit():
        return None
    return int(content_length)


def _resolve_response_total_size(response: Response, range_start: int) -> int | None:
    parsed_total = _parse_content_range_total(response.headers.get("content-range"))
    if parsed_total is not None:
        return parsed_total

    content_length = _parse_content_length(response.headers.get("content-length"))
    if content_length is None:
        return None

    if response.status_code == 206:
        return range_start + content_length
    if response.status_code == 200:
        return content_length
    return None


def _probe_remote_file_size(url: str) -> int | None:
    with requests.head(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=(30, 60),
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        return _parse_content_length(response.headers.get("content-length"))


def resolve_google_drive_download(session: requests.Session, url: str) -> tuple[str, dict[str, str]]:
    file_id = extract_google_drive_file_id(url)
    probe_url = "https://drive.google.com/uc"
    with session.get(
        probe_url,
        params={"id": file_id},
        stream=True,
        timeout=(30, 60),
        allow_redirects=True,
    ) as response:
        response.raise_for_status()
        content_type = (response.headers.get("content-type") or "").lower()
        if "text/html" not in content_type:
            return response.url, {}

        html = response.text
        action_match = GOOGLE_DRIVE_CONFIRM_ACTION_PATTERN.search(html)
        inputs = dict(GOOGLE_DRIVE_INPUT_PATTERN.findall(html))
        if action_match is not None and "confirm" in inputs:
            return urljoin(response.url, action_match.group(1)), inputs

    _raise_for_google_drive_html(html, url)
    raise RuntimeError(f"Google Drive download page did not expose a downloadable file: {url}")


def download_file(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    normalized_url = normalize_download_url(url)
    current_offset = destination.stat().st_size if destination.exists() else 0
    total_size: int | None = None

    print(f"[download] {normalized_url}")
    print(f"[save] {destination}")
    if current_offset > 0:
        print(f"[resume] from byte {current_offset}")

    while total_size is None or current_offset < total_size:
        range_start = current_offset
        headers = {"User-Agent": "Mozilla/5.0"}
        if range_start > 0:
            headers["Range"] = f"bytes={range_start}-"

        last_error: Exception | None = None
        for attempt in range(1, HTTP_DOWNLOAD_MAX_RETRIES + 1):
            try:
                with requests.get(
                    normalized_url,
                    headers=headers,
                    stream=True,
                    timeout=(30, 300),
                    allow_redirects=True,
                ) as response:
                    if response.status_code == 416 and range_start > 0:
                        remote_size = _probe_remote_file_size(normalized_url)
                        if remote_size is not None and current_offset >= remote_size:
                            return destination
                        raise RuntimeError(
                            f"Server rejected resume request at byte {range_start}: {normalized_url}"
                        )

                    response.raise_for_status()
                    _raise_for_html_response(response, normalized_url)

                    if range_start > 0 and response.status_code not in (200, 206):
                        raise RuntimeError(
                            f"HTTP server did not honor resume request from byte {range_start}: {normalized_url}"
                        )

                    mode = "ab" if range_start > 0 and response.status_code == 206 else "wb"
                    effective_offset = range_start if mode == "ab" else 0
                    if mode == "wb" and range_start > 0:
                        print(f"[restart] server ignored resume request; restarting download from byte 0")

                    resolved_total_size = _resolve_response_total_size(response, effective_offset)
                    if resolved_total_size is not None:
                        total_size = resolved_total_size

                    bytes_written = 0
                    with destination.open(mode) as output_file:
                        try:
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                if chunk:
                                    output_file.write(chunk)
                                    bytes_written += len(chunk)
                        except Exception:
                            output_file.flush()
                            output_file.truncate(effective_offset)
                            raise

                if bytes_written == 0:
                    raise RuntimeError(f"HTTP download returned no data from byte {range_start}: {normalized_url}")

                current_offset = destination.stat().st_size
                if total_size is not None and current_offset < total_size:
                    raise RuntimeError(
                        f"HTTP download ended early at byte {current_offset} / {total_size}: {normalized_url}"
                    )
                if total_size is None:
                    current_offset = destination.stat().st_size
                break
            except Exception as exc:
                last_error = exc
                if destination.exists() and destination.stat().st_size > range_start and range_start > 0:
                    with destination.open("r+b") as output_file:
                        output_file.truncate(range_start)
                if attempt == HTTP_DOWNLOAD_MAX_RETRIES:
                    raise
                print(
                    f"[retry] http download from byte {range_start} attempt {attempt}/{HTTP_DOWNLOAD_MAX_RETRIES}: {exc}"
                )
        else:
            if last_error is not None:
                raise last_error

    return destination


def download_from_google_drive(url: str, destination: Path, folder: bool = False) -> Path:
    target_dir = destination if folder else destination.parent
    ensure_dir(target_dir)

    if folder:
        command = [
            sys.executable,
            "-m",
            "gdown",
            "--fuzzy",
            "--continue",
            "--folder",
            url,
            "-O",
            str(destination),
        ]
        print(f"[gdown] {' '.join(command)}")
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                "Google Drive folder download failed. Install gdown and verify the shared link is still accessible."
            )
        return destination

    current_offset = destination.stat().st_size if destination.exists() else 0
    total_size: int | None = None

    print(f"[gdrive] {url}")
    print(f"[save] {destination}")
    if current_offset > 0:
        print(f"[resume] from byte {current_offset}")

    while total_size is None or current_offset < total_size:
        range_start = current_offset
        range_end = range_start + GOOGLE_DRIVE_RANGE_SIZE - 1
        range_header = f"bytes={range_start}-{range_end}"
        last_error: Exception | None = None

        for attempt in range(1, GOOGLE_DRIVE_MAX_RETRIES + 1):
            session = requests.Session()
            try:
                download_url, params = resolve_google_drive_download(session, url)
                with session.get(
                    download_url,
                    params=params,
                    headers={"User-Agent": "Mozilla/5.0", "Range": range_header},
                    stream=True,
                    timeout=(30, 300),
                    allow_redirects=True,
                ) as response:
                    response.raise_for_status()
                    content_type = (response.headers.get("content-type") or "").lower()
                    if "text/html" in content_type:
                        _raise_for_google_drive_html(response.text[:4000], url)
                    if range_start > 0 and response.status_code != 206:
                        raise RuntimeError(
                            f"Google Drive did not honor resume request {range_header}: {url}"
                        )

                    parsed_total = _parse_content_range_total(response.headers.get("content-range"))
                    if parsed_total is not None:
                        total_size = parsed_total
                    elif total_size is None and response.status_code == 200:
                        content_length = response.headers.get("content-length")
                        if content_length is not None and content_length.isdigit():
                            total_size = range_start + int(content_length)

                    mode = "ab" if range_start > 0 else "wb"
                    bytes_written = 0
                    with destination.open(mode) as output_file:
                        try:
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                if chunk:
                                    output_file.write(chunk)
                                    bytes_written += len(chunk)
                        except Exception:
                            output_file.flush()
                            output_file.truncate(range_start)
                            raise

                if bytes_written == 0:
                    raise RuntimeError(f"Google Drive returned no data for range {range_header}: {url}")

                current_offset += bytes_written
                if total_size is None and bytes_written < GOOGLE_DRIVE_RANGE_SIZE:
                    total_size = current_offset
                break
            except Exception as exc:
                last_error = exc
                if destination.exists() and destination.stat().st_size > range_start:
                    with destination.open("r+b") as output_file:
                        output_file.truncate(range_start)
                if attempt == GOOGLE_DRIVE_MAX_RETRIES:
                    raise
                print(
                    f"[retry] google drive range {range_header} attempt {attempt}/{GOOGLE_DRIVE_MAX_RETRIES}: {exc}"
                )
        else:
            if last_error is not None:
                raise last_error

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
