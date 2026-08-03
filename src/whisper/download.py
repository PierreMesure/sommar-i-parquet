"""Download transcription inputs without holding media files in memory."""

from __future__ import annotations

from pathlib import Path

import httpx


USER_AGENT = "sommar-i-parquet/0.1 (https://github.com/PierreMesure/sommar-i-parquet)"


def download_file(url: str, destination: Path, *, overwrite: bool = False) -> Path:
    """Download *url* atomically, returning an existing file unless overwritten."""
    if destination.exists() and not overwrite:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f"{destination.suffix}.part")
    with httpx.stream(
        "GET",
        url,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0),
    ) as response:
        response.raise_for_status()
        with temporary.open("wb") as output:
            for chunk in response.iter_bytes():
                output.write(chunk)
    temporary.replace(destination)
    return destination


def download_huggingface_model(repo_id: str, destination: Path) -> Path:
    """Materialise a Hugging Face model under *destination* for reproducible local runs."""
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id,
        local_dir=destination,
        max_workers=1,
        headers={"User-Agent": USER_AGENT},
    )
    return destination
