from __future__ import annotations

import html
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests

from app.config import (
    ONNX_MODEL_FAMILY,
    ONNX_FAMILY_CONFIG,
    ONNX_DRIVE_FOLDERS,
)
from app.services.cache_service import set_onnx_family_cache

# Simple in-memory cache to avoid re-scraping Drive for every request
_catalog_cache: Dict[str, object] = {
    "m2m": {"ts": 0.0, "files": []},
    "nllb": {"ts": 0.0, "files": []},
}
CATALOG_TTL_SECONDS = 300


def _normalize_family(family: str | None = None) -> str:
    selected = (family or ONNX_MODEL_FAMILY or "m2m").strip().lower()
    return selected if selected in ONNX_FAMILY_CONFIG else "m2m"


def _family_settings(family: str | None = None) -> Dict[str, object]:
    selected = _normalize_family(family)
    folder_info = ONNX_DRIVE_FOLDERS[selected]
    family_cfg = ONNX_FAMILY_CONFIG[selected]
    return {
        "family": selected,
        "models_dir": family_cfg["models_dir"],
        "tokenizer_dir": family_cfg["tokenizer_dir"],
        "tokenizer_model": family_cfg["tokenizer_model"],
        "default_files": list(family_cfg["default_files"]),
        "folder_url": folder_info["url"],
        "folder_id": folder_info["id"],
    }


def _embedded_folder_list_url(family: str | None = None) -> str:
    settings = _family_settings(family)
    return f"https://drive.google.com/embeddedfolderview?id={settings['folder_id']}#list"


def _normalize_name(name: str) -> str:
    return " ".join(name.strip().split())


def _parse_drive_embedded_listing(content: str) -> List[Dict[str, str]]:
    # Example anchor: <a ... href="/file/d/<id>/view?...">file_name.onnx</a>
    anchor_re = re.compile(
        r'<a[^>]*href="([^\"]*/file/d/([A-Za-z0-9_-]+)[^\"]*)"[^>]*>(.*?)</a>',
        re.S,
    )

    out: List[Dict[str, str]] = []
    seen: set[Tuple[str, str]] = set()
    for href, file_id, inner_html in anchor_re.findall(content):
        label = re.sub(r"<[^<]+?>", "", inner_html)
        label = _normalize_name(html.unescape(label))
        if not label:
            continue

        full_link = href if href.startswith("http") else f"https://drive.google.com{href}"
        key = (file_id, label)
        if key in seen:
            continue
        seen.add(key)

        out.append({
            "id": file_id,
            "name": label,
            "view_url": full_link,
        })

    return out


def _fetch_drive_catalog(family: str | None = None) -> List[Dict[str, str]]:
    resp = requests.get(_embedded_folder_list_url(family), timeout=30)
    resp.raise_for_status()
    files = _parse_drive_embedded_listing(resp.text)
    if not files:
        raise RuntimeError("No files found in Drive folder listing")
    return files


def get_onnx_catalog(force_refresh: bool = False, family: str | None = None) -> Dict[str, object]:
    settings = _family_settings(family)
    selected = settings["family"]
    now = time.time()
    cache_bucket = _catalog_cache.setdefault(selected, {"ts": 0.0, "files": []})
    cached_ts = float(cache_bucket.get("ts") or 0.0)
    cached_files = cache_bucket.get("files") or []

    if (not force_refresh) and cached_files and (now - cached_ts < CATALOG_TTL_SECONDS):
        files = cached_files
    else:
        files = _fetch_drive_catalog(selected)
        cache_bucket["ts"] = now
        cache_bucket["files"] = files

    sorted_files = sorted(files, key=lambda item: item["name"].lower())
    return {
        "family": selected,
        "folder_url": settings["folder_url"],
        "folder_id": settings["folder_id"],
        "default_files": settings["default_files"],
        "files": sorted_files,
    }


def _destination_for_model(filename: str, family: str | None = None) -> Path:
    settings = _family_settings(family)
    models_dir: Path = settings["models_dir"]
    name = filename.lower()
    if "encoder" in name:
        return models_dir / "encoder" / filename
    if "decoder" in name:
        return models_dir / "decoder" / filename
    if "lm_head" in name or name.startswith("lm_head"):
        return models_dir / "lm_head" / filename
    return models_dir / filename


def is_default_onnx_models_ready(family: str | None = None) -> bool:
    settings = _family_settings(family)
    for filename in settings["default_files"]:
        path = _destination_for_model(filename, settings["family"])
        if (not path.exists()) or path.stat().st_size < 1024:
            return False
        try:
            with path.open("rb") as handle:
                prefix = handle.read(256).lower()
                if b"<!doctype html" in prefix or b"<html" in prefix:
                    return False
        except Exception:
            return False
    return True


def ensure_default_onnx_models(force_download: bool = False, family: str | None = None) -> Dict[str, object]:
    settings = _family_settings(family)
    selected = settings["family"]
    if is_default_onnx_models_ready(selected) and not force_download:
        return {
            "ready": True,
            "downloaded": False,
            "family": selected,
            "requested_files": settings["default_files"],
        }

    details = download_onnx_models(
        selected_files=settings["default_files"],
        include_tokenizer=False,
        family=selected,
    )
    return {
        "ready": is_default_onnx_models_ready(selected),
        "downloaded": True,
        "family": selected,
        **details,
    }


def list_downloaded_onnx_model_files(family: str | None = None) -> List[str]:
    settings = _family_settings(family)
    selected = settings["family"]
    models_dir: Path = settings["models_dir"]
    if not models_dir.exists():
        set_onnx_family_cache(selected, downloaded_files=[], tokenizer_ready=is_onnx_tokenizer_ready(selected))
        return []

    files: List[str] = []
    for path in models_dir.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if not (name.endswith(".onnx") or name.endswith(".onnx.data")):
            continue
        try:
            if path.stat().st_size <= 0:
                continue
        except Exception:
            continue
        files.append(str(path.relative_to(models_dir)).replace("\\", "/"))

    files = sorted(files)
    set_onnx_family_cache(selected, downloaded_files=files, tokenizer_ready=is_onnx_tokenizer_ready(selected))
    return files


def is_onnx_tokenizer_ready(family: str | None = None) -> bool:
    settings = _family_settings(family)
    tokenizer_dir: Path = settings["tokenizer_dir"]
    if not tokenizer_dir.exists():
        return False

    required_config = tokenizer_dir / "tokenizer_config.json"
    if (not required_config.exists()) or required_config.stat().st_size <= 0:
        return False

    candidate_tokenizer_assets = [
        tokenizer_dir / "sentencepiece.bpe.model",
        tokenizer_dir / "sentencepiece.model",
        tokenizer_dir / "tokenizer.model",
        tokenizer_dir / "tokenizer.json",
    ]

    for asset in candidate_tokenizer_assets:
        try:
            if asset.exists() and asset.stat().st_size > 0:
                return True
        except Exception:
            continue

    return False


def ensure_onnx_tokenizer(force_download: bool = False, family: str | None = None) -> Dict[str, object]:
    settings = _family_settings(family)
    selected = settings["family"]
    tokenizer_dir: Path = settings["tokenizer_dir"]
    tokenizer_model: str = settings["tokenizer_model"]

    if is_onnx_tokenizer_ready(selected) and not force_download:
        set_onnx_family_cache(selected, tokenizer_ready=True)
        return {
            "ready": True,
            "family": selected,
            "path": str(tokenizer_dir),
            "downloaded": False,
            "model": tokenizer_model,
        }

    try:
        from transformers import AutoTokenizer, M2M100Tokenizer
    except Exception as e:
        raise RuntimeError(f"transformers is required to prepare tokenizer: {e}") from e

    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    try:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        except Exception:
            tokenizer = M2M100Tokenizer.from_pretrained(tokenizer_model)
        tokenizer.save_pretrained(str(tokenizer_dir))
    except Exception as e:
        raise RuntimeError(
            f"Failed to download/save tokenizer '{tokenizer_model}' to {tokenizer_dir}: {e}"
        ) from e

    if not is_onnx_tokenizer_ready(selected):
        raise RuntimeError(f"Tokenizer files are incomplete at {tokenizer_dir}")

    set_onnx_family_cache(selected, tokenizer_ready=True)

    return {
        "ready": True,
        "family": selected,
        "path": str(tokenizer_dir),
        "downloaded": True,
        "model": tokenizer_model,
    }


def _download_file_from_drive(file_id: str, destination: Path) -> None:
    # Handles Drive's confirmation flow for larger files.
    session = requests.Session()
    base_url = "https://drive.google.com/uc?export=download"
    fallback_url = "https://drive.usercontent.google.com/download"

    def _looks_like_html(resp: requests.Response) -> bool:
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" in content_type:
            return True
        try:
            prefix = resp.content[:512].lower()
            return b"<!doctype html" in prefix or b"<html" in prefix
        except Exception:
            return False

    def _extract_confirm_params(page_html: str) -> Dict[str, str]:
        params: Dict[str, str] = {}
        for key in ("id", "confirm", "uuid"):
            m = re.search(rf'name="{key}"\s+value="([^\"]+)"', page_html)
            if m:
                params[key] = m.group(1)

        if "confirm" not in params:
            m = re.search(r"confirm=([0-9A-Za-z_\-]+)", page_html)
            if m:
                params["confirm"] = m.group(1)

        if "id" not in params:
            params["id"] = file_id
        return params

    response = session.get(base_url, params={"id": file_id}, stream=True, timeout=60)
    response.raise_for_status()

    confirm_token = None
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    if confirm_token:
        response = session.get(
            base_url,
            params={"id": file_id, "confirm": confirm_token},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

    if _looks_like_html(response):
        html_text = response.text
        confirm_params = _extract_confirm_params(html_text)
        if confirm_params.get("confirm"):
            response = session.get(base_url, params=confirm_params, stream=True, timeout=60)
            response.raise_for_status()

    if _looks_like_html(response):
        response = session.get(
            fallback_url,
            params={"id": file_id, "export": "download", "confirm": "t"},
            stream=True,
            timeout=60,
        )
        response.raise_for_status()

    if _looks_like_html(response):
        raise RuntimeError(
            "Google Drive returned an HTML page instead of model binary (possible quota/permission/confirmation issue)."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def download_onnx_models(
    selected_files: List[str] | None = None,
    include_tokenizer: bool = True,
    family: str | None = None,
) -> Dict[str, object]:
    settings = _family_settings(family)
    selected = settings["family"]
    catalog = get_onnx_catalog(force_refresh=False, family=selected)
    available = {item["name"]: item for item in catalog["files"]}

    requested = selected_files or settings["default_files"]
    requested = [_normalize_name(name) for name in requested if str(name).strip()]

    alias_map = {
        "lm_head.onnx.data": "m2m100_lm_head.onnx.data",
    }
    requested = [
        alias_map.get(name, name)
        for name in requested
    ]

    if not requested:
        raise RuntimeError("No model files requested")

    unknown = [name for name in requested if name not in available]
    if unknown:
        raise RuntimeError(f"Requested files not found in Drive folder: {', '.join(unknown)}")

    # Auto-include sidecar external tensor files when present (e.g., *.onnx.data)
    expanded_requested: List[str] = list(requested)
    for filename in requested:
        if filename.endswith(".onnx"):
            sidecar = f"{filename}.data"
            if sidecar in available and sidecar not in expanded_requested:
                expanded_requested.append(sidecar)

    progress: List[str] = []
    downloaded: List[Dict[str, str]] = []
    for filename in expanded_requested:
        file_id = available[filename]["id"]
        destination = _destination_for_model(filename, selected)
        progress.append(f"Downloading {filename}")
        _download_file_from_drive(file_id, destination)
        progress.append(f"Downloaded {filename}")
        downloaded.append({
            "name": filename,
            "id": file_id,
            "saved_to": str(destination),
        })

    tokenizer_info = None
    if include_tokenizer:
        tokenizer_info = ensure_onnx_tokenizer(force_download=False, family=selected)

    downloaded_files = list_downloaded_onnx_model_files(selected)
    tokenizer_ready = bool(tokenizer_info["ready"]) if isinstance(tokenizer_info, dict) else is_onnx_tokenizer_ready(selected)
    set_onnx_family_cache(selected, downloaded_files=downloaded_files, tokenizer_ready=tokenizer_ready)

    return {
        "family": selected,
        "folder_url": settings["folder_url"],
        "requested_files": expanded_requested,
        "progress": progress,
        "downloaded": downloaded,
        "tokenizer": tokenizer_info,
    }
