import os
import json
import time
import hashlib
import re
import logging
from typing import Dict, Optional

from .config import CACHE_DIR

logger = logging.getLogger(__name__)


def _pdf_content_hash(pdf_path):
    with open(pdf_path, "rb") as f:
        return hashlib.sha256(f.read(8192)).hexdigest()


def _cache_key(pdf_path, model, provider):
    h = _pdf_content_hash(pdf_path)
    safe_model = re.sub(r'[^a-zA-Z0-9_-]', '_', model)
    safe_provider = re.sub(r'[^a-zA-Z0-9_-]', '_', provider)
    return f"{safe_provider}_{safe_model}_{h}"


def _get_cached_result(pdf_path, model, provider, max_age_hours=24):
    key = _cache_key(pdf_path, model, provider)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    if not os.path.exists(cache_file):
        return None
    try:
        mtime = os.path.getmtime(cache_file)
        age_hours = (time.time() - mtime) / 3600
        if age_hours > max_age_hours:
            logger.info(f"[ChemExtract] Cache expired ({age_hours:.1f}h old)")
            return None
        logger.info(f"[ChemExtract] Loading cached result ({age_hours:.1f}h old)")
        with open(cache_file, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        # OSError: file permission / disk issues. json.JSONDecodeError:
        # corrupt cache file. ValueError: unexpected JSON structure.
        logger.warning(f"[ChemExtract] Failed to load cache: {e}")
        return None


def _save_cached_result(result, pdf_path, model, provider):
    key = _cache_key(pdf_path, model, provider)
    cache_file = os.path.join(CACHE_DIR, f"{key}.json")
    try:
        with open(cache_file, "w") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info(f"[ChemExtract] Saved result to cache: {cache_file}")
    except (OSError, TypeError, ValueError) as e:
        # OSError: disk full / permission denied. TypeError: result contains
        # non-serializable objects that json.dump can't handle with default=str.
        logger.warning(f"[ChemExtract] Failed to save cache: {e}")
