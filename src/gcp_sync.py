"""
gcp_sync.py — Push / Pull artifacts to Google Cloud Storage
=============================================================
Syncs three artifact groups between local disk and GCP bucket mlops_b1:

  Group          Local                        GCS prefix
  ─────────────────────────────────────────────────────────────
  raw data       artifacts/raw/               artifacts/raw/
  processed data artifacts/processed/         artifacts/processed/
  models         artifacts/models/            artifacts/models/

Usage:
  # Push everything to GCP (after training)
  python -m src.gcp_sync --push

  # Pull everything from GCP (fresh clone / new machine)
  python -m src.gcp_sync --pull

  # Push only models
  python -m src.gcp_sync --push --group models

  # Pull only models
  python -m src.gcp_sync --pull --group models

Groups: raw | processed | models | all (default)
"""

import os
import argparse
from pathlib import Path

from google.cloud import storage

from src.logger import get_logger
from src.custom_exception import CustomException
from utils.common_fnctions import read_yaml
from config.paths_config import CONFIG_PATH

logger = get_logger(__name__)

# ── Artifact groups ───────────────────────────────────────────────────────────
SYNC_GROUPS = {
    "raw"       : "artifacts/raw",
    "processed" : "artifacts/processed",
    "models"    : "artifacts/models",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_bucket(bucket_name: str) -> storage.Bucket:
    client = storage.Client()
    return client.bucket(bucket_name)


def _push_folder(bucket: storage.Bucket, local_dir: str, gcs_prefix: str) -> int:
    """
    Upload every file in local_dir to GCS under gcs_prefix/.
    Returns number of files uploaded.
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        logger.warning(f"Local folder not found — skipping: {local_dir}")
        return 0

    files = list(local_path.rglob("*"))
    files = [f for f in files if f.is_file()]

    if not files:
        logger.warning(f"No files found in {local_dir}")
        return 0

    uploaded = 0
    for file in files:
        relative   = file.relative_to(local_path)
        blob_name  = f"{gcs_prefix}/{relative}".replace("\\", "/")
        blob       = bucket.blob(blob_name)
        blob.upload_from_filename(str(file))
        logger.info(f"  ↑ {file} → gs://{bucket.name}/{blob_name}")
        uploaded += 1

    return uploaded


def _pull_folder(bucket: storage.Bucket, gcs_prefix: str, local_dir: str) -> int:
    """
    Download every blob under gcs_prefix/ to local_dir/.
    Returns number of files downloaded.
    """
    blobs = list(bucket.list_blobs(prefix=gcs_prefix + "/"))
    blobs = [b for b in blobs if not b.name.endswith("/")]

    if not blobs:
        logger.warning(f"No files found in gs://{bucket.name}/{gcs_prefix}/")
        return 0

    downloaded = 0
    for blob in blobs:
        # Strip the gcs_prefix to get relative path
        relative   = blob.name[len(gcs_prefix) + 1:]
        local_file = Path(local_dir) / relative
        local_file.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file))
        logger.info(f"  ↓ gs://{bucket.name}/{blob.name} → {local_file}")
        downloaded += 1

    return downloaded


# ── Push ──────────────────────────────────────────────────────────────────────

def push(bucket_name: str, groups: list) -> dict:
    """Push selected artifact groups to GCP."""
    try:
        logger.info(f"Connecting to bucket: gs://{bucket_name}")
        bucket  = _get_bucket(bucket_name)
        summary = {}

        for group in groups:
            local_dir  = SYNC_GROUPS[group]
            gcs_prefix = SYNC_GROUPS[group]
            logger.info(f"Pushing [{group}] {local_dir} → gs://{bucket_name}/{gcs_prefix}/")
            count = _push_folder(bucket, local_dir, gcs_prefix)
            summary[group] = {"uploaded": count, "destination": f"gs://{bucket_name}/{gcs_prefix}/"}
            logger.info(f"  ✅ {count} file(s) uploaded for [{group}]")

        return summary

    except Exception as e:
        logger.error(f"Push failed: {e}")
        raise CustomException("GCP push failed", e)


# ── Pull ──────────────────────────────────────────────────────────────────────

def pull(bucket_name: str, groups: list) -> dict:
    """Pull selected artifact groups from GCP."""
    try:
        logger.info(f"Connecting to bucket: gs://{bucket_name}")
        bucket  = _get_bucket(bucket_name)
        summary = {}

        for group in groups:
            local_dir  = SYNC_GROUPS[group]
            gcs_prefix = SYNC_GROUPS[group]
            logger.info(f"Pulling [{group}] gs://{bucket_name}/{gcs_prefix}/ → {local_dir}/")
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            count = _pull_folder(bucket, gcs_prefix, local_dir)
            summary[group] = {"downloaded": count, "source": f"gs://{bucket_name}/{gcs_prefix}/"}
            logger.info(f"  ✅ {count} file(s) downloaded for [{group}]")

        return summary

    except Exception as e:
        logger.error(f"Pull failed: {e}")
        raise CustomException("GCP pull failed", e)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    config      = read_yaml(CONFIG_PATH)
    bucket_name = config["data_ingestion"]["bucket_name"]

    parser = argparse.ArgumentParser(description="Sync artifacts with GCP bucket")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--push", action="store_true", help="Upload local artifacts to GCP")
    group.add_argument("--pull", action="store_true", help="Download artifacts from GCP to local")
    parser.add_argument(
        "--group",
        choices=["raw", "processed", "models", "all"],
        default="all",
        help="Which artifact group to sync (default: all)"
    )
    args = parser.parse_args()

    groups = list(SYNC_GROUPS.keys()) if args.group == "all" else [args.group]

    if args.push:
        logger.info(f"PUSH → gs://{bucket_name} | groups: {groups}")
        summary = push(bucket_name, groups)
        action  = "uploaded"
    else:
        logger.info(f"PULL ← gs://{bucket_name} | groups: {groups}")
        summary = pull(bucket_name, groups)
        action  = "downloaded"

    print("\n" + "=" * 50)
    print(f"  GCP SYNC COMPLETE — {args.push and 'PUSH' or 'PULL'}")
    print("=" * 50)
    for grp, info in summary.items():
        count = info.get(action, 0)
        loc   = info.get("destination") or info.get("source")
        print(f"  {grp:<12}: {count} file(s)  |  {loc}")
    print("=" * 50)


if __name__ == "__main__":
    main()
