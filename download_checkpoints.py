#!/usr/bin/env python3
"""
Helper script to download Pi3 and Wan2.2 TI2V-5B checkpoints from Hugging Face.

Examples:
    python download_checkpoints.py \
        --pi3-dir ./pi3_ckpt \
        --wan-dir ./Wan2.2-TI2V-5B \
        --token $HF_TOKEN
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional


def _import_hf():
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download
    except ImportError as exc:
        print(
            "huggingface_hub is required. Install with `pip install huggingface_hub`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Pi3 and Wan2.2 TI2V-5B checkpoints from Hugging Face."
    )
    parser.add_argument(
        "--pi3-id",
        default="yyfz233/Pi3",
        help="Hugging Face repo id for Pi3 weights.",
    )
    parser.add_argument(
        "--pi3-dir",
        default="pi3_ckpt",
        help="Local directory to store Pi3 checkpoint files.",
    )
    parser.add_argument(
        "--wan-id",
        default="Wan-AI/Wan2.2-TI2V-5B",
        help="Hugging Face repo id for Wan2.2 TI2V-5B weights.",
    )
    parser.add_argument(
        "--wan-dir",
        default="Wan2.2-TI2V-5B",
        help="Local directory to store Wan2.2 TI2V-5B checkpoint files.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. If omitted, uses HF_TOKEN env var when available.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional revision/tag/commit to download for both repos.",
    )
    return parser.parse_args()


def _download(repo_id: str, target_dir: Path, token: Optional[str], revision: Optional[str]):
    snapshot_download = _import_hf()
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
        revision=revision,
        resume_download=True,
    )


def main():
    args = _parse_args()
    token = args.token or os.getenv("HF_TOKEN")
    repo_root = Path(__file__).resolve().parent

    _download(args.pi3_id, repo_root / args.pi3_dir, token, args.revision)
    _download(args.wan_id, repo_root / args.wan_dir, token, args.revision)
    print("Done.")


if __name__ == "__main__":
    main()
