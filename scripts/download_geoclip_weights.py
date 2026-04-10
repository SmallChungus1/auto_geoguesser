from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a local GeoCLIP snapshot for offline inference.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/geoclip-large-patch14"),
        help="Directory to store the downloaded snapshot.",
    )
    parser.add_argument(
        "--repo-id",
        default="Xenova/geoclip-large-patch14",
        help="Hugging Face repo id to download.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Repo revision to download.",
    )
    parser.add_argument(
        "--vision-model-file",
        default="vision_model_int8.onnx",
        help="Vision model filename inside the repo onnx/ folder.",
    )
    parser.add_argument(
        "--location-model-file",
        default="location_model_quantized.onnx",
        help="Location model filename inside the repo onnx/ folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json",
            "preprocessor_config.json",
            "quantize_config.json",
            "gps_gallery/coordinates_100K.json",
            f"onnx/{args.vision_model_file}",
            f"onnx/{args.location_model_file}",
        ],
    )
    print(f"Downloaded GeoCLIP snapshot to: {args.output_dir}")


if __name__ == "__main__":
    main()
