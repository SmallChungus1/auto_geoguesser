from __future__ import annotations

import random
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image


ImageInput = Image.Image | str | Path


def _ensure_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    with Image.open(image) as loaded:
        return loaded.convert("RGB")


def _pick_flag_path(flags_dir: Path, rng: random.Random) -> Path:
    flag_paths = sorted(
        path for path in flags_dir.iterdir() if path.is_file() and path.suffix.lower() == ".svg"
    )
    if not flag_paths:
        raise FileNotFoundError(f"No SVG flags found in {flags_dir}")
    return rng.choice(flag_paths)


def _load_flag_svg(flag_path: Path) -> Image.Image:
    import resvg_py

    png_bytes = resvg_py.svg_to_bytes(svg_string=flag_path.read_text())
    with Image.open(BytesIO(png_bytes)) as flag_image:
        return flag_image.convert("RGBA")


def overlay_random_flag(
    image: ImageInput,
    flags_dir: str | Path = "country_flags",
    *,
    min_scale: float = 0.45,
    max_scale: float = 0.55,
    rng: random.Random | None = None,
) -> Image.Image:
    """
    Overlay a small random country flag onto an image.

    The result keeps the flag intentionally small so it can be used as an
    augmentation before GeoCLIP inference.
    """
    rng = rng or random.Random()
    flags_dir = Path(flags_dir)

    base = _ensure_image(image).convert("RGBA")
    flag = _load_flag_svg(_pick_flag_path(flags_dir, rng))

    base_width, base_height = base.size
    max_dim = min(base_width, base_height)
    scale = rng.uniform(min_scale, max_scale)
    flag_size = max(12, int(max_dim * scale))

    aspect_ratio = flag.width / max(flag.height, 1)
    resized_width = max(12, flag_size)
    resized_height = max(12, int(resized_width / max(aspect_ratio, 1e-6)))
    flag = flag.resize((resized_width, resized_height), Image.Resampling.LANCZOS)

    flag.putalpha(flag.getchannel("A"))

    padding = max(4, int(max_dim * 0.02))
    positions = [
        (padding, padding),
        (base_width - resized_width - padding, padding),
        (padding, base_height - resized_height - padding),
        (base_width - resized_width - padding, base_height - resized_height - padding),
    ]
    x, y = rng.choice(positions)
    x = max(0, x)
    y = max(0, y)

    base.alpha_composite(flag, dest=(x, y))
    return base.convert("RGB")


def overlay_random_flag_batch(
    images: Iterable[ImageInput] | Sequence[ImageInput],
    flags_dir: str | Path = "country_flags",
    *,
    min_scale: float = 0.45,
    max_scale: float = 0.55,
    rng: random.Random | None = None,
) -> list[Image.Image]:
    rng = rng or random.Random()
    return [
        overlay_random_flag(
            image,
            flags_dir=flags_dir,
            min_scale=min_scale,
            max_scale=max_scale,
            rng=rng,
        )
        for image in images
    ]
