from __future__ import annotations

import random
from io import BytesIO
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw


ImageInput = Image.Image | str | Path


def _ensure_image(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    with Image.open(image) as loaded:
        return loaded.convert("RGB")


def compress_jpeg_image(
    image: ImageInput,
    quality: int = 75,
    *,
    optimize: bool = True,
    progressive: bool = True,
) -> Image.Image:
    """
    Re-encode an image as JPEG and load it back into memory.

    Lower quality means stronger compression.
    """
    image_rgb = _ensure_image(image).convert("RGB")
    buffer = BytesIO()
    image_rgb.save(buffer, format="JPEG", quality=quality, optimize=optimize, progressive=progressive)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def compress_jpeg_batch(
    images: Iterable[ImageInput] | Sequence[ImageInput],
    quality: int = 75,
    *,
    optimize: bool = True,
    progressive: bool = True,
) -> list[Image.Image]:
    return [
        compress_jpeg_image(
            image,
            quality=quality,
            optimize=optimize,
            progressive=progressive,
        )
        for image in images
    ]


def add_random_shapes_image(
    image: ImageInput,
    *,
    num_shapes: int = 3,
    max_shape_scale: float = 0.18,
    rng: random.Random | None = None,
) -> Image.Image:
    """
    Draw opaque random squares and circles onto an image.

    This is meant as a simple perturbation for robustness testing.
    """
    rng = rng or random.Random()
    base = _ensure_image(image).convert("RGBA")
    draw = ImageDraw.Draw(base, "RGBA")

    width, height = base.size
    max_dim = min(width, height)
    shape_max = max(10, int(max_dim * max_shape_scale))

    for _ in range(num_shapes):
        shape_type = rng.choice(["square", "circle"])
        size = rng.randint(max(8, shape_max // 2), shape_max)
        x0 = rng.randint(0, max(0, width - size))
        y0 = rng.randint(0, max(0, height - size))
        x1 = x0 + size
        y1 = y0 + size
        color = (
            rng.randint(0, 255),
            rng.randint(0, 255),
            rng.randint(0, 255),
            255,
        )
        if shape_type == "square":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)

    return base.convert("RGB")


def add_random_shapes_batch(
    images: Iterable[ImageInput] | Sequence[ImageInput],
    *,
    num_shapes: int = 3,
    max_shape_scale: float = 0.18,
    rng: random.Random | None = None,
) -> list[Image.Image]:
    rng = rng or random.Random()
    return [
        add_random_shapes_image(
            image,
            num_shapes=num_shapes,
            max_shape_scale=max_shape_scale,
            rng=rng,
        )
        for image in images
    ]
