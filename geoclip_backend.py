from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd
from PIL import Image


class GeoClipBackend(Protocol):
    def predict(self, image_path: Path, top_k: int) -> pd.DataFrame:
        ...


def _patch_transformers_clip_features() -> None:
    """
    GeoCLIP 1.2.0 expects CLIPModel.get_image_features() to return a tensor.
    Transformers 5.x returns a BaseModelOutputWithPooling instead, so we unwrap
    pooler_output for compatibility.
    """
    try:
        from transformers import CLIPModel
    except Exception:  # pragma: no cover - only relevant when transformers is installed
        return

    current = CLIPModel.get_image_features
    if getattr(current, "_geoclip_compat_patched", False):
        return

    def patched(self, *args, **kwargs):
        outputs = current(self, *args, **kwargs)
        return getattr(outputs, "pooler_output", outputs)

    patched._geoclip_compat_patched = True  # type: ignore[attr-defined]
    CLIPModel.get_image_features = patched  # type: ignore[assignment]


@dataclass(frozen=True)
class PredictionBackendConfig:
    mode: str = os.getenv("GEOCLIP_MODE", "auto").strip().lower()
    model_dir: Path = Path(os.getenv("GEOCLIP_MODEL_DIR", "models/geoclip-large-patch14"))
    vision_model_file: str = os.getenv("GEOCLIP_VISION_MODEL_FILE", "vision_model_int8.onnx")
    location_model_file: str = os.getenv("GEOCLIP_LOCATION_MODEL_FILE", "location_model_quantized.onnx")
    batch_size: int = int(os.getenv("GEOCLIP_BATCH_SIZE", "512"))


class OnlineGeoCLIPBackend:
    def __init__(self) -> None:
        _patch_transformers_clip_features()
        from geoclip import GeoCLIP

        self._model = GeoCLIP()

    def predict(self, image_path: Path, top_k: int) -> pd.DataFrame:
        top_pred_gps, top_pred_prob = self._model.predict(str(image_path), top_k=top_k)
        rows = []
        for rank, ((lat, lon), probability) in enumerate(zip(top_pred_gps, top_pred_prob), start=1):
            rows.append(
                {
                    "rank": rank,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "probability": float(probability),
                }
            )
        return pd.DataFrame(rows)


class OfflineONNXGeoCLIPBackend:
    def __init__(self, model_dir: Path, vision_model_file: str, location_model_file: str, batch_size: int) -> None:
        try:
            import numpy as np
            import onnxruntime as ort
            from transformers import CLIPImageProcessor
        except ImportError as exc:  # pragma: no cover - defensive guard for runtime deps
            raise RuntimeError(
                "Offline GeoCLIP mode requires `onnxruntime`, `transformers`, and `huggingface_hub`."
            ) from exc

        self._np = np
        self._batch_size = batch_size
        self._processor = CLIPImageProcessor.from_pretrained(model_dir, local_files_only=True)
        self._vision_session = ort.InferenceSession(
            str(model_dir / "onnx" / vision_model_file),
            providers=["CPUExecutionProvider"],
        )
        self._location_session = ort.InferenceSession(
            str(model_dir / "onnx" / location_model_file),
            providers=["CPUExecutionProvider"],
        )

        coordinates_path = model_dir / "gps_gallery" / "coordinates_100K.json"
        if not coordinates_path.exists():
            raise FileNotFoundError(
                f"Missing GPS gallery file: {coordinates_path}. "
                "Run the download script to fetch the GeoCLIP snapshot."
            )
        self._gps_data = self._np.asarray(json.loads(coordinates_path.read_text()), dtype=self._np.float32)

        self._image_input_name = self._vision_session.get_inputs()[0].name
        self._location_input_name = self._location_session.get_inputs()[0].name
        self._image_output_name = self._vision_session.get_outputs()[0].name
        self._location_output_name = self._location_session.get_outputs()[0].name
        self._logit_scale = math.exp(3.681034803390503)

    def _normalize(self, array: "np.ndarray") -> "np.ndarray":
        norms = self._np.linalg.norm(array, axis=-1, keepdims=True)
        return array / self._np.clip(norms, a_min=1e-12, a_max=None)

    def _softmax(self, values: "np.ndarray") -> "np.ndarray":
        values = values - values.max()
        exp_values = self._np.exp(values)
        return exp_values / exp_values.sum()

    def predict(self, image_path: Path, top_k: int) -> pd.DataFrame:
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="np")
        vision_inputs = inputs["pixel_values"].astype(self._np.float32)
        image_embeds = self._vision_session.run([self._image_output_name], {self._image_input_name: vision_inputs})[0]
        norm_image_embeds = self._normalize(image_embeds)[0]

        scores: list[float] = []
        for start in range(0, len(self._gps_data), self._batch_size):
            chunk = self._gps_data[start : start + self._batch_size]
            location_embeds = self._location_session.run(
                [self._location_output_name],
                {self._location_input_name: chunk.astype(self._np.float32)},
            )[0]
            norm_location_embeds = self._normalize(location_embeds)
            scores.extend((self._logit_scale * (norm_location_embeds @ norm_image_embeds)).tolist())

        probabilities = self._softmax(self._np.asarray(scores, dtype=self._np.float32))
        top_indices = probabilities.argsort()[::-1][:top_k]

        rows = []
        for rank, index in enumerate(top_indices, start=1):
            lat, lon = self._gps_data[index]
            rows.append(
                {
                    "rank": rank,
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "probability": float(probabilities[index]),
                }
            )

        return pd.DataFrame(rows)


def build_backend(config: PredictionBackendConfig | None = None) -> GeoClipBackend:
    config = config or PredictionBackendConfig()
    if config.mode == "online":
        return OnlineGeoCLIPBackend()
    if config.mode == "offline":
        return OfflineONNXGeoCLIPBackend(
            model_dir=config.model_dir,
            vision_model_file=config.vision_model_file,
            location_model_file=config.location_model_file,
            batch_size=config.batch_size,
        )

    if config.model_dir.exists():
        try:
            return OfflineONNXGeoCLIPBackend(
                model_dir=config.model_dir,
                vision_model_file=config.vision_model_file,
                location_model_file=config.location_model_file,
                batch_size=config.batch_size,
            )
        except Exception:
            return OnlineGeoCLIPBackend()
    return OnlineGeoCLIPBackend()
