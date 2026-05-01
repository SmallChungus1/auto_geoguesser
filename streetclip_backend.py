from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


class StreetCLIPBackend:
    def __init__(self, model_path: str = "geolocal/StreetCLIP") -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("StreetCLIP mode requires `torch` and `transformers`.") from exc

        self._torch = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        local_files_only = Path(model_path).exists()
        self._processor = CLIPProcessor.from_pretrained(model_path, local_files_only=local_files_only)
        self._model = CLIPModel.from_pretrained(model_path, local_files_only=local_files_only).to(self._device)
        self._model.eval()

    def predict(self, image_path: Path, labels: list[str], top_k: int) -> pd.DataFrame:
        cleaned_labels = [label.strip() for label in labels if label.strip()]
        if not cleaned_labels:
            raise ValueError("StreetCLIP requires at least one candidate label.")

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(text=cleaned_labels, images=image, return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.inference_mode():
            outputs = self._model(**inputs)
            probabilities = outputs.logits_per_image.softmax(dim=1)[0].detach().cpu()

        top_indices = probabilities.argsort(descending=True)[:top_k].tolist()
        rows = []
        for rank, index in enumerate(top_indices, start=1):
            rows.append(
                {
                    "rank": rank,
                    "label": cleaned_labels[index],
                    "probability": float(probabilities[index].item()),
                }
            )
        return pd.DataFrame(rows)
