from __future__ import annotations

import os
import random
import tempfile
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from image_utils import add_random_shapes_image, compress_jpeg_image
from streetclip_backend import StreetCLIPBackend


st.set_page_config(
    page_title="StreetCLIP Demo",
    page_icon="🌍",
    layout="wide",
)


DEFAULT_LABELS = "\n".join(
    [
        "San Francisco, USA",
        "Los Angeles, USA",
        "Seattle, USA",
        "New York City, USA",
        "London, UK",
        "Paris, France",
        "Tokyo, Japan",
        "Berlin, Germany",
    ]
)


@st.cache_resource
def load_model(model_path: str) -> StreetCLIPBackend:
    return StreetCLIPBackend(model_path=model_path)


def parse_candidate_labels(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


def save_pil_image(image: Image.Image, suffix: str = ".png", format: str = "PNG") -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        image.save(tmp, format=format)
        return Path(tmp.name)


def stable_rng(uploaded_file_name: str, image: Image.Image, salt: str) -> random.Random:
    digest = sha256(salt.encode("utf-8") + uploaded_file_name.encode("utf-8") + image.tobytes()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


default_model_path = os.getenv("STREETCLIP_MODEL_PATH", "geolocal/StreetCLIP")

st.title("StreetCLIP Image Geolocation")
st.caption(
    "Upload an image, provide candidate `{city, country}` labels, and let StreetCLIP rank the most likely match."
)

with st.sidebar:
    st.header("Settings")
    model_path = st.text_input(
        "Model path or Hugging Face id",
        value=default_model_path,
        help="Use `geolocal/StreetCLIP` for the baseline model or point this to a local fine-tuned checkpoint folder.",
    )
    top_k = st.slider("Top-K predictions", min_value=1, max_value=10, value=5)
    apply_jpeg_compression = st.toggle("Apply JPEG compression", value=False)
    jpeg_quality = st.slider("JPEG quality", min_value=10, max_value=95, value=70)
    apply_shapes = st.toggle("Add random shapes", value=False)
    num_shapes = st.slider("Shape count", min_value=1, max_value=10, value=3)
    shape_scale = st.slider("Shape size", min_value=0.05, max_value=0.4, value=0.18, step=0.01)
    labels_text = st.text_area(
        "Candidate location labels",
        value=DEFAULT_LABELS,
        height=220,
        help="One candidate label per line. StreetCLIP will choose among these labels only.",
    )
    st.code(f"Model: {model_path}", language="text")


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is None:
    st.info("Choose an image to start.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption=uploaded_file.name, use_container_width=True)

inference_image = image

if apply_jpeg_compression:
    inference_image = compress_jpeg_image(inference_image, quality=jpeg_quality)
    st.caption(f"JPEG-compressed preview at quality={jpeg_quality}")
    st.image(inference_image, caption="StreetCLIP input after compression", use_container_width=True)

if apply_shapes:
    rng = stable_rng(uploaded_file.name, image, "shapes")
    inference_image = add_random_shapes_image(
        inference_image,
        num_shapes=num_shapes,
        max_shape_scale=shape_scale,
        rng=rng,
    )
    st.caption(f"Random shapes preview: {num_shapes} shape(s), size={shape_scale:.2f}")
    st.image(inference_image, caption="StreetCLIP input after shapes", use_container_width=True)

predict_button = st.button("Predict location", type="primary")

if predict_button:
    candidate_labels = parse_candidate_labels(labels_text)
    if not candidate_labels:
        st.warning("Add at least one candidate label.")
        st.stop()

    suffix = ".jpg" if apply_jpeg_compression else ".png"
    fmt = "JPEG" if apply_jpeg_compression else "PNG"
    tmp_path = save_pil_image(inference_image, suffix=suffix, format=fmt)
    try:
        model = load_model(model_path)
        effective_top_k = min(top_k, len(candidate_labels))
        with st.spinner("Running StreetCLIP inference..."):
            results = model.predict(tmp_path, labels=candidate_labels, top_k=effective_top_k)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if results.empty:
        st.warning("StreetCLIP did not return any predictions.")
        st.stop()

    best = results.iloc[0]
    st.subheader("Best guess")
    left, right = st.columns([1, 1])
    with left:
        st.markdown(f'**Predicted location**  \n{best["label"]}')
    with right:
        st.metric("Confidence", f'{best["probability"]:.4f}')

    st.subheader("Top predictions")
    st.dataframe(
        results[["rank", "label", "probability"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.write("Click **Predict location** to run StreetCLIP on the uploaded image.")
