from __future__ import annotations

import os
import random
import tempfile
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from PIL import Image

from geoclip_backend import (
    OfflineONNXGeoCLIPBackend,
    OnlineGeoCLIPBackend,
    StreetCLIPZeroShotBackend,
    build_backend,
)
from image_utils import add_random_shapes_image, compress_jpeg_image


st.set_page_config(
    page_title="Geo Localization Demo",
    page_icon="🌍",
    layout="wide",
)


@st.cache_resource
def load_model(model_family: str):
    if model_family == "streetclip":
        return StreetCLIPZeroShotBackend()
    return build_backend()


def backend_name(model) -> str:
    if isinstance(model, OfflineONNXGeoCLIPBackend):
        return "Offline ONNX"
    if isinstance(model, OnlineGeoCLIPBackend):
        return "Online GeoCLIP"
    if isinstance(model, StreetCLIPZeroShotBackend):
        return "StreetCLIP"
    return type(model).__name__


@st.cache_resource
def load_geocoder() -> Nominatim:
    return Nominatim(user_agent="geoclip-streamlit-demo")


@st.cache_data(show_spinner=False)
def reverse_geocode(lat: float, lon: float) -> str | None:
    geocoder = load_geocoder()
    try:
        location = geocoder.reverse((lat, lon), exactly_one=True, language="en", timeout=10)
    except (GeocoderTimedOut, GeocoderServiceError):
        return None

    if location is None:
        return None
    return location.address


def save_pil_image(image: Image.Image, suffix: str = ".png", format: str = "PNG") -> Path:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        image.save(tmp, format=format)
        return Path(tmp.name)


def stable_rng(uploaded_file_name: str, image: Image.Image, salt: str) -> random.Random:
    digest = sha256(salt.encode("utf-8") + uploaded_file_name.encode("utf-8") + image.tobytes()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


DEFAULT_STREETCLIP_LABELS = "\n".join(
    [
        "San Francisco, California, USA",
        "Los Angeles, California, USA",
        "Las Vegas, Nevada, USA",
        "Seattle, Washington, USA",
        "New York City, New York, USA",
        "London, England, UK",
        "Paris, France",
        "Tokyo, Japan",
    ]
)


def parse_candidate_labels(raw_text: str) -> list[str]:
    return [line.strip() for line in raw_text.splitlines() if line.strip()]


model_family = st.sidebar.selectbox(
    "Model",
    options=["GeoCLIP", "StreetCLIP"],
    index=0,
    help="GeoCLIP predicts GPS coordinates. StreetCLIP scores candidate place labels directly.",
)
model_key = model_family.lower()
model = load_model(model_key)
loaded_backend = backend_name(model)

st.title("Image Geolocalization")
st.markdown(f"**Backend:** `{loaded_backend}`")
if model_key == "streetclip":
    st.caption("Upload an image, provide candidate place labels, and let StreetCLIP rank the most likely match.")
else:
    st.caption("Upload an image, let GeoCLIP predict GPS coordinates, then reverse-geocode the best guess into a readable location.")

with st.sidebar:
    st.header("Settings")
    mode = os.getenv("GEOCLIP_MODE", "auto").strip().lower()
    model_dir = os.getenv("GEOCLIP_MODEL_DIR", "models/geoclip-large-patch14")
    debug_lines = [f"Backend: {loaded_backend}"]
    if model_key == "geoclip":
        debug_lines.insert(0, f"Mode: {mode}")
        debug_lines.append(f"Model dir: {model_dir}")
    else:
        debug_lines.append("Model id: geolocal/StreetCLIP")
    st.code("\n".join(debug_lines), language="text")
    top_k = st.slider("Top-K predictions", min_value=1, max_value=10, value=1)
    apply_jpeg_compression = st.toggle("Apply JPEG compression", value=False)
    jpeg_quality = st.slider("JPEG quality (lower = stronger compression)", min_value=10, max_value=95, value=70)
    apply_shapes = st.toggle("Add random shapes", value=False)
    num_shapes = st.slider("Shape count", min_value=1, max_value=10, value=3)
    shape_scale = st.slider("Shape size", min_value=0.05, max_value=0.4, value=0.18, step=0.01)
    if model_key == "streetclip":
        st.write("StreetCLIP does not output raw GPS coordinates here. It ranks the candidate location labels you provide.")
        streetclip_labels_text = st.text_area(
            "Candidate location labels",
            value=DEFAULT_STREETCLIP_LABELS,
            height=220,
            help="Use one place label per line. StreetCLIP chooses among these candidates only.",
        )
    else:
        st.write("GeoCLIP returns GPS coordinates. This app converts the top result into a human-readable place name when possible.")

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
    st.image(inference_image, caption="GeoCLIP input after compression", use_container_width=True)

if apply_shapes:
    rng = stable_rng(uploaded_file.name, image, "shapes")
    inference_image = add_random_shapes_image(
        inference_image,
        num_shapes=num_shapes,
        max_shape_scale=shape_scale,
        rng=rng,
    )
    st.caption(f"Random shapes preview: {num_shapes} shape(s), size={shape_scale:.2f}")
    st.image(inference_image, caption="GeoCLIP input after shapes", use_container_width=True)

predict_button = st.button("Predict location", type="primary")

if predict_button:
    suffix = ".jpg" if apply_jpeg_compression else ".png"
    fmt = "JPEG" if apply_jpeg_compression else "PNG"
    tmp_path = save_pil_image(inference_image, suffix=suffix, format=fmt)
    try:
        if model_key == "streetclip":
            candidate_labels = parse_candidate_labels(streetclip_labels_text)
            if not candidate_labels:
                st.warning("Add at least one candidate label for StreetCLIP.")
                st.stop()
            effective_top_k = min(top_k, len(candidate_labels))
            with st.spinner("Running StreetCLIP inference..."):
                results = model.predict(tmp_path, labels=candidate_labels, top_k=effective_top_k)
        else:
            with st.spinner("Running GeoCLIP inference..."):
                results = model.predict(tmp_path, top_k=top_k)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if results.empty:
        st.warning(f"{model_family} did not return any predictions.")
        st.stop()

    best = results.iloc[0]
    st.subheader("Best guess")
    if model_key == "streetclip":
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
        best_location = reverse_geocode(float(best["latitude"]), float(best["longitude"]))
        if best_location is None:
            best_location = f'{best["latitude"]:.6f}, {best["longitude"]:.6f}'

        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"**Predicted location**  \n{best_location}")
            st.metric("Latitude", f'{best["latitude"]:.6f}')
            st.metric("Longitude", f'{best["longitude"]:.6f}')
        with right:
            st.metric("Confidence", f'{best["probability"]:.4f}')
            st.map(pd.DataFrame([{"lat": best["latitude"], "lon": best["longitude"]}]))

        st.subheader("Top predictions")
        st.dataframe(
            results[["rank", "latitude", "longitude", "probability"]],
            use_container_width=True,
            hide_index=True,
        )
else:
    st.write(f"Click **Predict location** to run {model_family} on the uploaded image.")
