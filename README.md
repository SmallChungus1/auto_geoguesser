# GeoCLIP Streamlit Demo

Minimal Streamlit app for running GeoCLIP on an uploaded image and converting the top GPS prediction into a readable location.

## What it does

- Upload an image
- Run GeoCLIP inference
- Show the top GPS predictions
- Reverse-geocode the best coordinate into a human-readable place name when possible
- Optionally run fully offline from a local Hugging Face snapshot
- Optionally overlay a small random country flag before inference

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Offline mode

Download the model snapshot locally:

```bash
python scripts/download_geoclip_weights.py --output-dir models/geoclip-large-patch14
```

Then run the app in offline mode:

```bash
export GEOCLIP_MODE=offline
export GEOCLIP_MODEL_DIR=models/geoclip-large-patch14
streamlit run app.py
```

You can also leave `GEOCLIP_MODE=auto` and the app will use the local snapshot when `GEOCLIP_MODEL_DIR` exists, otherwise it falls back to the packaged `geoclip` backend.

## Notes

- GeoCLIP predicts GPS coordinates, not a city name directly.
- The reverse-geocoding step uses OpenStreetMap Nominatim through `geopy`.
- If reverse geocoding fails or is rate-limited, the app falls back to showing raw coordinates.
- The offline path uses `onnxruntime` and a local copy of `Xenova/geoclip-large-patch14`.
- The flag overlay uses SVG flags from `country_flags/` and rasterizes them locally with `resvg_py`.
