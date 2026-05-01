# StreetCLIP City-Scene Geolocation

This repo is centered on `StreetCLIP` for image-to-location-label matching in city and street-scene settings.

## What the app does

- Upload an image
- Provide candidate `{city, country}` labels
- Run `StreetCLIP` and rank the candidate labels by image-text similarity
- Optionally apply JPEG compression before inference
- Optionally add random opaque shapes before inference
- Optionally load a local fine-tuned checkpoint instead of the baseline `geolocal/StreetCLIP` model

The Streamlit app does not do open-world coordinate prediction. It evaluates a fixed candidate-label set.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

By default the app loads:

```text
geolocal/StreetCLIP
```

You can also point the sidebar model field to a local fine-tuned checkpoint folder that contains files such as:

- `config.json`
- `model.safetensors`
- `preprocessor_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.json`

## Project workflow

### Training

Training lives in:

- [notebooks/osv5m_streetclip_train_colab.ipynb](/Users/hansonli/Desktop/auto_geoguesser/notebooks/osv5m_streetclip_train_colab.ipynb)

The notebook fine-tunes two models on an OSV-5M subset:

- `streetclip_clean`
- `streetclip_jpeg_aug`

Important details:

- the training subset is built from `OpenStreetView-5M`
- labels are reduced to `{city, country}` text
- training runs directly from the extracted image cache
- the JPEG-augmented run re-encodes images with randomized JPEG quality during training
- checkpoints are saved to a user-defined Google Drive path

### Evaluation

Evaluation lives in:

- [notebooks/osv5m_streetclip_eval_colab.ipynb](/Users/hansonli/Desktop/auto_geoguesser/notebooks/osv5m_streetclip_eval_colab.ipynb)
- You'll need the 2 finetuned model weights for the eval. [Aug model here weights here](https://drive.google.com/drive/folders/1yQynYrrLEO7dBWPs9w60Whhjc6eMCjJO?usp=share_link) | [Clean model weights here](https://drive.google.com/drive/folders/1emPBGvwp1O0JkNBgG2jrcbQkQcvQOhB7?usp=share_link)

Evaluation protocol:

- sample a held-out OSV-5M subset
- convert each ground-truth label to `{city, country}`
- construct a 10-way candidate set:
  - 1 ground-truth label
  - 9 sampled distractor labels
- run StreetCLIP on:
  - `clean`
  - `jpeg-q90`
  - `jpeg-q70`
  - `jpeg-q50`
  - `jpeg-q30`

Reported metrics:

- `Top-1 accuracy`
- `Top-5 accuracy`
- `MRR`
- centroid-based geographic error as a secondary proxy metric

## Main results

These are the older result numbers currently used in the report.

### Accuracy table

| Model | Condition | Top-1 | Top-5 |
| --- | --- | ---: | ---: |
| Baseline | Clean | 0.3460 | 0.7970 |
| Baseline | JPEG q=90 | 0.3450 | 0.7965 |
| Baseline | JPEG q=70 | 0.3370 | 0.7985 |
| Baseline | JPEG q=50 | 0.3255 | 0.7865 |
| Baseline | JPEG q=30 | 0.3240 | 0.7845 |
| Fine-tuned (clean) | Clean | 0.3490 | 0.8170 |
| Fine-tuned (clean) | JPEG q=90 | 0.3520 | 0.8145 |
| Fine-tuned (clean) | JPEG q=70 | 0.3515 | 0.8160 |
| Fine-tuned (clean) | JPEG q=50 | 0.3485 | 0.8050 |
| Fine-tuned (clean) | JPEG q=30 | 0.3405 | 0.8065 |
| Fine-tuned (JPEG aug.) | Clean | 0.3440 | 0.8205 |
| Fine-tuned (JPEG aug.) | JPEG q=90 | 0.3405 | 0.8190 |
| Fine-tuned (JPEG aug.) | JPEG q=70 | 0.3445 | 0.8195 |
| Fine-tuned (JPEG aug.) | JPEG q=50 | 0.3435 | 0.8090 |
| Fine-tuned (JPEG aug.) | JPEG q=30 | 0.3340 | 0.8085 |

### Headline summary

- The clean fine-tuned model is best on `Top-1` across all JPEG conditions.
- The JPEG-augmented fine-tuned model is best on `Top-5` across all JPEG conditions.
- Fine-tuning improves over the baseline on the older evaluation results used in the report.

Average across all five conditions:

- Baseline:
  - `Top-1 = 0.3355`
  - `Top-5 = 0.7926`
- Fine-tuned (clean):
  - `Top-1 = 0.3483`
  - `Top-5 = 0.8118`
  - gain vs baseline:
    - `Top-1: +0.0128`
    - `Top-5: +0.0192`
- Fine-tuned (JPEG aug.):
  - `Top-1 = 0.3413`
  - `Top-5 = 0.8153`
  - gain vs baseline:
    - `Top-1: +0.0058`
    - `Top-5: +0.0227`

Interpretation:

- clean fine-tuning gives the stronger `Top-1` gain
- JPEG augmentation gives the stronger `Top-5` gain
- JPEG augmentation appears to help broader ranking robustness more than final top-label selection

## Notes

- StreetCLIP does not generate captions here; it is used as an image-to-label matching model.
- Candidate-label choice matters. This setup is a 10-way label-ranking evaluation, not unrestricted world-scale retrieval.
- `JPEG q` refers to the JPEG quality factor. Lower `q` means stronger compression.
- AI Usage: Codex was used to help debug and develop the training/eval pipelines as well as the streamlit app. 
