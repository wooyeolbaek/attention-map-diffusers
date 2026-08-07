---
title: SDXL Attention Explorer
emoji: 🔎
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.22.0
python_version: "3.11"
app_file: app.py
pinned: false
license: mit
suggested_hardware: t4-medium
models:
  - stabilityai/stable-diffusion-xl-base-1.0
---

# SDXL Attention Explorer

Generate an SDXL image and inspect the two attention relations exposed by its
denoiser:

- `image→text`: one image-shaped cross-attention overlay per active prompt token
- `image→image`: spatial self-attention from the center query patch

The demo uses `attention-map-diffusers==1.0.0` instead of copying or monkey
patching Diffusers internals. It captures only the final denoising timestep,
averages attention heads, keeps Q/K tensors on CPU, and limits self-attention to
one query patch so it fits the Space's T4 GPU.

The default `MODEL_VARIANT=fp16` downloads the smaller half-precision SDXL
checkpoint. Set the Space variable `MODEL_ID` only when intentionally replacing
the checkpoint.

Raw `.pt` tensors are not written. Returned images use Gradio's temporary cache
instead of a persistent result directory.

## Local development

Publish `attention-map-diffusers==1.0.0` first, then run:

```bash
python -m pip install -r requirements.txt
python app.py
```

Project: <https://github.com/wooyeolbaek/attention-map-diffusers>
