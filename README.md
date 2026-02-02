# attention-map-diffusers

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18304024.svg)](https://doi.org/10.5281/zenodo.18304024)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/We-Want-GPU/diffusers-cross-attention-map-SDXL-t2i)

Thanks to HuggingFace [Diffusers](https://github.com/huggingface/diffusers) team for the GPU sponsorship!

This repository is for extracting and visualizing cross attention maps, based on the latest [Diffusers](https://github.com/huggingface/diffusers) code (`v0.32.0`).

For errors reports or feature requests, feel free to raise an issue.

## Update Log.
[2024-12-22] It is now compatible with _"Stable Diffusion 3.5"_, _"Flux-dev"_ and _"Flux-schnell"_! (“Sana" will be the focus of the next update.)

[2024-12-17] Refactor and add setup.py

[2024-11-12] _"Stable Diffusion 3"_ is compatible and supports _batch operations_! (Flux and "Stable Diffusion 3.5" is not compatible yet.)

[2024-07-04] Added features for _saving attention maps based on timesteps and layers_.


## Compatible models.
<!-- Compatible with various models, including both UNet/DiT based models listed below. -->
Compatible with various models listed below.
- [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)
- [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [stable-diffusion-2-1-base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base)
- ...

<!-- - [sdxl-turbo](https://huggingface.co/stabilityai/sdxl-turbo) -->
<!-- - [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) -->


## Example.


<div style="text-align: center;">
    <img src="./assets/sd3.png" alt="Image 1" width="400" height="400">
    <img src="./assets/4--bara>.png" alt="Image 2" width="400" height="400">
</div>



<details>
<summary>cap-</summary>
<div markdown="1">

<div style="text-align: center;">
    <img src="./assets/sd3.png" alt="Image 1" width="400" height="400">
    <img src="./assets/2-<cap-.png" alt="<cap-" width="400" height="400">
</div>

</div>
</details>


<details>
<summary>-y-</summary>
<div markdown="1">

<div style="text-align: center;">
    <img src="./assets/sd3.png" alt="Image 1" width="400" height="400">
    <img src="./assets/3--y-.png" alt="-y-" width="400" height="400">
</div>

</div>
</details>


<details>
<summary>-bara</summary>
<div markdown="1">

<div style="text-align: center;">
    <img src="./assets/sd3.png" alt="Image 1" width="400" height="400">
    <img src="./assets/4--bara>.png" alt="-bara>" width="400" height="400">
</div>

</div>
</details>


<details>
<summary>hello</summary>
<div markdown="1">

<div style="text-align: center;">
    <img src="./assets/sd3.png" alt="Image 1" width="400" height="400">
    <img src="./assets/10-<hello>.png" alt="<hello>" width="400" height="400">
</div>

</div>
</details>


<details>
<summary>world</summary>
<div markdown="1">

<div style="text-align: center;">
    <img src="./assets/sd3.png" alt="Image 1" width="400" height="400">
    <img src="./assets/11-<world>.png" alt="<world>>" width="400" height="400">
</div>

</div>
</details>



## demo
```bash
git clone https://github.com/wooyeolBaek/attention-map-diffusers.git
cd attention-map-diffusers
pip install -e .
```
or
```bash
pip install attention_map_diffusers
```

### Flux-dev
```python
import torch
from diffusers import FluxPipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.to('cuda')

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

# recommend not using batch operations for sd3, as cpu memory could be exceeded.
prompts = [
    # "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
    guidance_scale=4.5,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-flux-dev.png')

##### 2. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps-flux-dev', unconditional=False)
#############################################
```

### Flux-schnell
```python
import torch
from diffusers import FluxPipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16
)
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power
pipe.to('cuda')

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

# recommend not using batch operations for sd3, as cpu memory could be exceeded.
prompts = [
    # "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
    guidance_scale=4.5,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-flux-schnell.png')

##### 2. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps-flux-schnell', unconditional=False)
#############################################
```

### Stable Diffusion 3.5
```python
import torch
from diffusers import StableDiffusion3Pipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

# recommend not using batch operations for sd3, as cpu memory could be exceeded.
prompts = [
    # "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
    guidance_scale=4.5,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-sd3-5.png')

##### 2. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps-sd3-5', unconditional=True)
#############################################
```

### Stable Diffusion 3.0
```python
import torch
from diffusers import StableDiffusion3Pipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)


pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

# recommend not using batch operations for sd3, as cpu memory could be exceeded.
prompts = [
    # "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
    guidance_scale=4.5,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-sd3.png')

##### 2. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps', unconditional=True)
#############################################
```

### Stable Diffusion XL
```python
import torch
from diffusers import DiffusionPipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)


pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

prompts = [
    "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-sdxl.png')

##### 2. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps', unconditional=True)
#############################################
```

### Stable Diffusion 2.1
```python
import torch
from diffusers import DiffusionPipeline
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)


pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

prompts = [
    "A photo of a puppy wearing a hat.",
    "A capybara holding a sign that reads Hello World.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
).images

for batch, image in enumerate(images):
    image.save(f'{batch}-sd2-1.png')

##### 2. Process and Save attention map #####
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps', unconditional=True)
#############################################

```

# Citation
If you find this code useful in your research, please consider citing:

```
@misc{baek2024attentionmap,
  author = {Wooyeol Baek},
  title = {attention-map-diffusers},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/wooyeolbaek/attention-map-diffusers](https://github.com/wooyeolbaek/attention-map-diffusers)}},
}
```
