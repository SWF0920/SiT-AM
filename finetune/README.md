# SiT + Adjoint Matching: Reward-Based Fine-tuning

This module provides three approaches to fine-tune **SiT (Scalable Interpolant Transformers)** using **Adjoint Matching** for reward optimization.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Three Fine-tuning Approaches](#three-fine-tuning-approaches)
5. [Reward Functions](#reward-functions)
6. [Detailed Usage](#detailed-usage)
7. [Sanity Testing](#sanity-testing)
8. [Hyperparameter Guide](#hyperparameter-guide)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Overview

### What is Adjoint Matching?

Adjoint Matching (AM) is a method for fine-tuning flow/diffusion models to maximize a reward function. It frames fine-tuning as **Stochastic Optimal Control (SOC)**:

```
Given: Pre-trained velocity model v_base(x, t)
Goal:  Learn v_finetuned(x, t) that maximizes E[r(x_T)]

Key insight: The optimal control is proportional to the "adjoint" —
             the gradient of future reward with respect to current state.
```

### Algorithm Summary

```
1. Sample noise z_0 ~ N(0, I)
2. Generate trajectory z_0 → x_T using current model
3. Compute terminal adjoint: a(T) = ∇r(x_T)
4. Solve adjoint ODE backward: a(T) → a(0)
5. At random time t, compute AM loss:
   L = ||u_θ(x_t, t) - σ(t)² · λ · a(t)||²
6. Update model parameters
```

### Three Approaches

| Approach | File | What's Trained | Params | Use Case |
|----------|------|----------------|--------|----------|
| **Full Model** | `adjoint_matching.py` | All SiT weights | ~675M | Maximum expressivity |
| **Control Net** | `adjoint_matching_control_net.py` | Separate small network | ~2-20M | Fast experiments |
| **LoRA** | `adjoint_matching_lora.py` | Low-rank adapters | ~1-10M | Best balance |

---

## Installation

### 1. Clone SiT Repository

```bash
git clone https://github.com/willisma/SiT.git
cd SiT
conda env create -f environment.yml
conda activate SiT
```

### 2. Add Fine-tuning Module

Copy the `finetune/` folder into SiT:

```
SiT/
├── train.py              # Original (unchanged)
├── models.py             # Original (unchanged)  
├── transport/            # Original (unchanged)
└── finetune/             # ← Add this folder
    ├── __init__.py
    ├── adjoint_matching.py
    ├── adjoint_matching_control_net.py
    ├── adjoint_matching_lora.py
    ├── rewards.py
    ├── finetune_am.py
    ├── tests.py
    └── README.md
```

### 3. Install Additional Dependencies

```bash
# Core (required)
pip install torchdiffeq

# For pixel-space rewards (recommended)
pip install diffusers  # VAE for decoding latents

# For aesthetic/ImageReward (optional)
pip install clip
pip install image-reward
```

### 4. Download Pre-trained Checkpoint

```bash
# SiT-XL/2 (256x256)
wget https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt
```

---

## Quick Start

### Minimal Example (LoRA, recommended)

```python
import torch
from models import SiT_XL_2
from finetune.adjoint_matching_lora import setup_lora_finetuning
from finetune.rewards import QuadraticReward

# 1. Load pre-trained model
model = SiT_XL_2(input_size=32, num_classes=1000)
ckpt = torch.load("SiT-XL-2-256.pt", map_location='cpu')
model.load_state_dict(ckpt['ema'])
model = model.cuda()

# 2. Setup LoRA fine-tuning
reward_fn = QuadraticReward()  # Simple test reward
trainer, optimizer = setup_lora_finetuning(
    model=model,
    reward_fn=reward_fn,
    rank=16,
    lr=1e-4
)

# 3. Train
for step in range(100):
    optimizer.zero_grad()
    loss_dict = trainer.training_step(batch_size=4, latent_size=32)
    loss_dict['loss'].backward()
    optimizer.step()
    
    if step % 10 == 0:
        print(f"Step {step}: loss={loss_dict['loss']:.4f}")

# 4. Save (only ~10MB for LoRA weights)
trainer.save_lora_weights("lora_aesthetic.pt")
```

### Command Line

```bash
# Run sanity test first
python finetune/sanity_test.py --ckpt SiT-XL-2-256.pt --quick

# Full fine-tuning
python finetune/finetune_am.py \
    --ckpt SiT-XL-2-256.pt \
    --reward aesthetic \
    --num-iterations 5000
```

---

## Three Fine-tuning Approaches

### Approach 1: Full Model Fine-tuning

**File:** `adjoint_matching.py`

The entire SiT model is trainable. Control is implicit:
```
u_θ = v_θ - v_base
```

```python
from finetune.adjoint_matching import AdjointMatchingTrainer
import copy

# Load model
model = load_sit_model(...)
base_model = copy.deepcopy(model)
for p in base_model.parameters():
    p.requires_grad = False

# Create trainer
trainer = AdjointMatchingTrainer(
    model=model,              # Trainable
    base_model=base_model,    # Frozen reference
    reward_fn=reward_fn,
    path_type="Linear",
    reward_scale=10.0
)

# Optimize ALL parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for step in range(1000):
    optimizer.zero_grad()
    loss = trainer.training_step(batch_size=4, latent_size=32)
    loss['loss'].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
```

**Pros:** Maximum expressivity  
**Cons:** Slow, high memory, risk of catastrophic forgetting

---

### Approach 2: Control Network

**File:** `adjoint_matching_control_net.py`

A separate small network outputs the control signal:
```
v_finetuned = v_base + ControlNet(x, t)
```

```python
from finetune.adjoint_matching_control_net import (
    AdjointMatchingControlTrainer,
    create_control_network
)

# Load frozen base model
base_model = load_sit_model(...)
for p in base_model.parameters():
    p.requires_grad = False

# Create small control network
control_net = create_control_network(
    arch="conv",           # "mlp", "conv", or "lora"
    latent_channels=4,
    spatial_size=32
).cuda()

# Create trainer
trainer = AdjointMatchingControlTrainer(
    control_net=control_net,  # Trainable (small)
    base_model=base_model,    # Frozen (large)
    reward_fn=reward_fn
)

# Optimize ONLY control network
optimizer = torch.optim.AdamW(control_net.parameters(), lr=1e-4)

for step in range(1000):
    optimizer.zero_grad()
    loss = trainer.training_step(batch_size=8, latent_size=32)
    loss['loss'].backward()
    optimizer.step()
```

**Control Network Options:**

| Architecture | Params | Description |
|--------------|--------|-------------|
| `mlp` | ~2-10M | Flatten → MLP → reshape |
| `conv` | ~5-20M | Shallow U-Net style |
| `lora` | ~0.5-2M | Low-rank modulation |

**Pros:** Fast, low memory  
**Cons:** Limited expressivity, architectural assumptions

---

### Approach 3: LoRA Fine-tuning (Recommended)

**File:** `adjoint_matching_lora.py`

Low-rank adapters injected into SiT's attention layers:
```
W_finetuned = W_frozen + α · (A @ B^T)
```

```python
from finetune.adjoint_matching_lora import (
    inject_lora_into_sit,
    AdjointMatchingLoRATrainer,
    setup_lora_finetuning,
    merge_lora_weights,
    set_lora_scale
)

# Option A: Quick setup
trainer, optimizer = setup_lora_finetuning(
    model=model,
    reward_fn=reward_fn,
    rank=16,
    lr=1e-4
)

# Option B: Manual setup
model, lora_params = inject_lora_into_sit(
    model,
    rank=16,
    alpha=1.0,
    target_modules=['qkv', 'proj', 'fc1', 'fc2']
)
optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

trainer = AdjointMatchingLoRATrainer(
    model=model,
    lora_params=lora_params,
    reward_fn=reward_fn
)

# Training loop
for step in range(1000):
    optimizer.zero_grad()
    loss = trainer.training_step(batch_size=8, latent_size=32)
    loss['loss'].backward()
    optimizer.step()

# Save tiny checkpoint (~10MB)
trainer.save_lora_weights("lora_weights.pt")

# Interpolate between base and fine-tuned
set_lora_scale(model, 0.5)  # 50% blend

# Merge for inference (no overhead)
model = merge_lora_weights(model)
```

**Pros:** Good expressivity, efficient, tiny checkpoints, can interpolate  
**Cons:** Slightly more complex setup

---

## Reward Functions

### Available Rewards

| Reward | Description | Space | File |
|--------|-------------|-------|------|
| `QuadraticReward` | `-\|\|x\|\|²` (for testing) | Latent | `rewards.py` |
| `BrightnessReward` | Target brightness | Pixel | `rewards.py` |
| `AestheticReward` | LAION aesthetic score | Pixel | `rewards.py` |
| `ImageRewardScorer` | Text-image alignment | Pixel | `rewards.py` |
| `CompositeReward` | Weighted combination | Any | `rewards.py` |

### Using Pixel-Space Rewards

Pixel-space rewards require VAE decoding. Wrap them with `LatentSpaceReward`:

```python
from finetune.rewards import AestheticReward, LatentSpaceReward
from diffusers.models import AutoencoderKL

# Load VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").cuda()
vae.eval()

# Wrap pixel reward
pixel_reward = AestheticReward(device='cuda')
reward_fn = LatentSpaceReward(pixel_reward, vae)

# Now reward_fn works on latents
```

### Custom Reward Functions

```python
import torch.nn as nn

class MyCustomReward(nn.Module):
    """Custom reward: must return [B] tensor from [B,C,H,W] input."""
    
    def __init__(self):
        super().__init__()
        # Load any models you need
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] images (latent or pixel space)
        Returns:
            rewards: [B] scalar rewards (higher = better)
        """
        # Your reward computation
        # MUST be differentiable for adjoint computation!
        return rewards
```

### Combining Multiple Rewards

```python
from finetune.rewards import CompositeReward, AestheticReward, BrightnessReward

reward_fn = CompositeReward([
    (AestheticReward(), 1.0),      # Weight 1.0
    (BrightnessReward(), 0.5),    # Weight 0.5
])
```

---

## Detailed Usage

### Full Training Script

```python
import torch
import copy
from models import SiT_XL_2
from finetune.adjoint_matching_lora import (
    inject_lora_into_sit,
    AdjointMatchingLoRATrainer
)
from finetune.rewards import AestheticReward, LatentSpaceReward
from diffusers.models import AutoencoderKL
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================
CHECKPOINT = "SiT-XL-2-256.pt"
RANK = 16
LR = 1e-4
REWARD_SCALE = 10.0
CFG_SCALE = 4.0
BATCH_SIZE = 4
NUM_ITERATIONS = 5000
SAVE_EVERY = 1000

# ============================================================
# Load Model
# ============================================================
print("Loading model...")
model = SiT_XL_2(input_size=32, num_classes=1000)
ckpt = torch.load(CHECKPOINT, map_location='cpu')
model.load_state_dict(ckpt['ema'])
model = model.cuda()

# ============================================================
# Inject LoRA
# ============================================================
print("Injecting LoRA...")
model, lora_params = inject_lora_into_sit(model, rank=RANK)

# ============================================================
# Setup Reward
# ============================================================
print("Setting up reward...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").cuda()
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

pixel_reward = AestheticReward(device='cuda')
reward_fn = LatentSpaceReward(pixel_reward, vae)

# ============================================================
# Create Trainer
# ============================================================
trainer = AdjointMatchingLoRATrainer(
    model=model,
    lora_params=lora_params,
    reward_fn=reward_fn,
    path_type="Linear",
    reward_scale=REWARD_SCALE,
    cfg_scale=CFG_SCALE
)

optimizer = torch.optim.AdamW(lora_params, lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=NUM_ITERATIONS
)

# ============================================================
# Training Loop
# ============================================================
print(f"Training for {NUM_ITERATIONS} iterations...")

for step in tqdm(range(1, NUM_ITERATIONS + 1)):
    optimizer.zero_grad()
    
    loss_dict = trainer.training_step(
        batch_size=BATCH_SIZE,
        latent_size=32
    )
    
    loss_dict['loss'].backward()
    torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
    optimizer.step()
    scheduler.step()
    
    if step % 100 == 0:
        print(f"Step {step}: loss={loss_dict['loss']:.4f}, "
              f"ctrl={loss_dict['control_norm']:.4f}")
    
    if step % SAVE_EVERY == 0:
        trainer.save_lora_weights(f"lora_step{step}.pt")

# ============================================================
# Final Save
# ============================================================
trainer.save_lora_weights("lora_final.pt")
print("Done!")
```

### Inference After Fine-tuning

```python
from finetune.adjoint_matching_lora import set_lora_scale, merge_lora_weights

# Option 1: Keep LoRA separate (can interpolate)
set_lora_scale(model, 1.0)  # Full fine-tuned
set_lora_scale(model, 0.5)  # 50% blend
set_lora_scale(model, 0.0)  # Base model

# Option 2: Merge for inference (no overhead)
model = merge_lora_weights(model)

# Then sample normally using SiT's sample.py
```

---

## Sanity Testing

Before full training, run the sanity test:

```bash
# Quick test (64 samples, ~5 min)
python finetune/sanity_test.py \
    --ckpt SiT-XL-2-256.pt \
    --quick

# Full test (256 samples)
python finetune/sanity_test.py \
    --ckpt SiT-XL-2-256.pt \
    --num-eval-samples 256

# With hyperparameter sweep
python finetune/sanity_test.py \
    --ckpt SiT-XL-2-256.pt \
    --sweep
```

### Expected Results

| Experiment | Quadratic Reward | What to Check |
|------------|------------------|---------------|
| Baseline | ~ -8.5 | Reference |
| 1-step AM | ~ -8.4 | Small Δ (gradient flows) |
| 100-step AM | ~ -6.5 | Clear improvement |

**Red flags:**
- 1-step shows zero change → gradient not flowing
- 100-step shows huge jump → unstable, check LR
- Loss goes NaN → reduce LR, increase grad clip

---

## Hyperparameter Guide

### Critical Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `lr` | 1e-5 (full), 1e-4 (LoRA) | 1e-6 to 1e-3 | Higher = faster but unstable |
| `reward_scale` | 10.0 | 1.0 to 100.0 | Higher = stronger reward signal |
| `rank` (LoRA) | 16 | 4 to 64 | Higher = more expressive |
| `cfg_scale` | 4.0 | 1.0 to 10.0 | Match your sampling CFG |
| `num_sampling_steps` | 50 | 20 to 100 | More = accurate but slow |
| `eta` | 1.0 | ≥ 1.0 | Must be ≥1 for AM theory |

### Recommended Settings by Approach

| Setting | Full Model | Control Net | LoRA |
|---------|------------|-------------|------|
| LR | 1e-5 | 1e-4 | 1e-4 |
| Batch size | 2-4 | 8-16 | 4-8 |
| Grad clip | 1.0 | 1.0 | 1.0 |
| Iterations | 5000+ | 2000+ | 3000+ |

### Tuning Tips

1. **Start with QuadraticReward** — fast, no VAE, easy to verify
2. **Watch the loss curve** — should decrease steadily
3. **Monitor control_norm** — if it explodes, reduce reward_scale
4. **Check samples periodically** — reward hacking is possible

---

## Troubleshooting

### Loss is NaN
```python
# Reduce learning rate
optimizer = AdamW(params, lr=1e-6)

# Increase gradient clipping
torch.nn.utils.clip_grad_norm_(params, 0.5)

# Check reward function outputs
rewards = reward_fn(samples)
print(f"Reward range: [{rewards.min():.2f}, {rewards.max():.2f}]")
```

### No improvement after training
```python
# Check if gradients flow
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm():.4f}")

# Increase reward_scale
trainer = AdjointMatchingLoRATrainer(..., reward_scale=50.0)
```

### Out of memory
```python
# Reduce batch size
loss = trainer.training_step(batch_size=2)

# Use gradient checkpointing (if available)
model.gradient_checkpointing_enable()

# Use LoRA instead of full model
```

### Samples look worse
```python
# Reduce LoRA contribution
set_lora_scale(model, 0.3)  # Partial blend

# Check for reward hacking — add diversity regularization
# Or use lower reward_scale
```

---

## File Structure

```
finetune/
├── __init__.py                      # Module exports
├── adjoint_matching.py              # Full model fine-tuning
├── adjoint_matching_control_net.py  # Control network approach
├── adjoint_matching_lora.py         # LoRA approach (recommended)
├── rewards.py                       # Reward function library
├── finetune_am.py                   # CLI training script
├── sanity_test.py                   # Sanity test script
└── README.md                        # This file
```

---

## References

### Papers

- **Adjoint Matching**: Domingo-Enrich et al., "Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models with Memoryless Stochastic Optimal Control", ICLR 2025. [arXiv:2409.08861](https://arxiv.org/abs/2409.08861)

- **SiT**: Ma et al., "SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers", 2024. [arXiv:2401.08740](https://arxiv.org/abs/2401.08740)

- **LoRA**: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)

### Code

- **Official AM Code**: [microsoft/soc-fine-tuning-sd](https://github.com/microsoft/soc-fine-tuning-sd)
- **SiT Repository**: [willisma/SiT](https://github.com/willisma/SiT)

---

## Citation

```bibtex
@inproceedings{domingo-enrich2025adjoint,
    title={Adjoint Matching: Fine-tuning Flow and Diffusion Generative Models 
           with Memoryless Stochastic Optimal Control},
    author={Domingo-Enrich, Carles and Drozdzal, Michal and Karrer, Brian 
            and Chen, Ricky T. Q.},
    booktitle={ICLR},
    year={2025}
}

@article{ma2024sit,
    title={SiT: Exploring Flow and Diffusion-based Generative Models 
           with Scalable Interpolant Transformers},
    author={Ma, Nanye and Goldstein, Mark and Albergo, Michael S. 
            and Boffi, Nicholas M. and Vanden-Eijnden, Eric and Xie, Saining},
    journal={arXiv preprint arXiv:2401.08740},
    year={2024}
}

@inproceedings{hu2022lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and 
            Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and 
            Wang, Lu and Chen, Weizhu},
    booktitle={ICLR},
    year={2022}
}
```

---

## License

MIT License (following SiT repository)