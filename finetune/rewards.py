"""
Reward Functions for SiT Fine-tuning
=====================================

Provides reward functions for Adjoint Matching fine-tuning:
- AestheticReward: LAION aesthetic predictor
- ImageRewardScorer: Text-image alignment score
- CompositeReward: Combine multiple rewards

Usage:
    from finetune.rewards import AestheticReward, ImageRewardScorer
    
    reward_fn = AestheticReward(device='cuda')
    scores = reward_fn(images)  # [B] tensor of scores
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable, Union
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms


# ==============================================================================
# Aesthetic Reward (LAION-style)
# ==============================================================================
class AestheticReward(nn.Module):
    """
    Aesthetic score predictor using CLIP + MLP.
    
    Based on LAION aesthetic predictor:
    https://github.com/LAION-AI/aesthetic-predictor
    
    Returns scores in [0, 10] range, higher = more aesthetic.
    """
    
    def __init__(
        self,
        clip_model: str = "ViT-L/14",
        predictor_path: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self._loaded = False
        self.clip_model_name = clip_model
        self.predictor_path = predictor_path
        
    def _lazy_load(self):
        """Load models on first use."""
        if self._loaded:
            return
            
        try:
            import clip
            self.clip_model, self.preprocess = clip.load(
                self.clip_model_name, device=self.device
            )
            self.clip_model.eval()
            for p in self.clip_model.parameters():
                p.requires_grad = False
        except ImportError:
            print("Warning: CLIP not available. Using dummy aesthetic scorer.")
            self.clip_model = None
            self._loaded = True
            return
        
        # MLP predictor (LAION aesthetic predictor architecture)
        self.mlp = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        ).to(self.device)
        
        # Load pretrained weights if available
        if self.predictor_path:
            state_dict = torch.load(self.predictor_path, map_location=self.device)
            self.mlp.load_state_dict(state_dict)
        
        self.mlp.eval()
        for p in self.mlp.parameters():
            p.requires_grad = False
            
        self._loaded = True
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute aesthetic scores.
        
        Args:
            images: [B, C, H, W] tensor, values in [-1, 1] or [0, 1]
            
        Returns:
            scores: [B] aesthetic scores
        """
        self._lazy_load()
        
        if self.clip_model is None:
            # Dummy scorer: prefer images with moderate variance
            return -torch.var(images.view(images.shape[0], -1), dim=1)
        
        # Normalize to [0, 1] if needed
        if images.min() < 0:
            images = (images + 1) / 2
        
        # Resize to CLIP input size (224x224)
        images = torch.nn.functional.interpolate(
            images, size=(224, 224), mode='bilinear', align_corners=False
        )
        
        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], 
                           device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                          device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std
        
        # Get CLIP embeddings
        with torch.no_grad():
            embeddings = self.clip_model.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # Predict aesthetic score
        scores = self.mlp(embeddings.float()).squeeze(-1)
        
        return scores


# ==============================================================================
# ImageReward (Text-Image Alignment)
# ==============================================================================
class ImageRewardScorer(nn.Module):
    """
    ImageReward scorer for text-image alignment.
    
    Reference: https://github.com/THUDM/ImageReward
    
    For class-conditional generation, converts class labels to text prompts.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = device
        self._loaded = False
        self.model_path = model_path
        
        # ImageNet class names for class-conditional generation
        self._class_names = None
        self.use_class_prompts = True
        
    def _lazy_load(self):
        if self._loaded:
            return
        try:
            import ImageReward as IR
        except ImportError as e:
            raise ImportError(
                "ImageReward is not installed in this env. "
                "AM needs a differentiable reward; dummy scorer cannot work.\n"
                "Install inside the SAME conda env you run torchrun from:\n"
                "  pip install -U image-reward\n"
            ) from e

        self.model = IR.load("ImageReward-v1.0")  # or IR.load(..., device=self.device) if supported
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._loaded = True

    def _load_class_names_local(self):
        import os, json
        if self._class_names is not None:
            return
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "imagenet_labels.json")
        with open(path, "r") as f:
            self._class_names = json.load(f)
        if not (isinstance(self._class_names, list) and len(self._class_names) >= 1000):
            raise RuntimeError(f"Bad ImageNet labels at {path}")

    def _get_class_name(self, class_idx: int) -> str:
        self._load_class_names_local()
        return self._class_names[int(class_idx)]
    
    '''
    def _get_class_name(self, class_idx: int) -> str:
        """Convert ImageNet class index to human-readable name."""
        if self._class_names is None:
            # Load ImageNet class names
            # try:
                # import json
                # import urllib.request
                # url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
                # with urllib.request.urlopen(url) as response:
                    # self._class_names = json.loads(response.read().decode())
            try:
                from torchvision.models import get_model_weights
                self._class_names = get_model_weights("resnet50").meta["categories"]
            except:
                # Fallback: generic labels
                self._class_names = [f"class_{i}" for i in range(1000)]
        
        return self._class_names[class_idx]
    '''
    
    def forward(
        self,
        images: torch.Tensor,                     # [B,3,H,W] in [-1,1] or [0,1]
        prompts=None,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # print("IN scorer: images.requires_grad =", images.requires_grad, "shape=", tuple(images.shape))

        self._lazy_load()
        rm = self.model  # ImageReward object from IR.load

        B = images.shape[0]

        # prompts
        if (prompts is None and class_labels is not None) and self.use_class_prompts:
            prompts = [f"a photo of a {self._get_class_name(int(c))}" for c in class_labels]
        elif prompts is None:
            prompts = ["a high quality image"] * B
        elif isinstance(prompts, str):
            prompts = [prompts] * B

        print("The prompt is", prompts)

        # ---- keep grads: NO .cpu(), NO numpy(), NO PIL, NO torch.no_grad() ----
        x = images
        if x.min() < 0:
            x = (x + 1) / 2
        x = x.clamp(0, 1)

        # move to reward device, float32 for stability
        x = x.to(self.device, dtype=torch.float32)

        # ImageReward/BLIP typically uses 224
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        # CLIP-style normalization (commonly used by ImageReward pipelines)
        clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1,3,1,1)
        clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1,3,1,1)
        x = (x - clip_mean) / clip_std

        # Locate BLIP + tokenizer + heads (common ImageReward layout)
        blip = getattr(rm, "blip", None)
        mlp  = getattr(rm, "mlp", None)
        if blip is None or mlp is None:
            raise RuntimeError(
                f"Unexpected ImageReward model layout: has blip={blip is not None}, mlp={mlp is not None}. "
                "Print dir(rm) and adapt attribute names."
            )

        tokenizer = getattr(blip, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Cannot find blip.tokenizer on the loaded ImageReward model.")

        tok = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        tok = {k: v.to(x.device) for k, v in tok.items()}

        # Vision encoder
        image_embeds = blip.visual_encoder(x)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=x.device)

        # Text encoder with cross-attn to image embeds
        text_out = blip.text_encoder(
            tok["input_ids"],
            attention_mask=tok["attention_mask"],
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        feat = text_out.last_hidden_state[:, 0, :].float()  # [B, hidden]
        reward = mlp(feat).squeeze(-1)                       # [B]

        # optional normalization if present
        if hasattr(rm, "mean") and hasattr(rm, "std"):
            reward = (reward - rm.mean) / rm.std

        # print("grad_enabled:", torch.is_grad_enabled(),
            # "inference_mode:", torch.is_inference_mode_enabled(),
            # "reward.requires_grad:", reward.requires_grad)

        # return on caller device/dtype
        return reward.to(device=images.device, dtype=images.dtype)



# ==============================================================================
# Simple Reward Functions (for testing)
# ==============================================================================
class QuadraticReward(nn.Module):
    """
    Simple quadratic reward: r(x) = -||x - target||²
    
    Useful for testing AM implementation.
    
    
    def __init__(self, target: Optional[torch.Tensor] = None):
        super().__init__()
        self.target = target  # If None, target is zero
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute quadratic reward.
        if self.target is not None:
            diff = x - self.target.to(x.device)
        else:
            diff = x
        
        # Sum over all dims except batch
        return -torch.sum(diff ** 2, dim=tuple(range(1, x.dim())))
    """

    """
    Reward r(x) = -mean(||x||^2)
    
    This is a stable, consistent reward for testing AM.
    Encourages contraction toward zero without requiring 
    a per-example target.
    """
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean squared value per sample
        # Shape: x = [B, C, H, W]
        mse = (x.pow(2)).mean(dim=tuple(range(1, x.dim())))  # [B]
        
        # Reward is negative MSE
        return -self.scale * mse

class LinearReward(nn.Module):
    """
    Simple linear reward: r(x) = <w, x>  (per-sample dot product)

    Useful for testing AM implementation because:
    - grad_x r(x) = w (constant), so the adjoint terminal condition is easy to verify.
    - If you set the backbone dynamics b(x,t)=0 (or near), the adjoint should remain ~constant.
    """

    def __init__(
        self,
        w: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        normalize_w: bool = True,
        detach_w: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.normalize_w = normalize_w
        self.detach_w = detach_w

        if w is None:
            # If w is not provided, we create it lazily on first forward
            self.register_buffer("w", None, persistent=False)
        else:
            w = w.float()
            if normalize_w:
                w = w / (w.norm().clamp_min(1e-12))
            # buffer is enough (no training), but you can switch to nn.Parameter if you want to learn w
            self.register_buffer("w", w)

    def _init_w_like(self, x: torch.Tensor) -> torch.Tensor:
        # Create a deterministic w for reproducibility (you can change this to randn if desired).
        # Shape matches x[0] (no batch dim)
        w = torch.ones_like(x[0]).float()
        if self.normalize_w:
            w = w / (w.norm().clamp_min(1e-12))
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, ...] (e.g. [B,C,H,W] or latent [B,D])
        returns: [B]
        """

        # print("LinearReward got x.shape =", tuple(x.shape), "dtype=", x.dtype, "device=", x.device)

        if self.w is None:
            # Lazy init on the right shape/device
            w = self._init_w_like(x).to(x.device)
        else:
            w = self.w.to(x.device)
            # If x has different spatial size than w (common in tests), you should pass w explicitly
            if w.shape != x.shape[1:]:
                raise ValueError(
                    f"LinearReward: w.shape={tuple(w.shape)} must match x.shape[1:]={tuple(x.shape[1:])}. "
                    "Pass a correctly-shaped w."
                )

        if self.detach_w:
            w = w.detach()

        # dot per sample
        r = (x * w.unsqueeze(0)).sum(dim=tuple(range(1, x.dim())))  # [B]
        return self.scale * r


class BrightnessReward(nn.Module):
    """
    Reward for image brightness.
    
    Useful for testing: encourages brighter/darker images.
    """
    
    def __init__(self, target_brightness: float = 0.5, invert: bool = False):
        super().__init__()
        self.target = target_brightness
        self.invert = invert
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute brightness reward."""
        # Normalize to [0, 1]
        if x.min() < 0:
            x = (x + 1) / 2
        
        # Mean brightness per image
        brightness = x.mean(dim=(1, 2, 3))
        
        # Reward: negative distance from target
        reward = -((brightness - self.target) ** 2)
        
        if self.invert:
            reward = -reward
            
        return reward


# ==============================================================================
# Composite Reward
# ==============================================================================
class CompositeReward(nn.Module):
    """
    Combine multiple reward functions with weights.
    
    Usage:
        reward = CompositeReward([
            (AestheticReward(), 1.0),
            (ImageRewardScorer(), 0.5)
        ])
    """
    
    def __init__(
        self, 
        rewards: List[tuple],  # [(reward_fn, weight), ...]
    ):
        super().__init__()
        self.rewards = nn.ModuleList([r for r, w in rewards])
        self.weights = [w for r, w in rewards]
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute weighted sum of rewards."""
        total = torch.zeros(x.shape[0], device=x.device)
        
        for reward_fn, weight in zip(self.rewards, self.weights):
            total = total + weight * reward_fn(x, **kwargs)
        
        return total


# ==============================================================================
# Latent Space Rewards (for VAE latent space)
# ==============================================================================
class LatentSpaceReward(nn.Module):
    """
    Wrapper to apply pixel-space reward to VAE latents.
    
    Decodes latents using VAE, then applies reward function.
    """
    
    def __init__(
        self,
        pixel_reward: nn.Module,
        vae: nn.Module,
        scale_factor: float = 0.18215  # SD VAE scaling
    ):
        super().__init__()
        self.pixel_reward = pixel_reward
        self.vae = vae
        self.scale_factor = scale_factor
        
        # Freeze VAE
        for p in self.vae.parameters():
            p.requires_grad = False
        self.vae.eval()
    
    def forward(self, images: torch.Tensor, **kwargs):
    # images can be:
    #   latents: [B,4,32,32]
    #   pixels:  [B,3,256,256]  (already decoded somewhere else)

        if images.dim() == 4 and images.shape[1] == 4:
            latents = images  # DO NOT clamp / do NOT detach

            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad_(False)

            # diffusers VAE expects latents / scaling_factor
            if hasattr(self.vae, "config") and hasattr(self.vae.config, "scaling_factor"):
                latents = latents / self.vae.config.scaling_factor

            pixels = self.vae.decode(latents).sample
            pixels = (pixels / 2 + 0.5).clamp(0, 1)
        else:
            # already pixels
            pixels = images
            if pixels.min() < 0:
                pixels = (pixels + 1) / 2
            pixels = pixels.clamp(0, 1)

        return self.pixel_reward(pixels, **kwargs)