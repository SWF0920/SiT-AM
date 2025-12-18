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
        
    def _lazy_load(self):
        """Load model on first use."""
        if self._loaded:
            return
            
        try:
            import ImageReward as IR
            self.model = IR.load("ImageReward-v1.0")
            self.model.to(self.device)
            self.model.eval()
        except ImportError:
            print("Warning: ImageReward not available. Using dummy scorer.")
            self.model = None
            
        self._loaded = True
    
    def _get_class_name(self, class_idx: int) -> str:
        """Convert ImageNet class index to human-readable name."""
        if self._class_names is None:
            # Load ImageNet class names
            try:
                import json
                import urllib.request
                url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
                with urllib.request.urlopen(url) as response:
                    self._class_names = json.loads(response.read().decode())
            except:
                # Fallback: generic labels
                self._class_names = [f"class_{i}" for i in range(1000)]
        
        return self._class_names[class_idx]
    
    def forward(
        self, 
        images: torch.Tensor,
        prompts: Optional[Union[str, List[str]]] = None,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute ImageReward scores.
        
        Args:
            images: [B, C, H, W] tensor
            prompts: Text prompts (single string or list)
            class_labels: [B] ImageNet class indices (alternative to prompts)
            
        Returns:
            scores: [B] alignment scores
        """
        self._lazy_load()
        
        batch_size = images.shape[0]
        
        # Generate prompts from class labels if needed
        if prompts is None and class_labels is not None:
            prompts = [
                f"a photo of a {self._get_class_name(c.item())}"
                for c in class_labels
            ]
        elif prompts is None:
            prompts = ["a high quality image"] * batch_size
        elif isinstance(prompts, str):
            prompts = [prompts] * batch_size
        
        if self.model is None:
            # Dummy scorer
            return torch.zeros(batch_size, device=images.device)
        
        # Normalize images to [0, 1]
        if images.min() < 0:
            images = (images + 1) / 2
        
        # Convert to PIL and score
        scores = []
        for i in range(batch_size):
            img = images[i].cpu()
            # Convert to PIL
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            from PIL import Image
            img_pil = Image.fromarray(img_np)
            
            # Score
            with torch.no_grad():
                score = self.model.score(prompts[i], img_pil)
            scores.append(score)
        
        return torch.tensor(scores, device=images.device, dtype=images.dtype)


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
    
    def forward(self, latents: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Decode latents and compute reward.
        
        Args:
            latents: [B, 4, H, W] VAE latents
            
        Returns:
            rewards: [B] reward scores
        """
        # Decode to pixel space
        with torch.no_grad():
            # Scale latents
            latents_scaled = latents / self.scale_factor
            # Decode
            images = self.vae.decode(latents_scaled).sample
        
        # Apply pixel-space reward
        return self.pixel_reward(images, **kwargs)