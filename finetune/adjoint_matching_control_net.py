"""
Adjoint Matching with Explicit Control Network
===============================================

This is an ALTERNATIVE design to adjoint_matching.py

Key difference:
- adjoint_matching.py: u_θ = v_θ - v_base (implicit, full model trainable)
- This file: u_θ = ControlNet(x, t) (explicit small network)

The fine-tuned velocity is:
    v_finetuned(x, t) = v_base(x, t) + u_θ(x, t)

Where:
- v_base: Frozen pre-trained SiT (NEVER updated)
- u_θ: Small trainable control network (ONLY this is updated)

Usage:
    from finetune.adjoint_matching_control_net import (
        ControlNetwork,
        AdjointMatchingControlTrainer
    )
    
    control_net = ControlNetwork(latent_dim=4, spatial_size=32)
    trainer = AdjointMatchingControlTrainer(
        control_net=control_net,
        base_model=base_model,
        reward_fn=reward_fn
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Dict, Tuple
import math


# ==============================================================================
# Control Network Architectures
# ==============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time embeddings (same as diffusion models)."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class ControlNetworkMLP(nn.Module):
    """
    Simple MLP control network.
    
    Flattens spatial dimensions, concatenates time embedding,
    and outputs control signal.
    
    Parameters: ~2-10M depending on hidden_dim
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        spatial_size: int = 32,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        time_embed_dim: int = 256
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.flat_dim = latent_channels * spatial_size * spatial_size
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Main network
        layers = []
        in_dim = self.flat_dim + time_embed_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else self.flat_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.SiLU() if i < num_layers - 1 else nn.Identity()
            ])
            in_dim = hidden_dim
        
        self.net = nn.Sequential(*layers)
        
        # Initialize last layer to zero (start with u_θ = 0)
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None  # Ignored, for API compatibility
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] latent state
            t: [B] time values
            y: [B] class labels (ignored in this simple version)
        
        Returns:
            control: [B, C, H, W] control signal u_θ(x, t)
        """
        batch_size = x.shape[0]
        
        # Flatten spatial dims
        x_flat = x.view(batch_size, -1)  # [B, C*H*W]
        
        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_embed_dim]
        
        # Concatenate and forward
        h = torch.cat([x_flat, t_emb], dim=-1)
        out = self.net(h)
        
        # Reshape to spatial
        control = out.view(batch_size, self.latent_channels, 
                          self.spatial_size, self.spatial_size)
        
        return control


class ControlNetworkConv(nn.Module):
    """
    Convolutional control network (U-Net style, but shallow).
    
    Preserves spatial structure, more expressive than MLP.
    
    Parameters: ~5-20M depending on base_channels
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        base_channels: int = 64,
        time_embed_dim: int = 256
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Encoder
        self.conv_in = nn.Conv2d(latent_channels, base_channels, 3, padding=1)
        
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )
        
        # Middle (with time conditioning)
        self.mid_time_proj = nn.Linear(time_embed_dim, base_channels * 4)
        self.mid = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )
        
        # Decoder
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        
        self.conv_out = nn.Conv2d(base_channels, latent_channels, 3, padding=1)
        
        # Initialize output to zero
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] latent state
            t: [B] time values
            y: [B] class labels (ignored)
        
        Returns:
            control: [B, C, H, W] control signal
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        h = self.conv_in(x)
        h = self.down1(h)
        h = self.down2(h)
        
        # Middle with time conditioning
        t_proj = self.mid_time_proj(t_emb)[:, :, None, None]
        h = h + t_proj
        h = self.mid(h)
        
        # Decoder
        h = self.up1(h)
        h = self.up2(h)
        
        # Output
        control = self.conv_out(h)
        
        return control


class ControlNetworkLoRA(nn.Module):
    """
    LoRA-style control: low-rank additive control in feature space.
    
    Instead of outputting full control signal, outputs low-rank 
    matrices that modulate the base model's behavior.
    
    This is a simplified version - true LoRA would inject into 
    the base model's attention layers.
    
    Parameters: ~0.5-2M (very efficient)
    """
    
    def __init__(
        self,
        latent_channels: int = 4,
        spatial_size: int = 32,
        rank: int = 16,
        time_embed_dim: int = 128
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.spatial_size = spatial_size
        self.flat_dim = latent_channels * spatial_size * spatial_size
        self.rank = rank
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU()
        )
        
        # Low-rank factors: control = A @ B^T @ x (roughly)
        # A: [flat_dim, rank], B: [flat_dim, rank]
        self.A = nn.Linear(time_embed_dim, self.flat_dim * rank)
        self.B = nn.Linear(time_embed_dim, self.flat_dim * rank)
        
        # Scaling factor (learnable)
        self.scale = nn.Parameter(torch.zeros(1))
        
        # Initialize to near-zero
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.normal_(self.B.weight, std=0.01)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        
        # Flatten input
        x_flat = x.view(batch_size, -1)  # [B, D]
        
        # Time embedding
        t_emb = self.time_mlp(t)  # [B, time_embed_dim]
        
        # Generate low-rank factors
        A = self.A(t_emb).view(batch_size, self.flat_dim, self.rank)  # [B, D, r]
        B = self.B(t_emb).view(batch_size, self.flat_dim, self.rank)  # [B, D, r]
        
        # Low-rank control: u = scale * A @ (B^T @ x)
        Bx = torch.einsum('bdr,bd->br', B, x_flat)  # [B, r]
        control_flat = torch.einsum('bdr,br->bd', A, Bx)  # [B, D]
        control_flat = self.scale * control_flat
        
        # Reshape
        control = control_flat.view(
            batch_size, self.latent_channels,
            self.spatial_size, self.spatial_size
        )
        
        return control


# ==============================================================================
# Helper: Interpolant (same as adjoint_matching.py)
# ==============================================================================

def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if t.dim() == 0:
        t = t.unsqueeze(0)
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t.expand_as(x)


class Interpolant:
    """Interpolant definitions matching SiT."""
    
    def __init__(self, path_type: str = "Linear"):
        self.path_type = path_type
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        if self.path_type == "Linear":
            return t
        elif self.path_type == "GVP":
            return torch.sqrt(t.clamp(min=1e-8))
        elif self.path_type == "VP":
            return torch.sqrt((1 - (1 - t) ** 2).clamp(min=1e-8))
        return t
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        if self.path_type == "Linear":
            return 1 - t
        elif self.path_type == "GVP":
            return torch.sqrt((1 - t).clamp(min=1e-8))
        elif self.path_type == "VP":
            return 1 - t
        return 1 - t
    
    def interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor,
                    t: torch.Tensor) -> torch.Tensor:
        t = expand_t_like_x(t, x_0)
        return self.alpha(t) * x_1 + self.sigma(t) * x_0


class MemorylessSchedule:
    """Memoryless schedule for AM (eta >= 1)."""
    
    def __init__(self, eta: float = 1.0):
        if eta < 1.0:
            raise ValueError(f"eta must be >= 1.0, got {eta}")
        self.eta = eta
    
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.eta * (1 - t)


# ==============================================================================
# Adjoint Matching Loss for Control Network
# ==============================================================================

class AdjointMatchingControlLoss(nn.Module):
    """
    AM loss for explicit control network.
    
    Loss = ||u_θ(x,t) - u*||²
    
    Where u* = σ(t)² · λ · adjoint
    
    Note: This is simpler than the implicit version because
    u_θ is directly the output of control_net, not a difference.
    """
    
    def __init__(
        self,
        schedule: MemorylessSchedule,
        reward_scale: float = 10.0
    ):
        super().__init__()
        self.schedule = schedule
        self.reward_scale = reward_scale
    
    def forward(
        self,
        control_net: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        adjoint: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute AM loss for control network.
        
        Args:
            control_net: Control network u_θ
            x_t: State at time t
            t: Time
            y: Class labels (may be ignored by control_net)
            adjoint: Adjoint vector at time t
        
        Returns:
            Dict with loss and diagnostics
        """
        # Get control from network
        control = control_net(x_t, t, y)
        
        # Optimal control
        t_expanded = expand_t_like_x(t, x_t)
        sigma_t = self.schedule.get_sigma(t_expanded)
        optimal_control = (sigma_t ** 2) * self.reward_scale * adjoint
        
        # MSE loss
        loss = F.mse_loss(control, optimal_control)
        
        return {
            'loss': loss,
            'control_norm': control.norm().item(),
            'optimal_norm': optimal_control.norm().item(),
            'adjoint_norm': adjoint.norm().item()
        }


# ==============================================================================
# Lean Adjoint for Control Network
# ==============================================================================

def compute_lean_adjoint_control(
    base_model: nn.Module,
    control_net: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    y: Optional[torch.Tensor],
    adjoint: torch.Tensor,
    cfg_scale: float = 1.0
) -> torch.Tensor:
    """
    Compute VJP for combined model: v = v_base + u_θ
    
    Since v_base is frozen, we only need VJP through the 
    combined forward pass for the adjoint ODE.
    """
    x_t = x_t.detach().requires_grad_(True)
    
    # Base velocity (frozen)
    with torch.no_grad():
        if cfg_scale > 1.0 and y is not None:
            v_base_cond = base_model(x_t, t, y)
            v_base_uncond = base_model(x_t, t, torch.zeros_like(y))
            v_base = v_base_uncond + cfg_scale * (v_base_cond - v_base_uncond)
        else:
            v_base = base_model(x_t, t, y)
    
    # Control (may require grad for VJP)
    control = control_net(x_t, t, y)
    
    # Combined velocity
    v = v_base + control
    
    # VJP
    vjp = torch.autograd.grad(
        outputs=v,
        inputs=x_t,
        grad_outputs=adjoint,
        create_graph=False
    )[0]
    
    return vjp


def solve_adjoint_ode_control(
    base_model: nn.Module,
    control_net: nn.Module,
    trajectory: torch.Tensor,
    times: torch.Tensor,
    y: Optional[torch.Tensor],
    terminal_adjoint: torch.Tensor,
    cfg_scale: float = 1.0
) -> torch.Tensor:
    """Solve adjoint ODE backward for control network setup."""
    
    num_steps = len(times) - 1
    batch_size = trajectory.shape[1]
    
    adjoint_traj = torch.zeros_like(trajectory)
    adjoint_traj[-1] = terminal_adjoint
    
    for i in range(num_steps - 1, -1, -1):
        t_curr = times[i + 1]
        dt = times[i + 1] - times[i]
        
        x_curr = trajectory[i + 1]
        a_curr = adjoint_traj[i + 1]
        
        t_batch = t_curr.expand(batch_size)
        
        vjp = compute_lean_adjoint_control(
            base_model, control_net, x_curr, t_batch, y, a_curr, cfg_scale
        )
        
        adjoint_traj[i] = a_curr + dt * vjp
    
    return adjoint_traj


# ==============================================================================
# Main Trainer
# ==============================================================================

class AdjointMatchingControlTrainer:
    """
    AM trainer with explicit control network.
    
    Key difference from AdjointMatchingTrainer:
    - Only control_net is trained (much smaller)
    - base_model is completely frozen
    - v_finetuned = v_base + control_net(x, t)
    
    Usage:
        control_net = ControlNetworkConv(latent_channels=4)
        trainer = AdjointMatchingControlTrainer(
            control_net=control_net,
            base_model=frozen_sit_model,
            reward_fn=aesthetic_reward
        )
        
        # Only optimize control_net parameters!
        optimizer = AdamW(control_net.parameters(), lr=1e-4)
        
        for step in range(1000):
            loss = trainer.training_step(batch_size=8)
            loss['loss'].backward()
            optimizer.step()
    """
    
    def __init__(
        self,
        control_net: nn.Module,
        base_model: nn.Module,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        path_type: str = "Linear",
        reward_scale: float = 10.0,
        cfg_scale: float = 4.0,
        num_sampling_steps: int = 50,
        eta: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Args:
            control_net: Small trainable control network
            base_model: Frozen pre-trained SiT
            reward_fn: Reward function r(x)
            path_type: Interpolant type (must match base_model)
            reward_scale: λ multiplier
            cfg_scale: CFG scale for base model
            num_sampling_steps: Steps for trajectory
            eta: DDIM eta (>=1)
            device: Device
        """
        self.control_net = control_net
        self.base_model = base_model
        self.reward_fn = reward_fn
        self.device = device
        self.cfg_scale = cfg_scale
        self.num_sampling_steps = num_sampling_steps
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        
        # Setup
        self.interpolant = Interpolant(path_type)
        self.schedule = MemorylessSchedule(eta=eta)
        self.loss_fn = AdjointMatchingControlLoss(
            schedule=self.schedule,
            reward_scale=reward_scale
        )
        
        # Count parameters
        base_params = sum(p.numel() for p in base_model.parameters())
        control_params = sum(p.numel() for p in control_net.parameters())
        print(f"Base model: {base_params:,} params (frozen)")
        print(f"Control net: {control_params:,} params (trainable)")
        print(f"Ratio: {control_params/base_params*100:.2f}%")
    
    @torch.no_grad()
    def generate_trajectory(
        self,
        z_0: torch.Tensor,
        y: Optional[torch.Tensor],
        use_control: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate trajectory using v = v_base + u_θ (if use_control=True)
        or just v_base (if use_control=False, for baseline comparison).
        """
        batch_size = z_0.shape[0]
        device = z_0.device
        
        times = torch.linspace(0, 1, self.num_sampling_steps + 1, device=device)
        dt = 1.0 / self.num_sampling_steps
        
        trajectory = torch.zeros(
            self.num_sampling_steps + 1, *z_0.shape, device=device
        )
        trajectory[0] = z_0
        
        x = z_0.clone()
        
        for i, t in enumerate(times[:-1]):
            t_batch = t.expand(batch_size)
            
            # Base velocity
            if self.cfg_scale > 1.0 and y is not None:
                v_cond = self.base_model(x, t_batch, y)
                v_uncond = self.base_model(x, t_batch, torch.zeros_like(y))
                v_base = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            else:
                v_base = self.base_model(x, t_batch, y)
            
            # Add control if enabled
            if use_control:
                control = self.control_net(x, t_batch, y)
                v = v_base + control
            else:
                v = v_base
            
            x = x + dt * v
            trajectory[i + 1] = x
        
        return trajectory, times
    
    def compute_terminal_adjoint(self, x_T: torch.Tensor) -> torch.Tensor:
        """Compute ∇_x r(x_T)."""
        x_T = x_T.detach().requires_grad_(True)
        reward = self.reward_fn(x_T)
        total_reward = reward.sum()
        terminal_adjoint = torch.autograd.grad(total_reward, x_T)[0]
        return terminal_adjoint.detach()
    
    def training_step(
        self,
        batch_size: int,
        latent_size: int = 32,
        num_classes: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """
        Complete training step.
        
        Note: Only control_net gets gradients!
        """
        device = next(self.control_net.parameters()).device
        
        # Sample noise and labels
        z_0 = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
        
        # Generate trajectory (with current control)
        trajectory, times = self.generate_trajectory(z_0, y, use_control=True)
        
        # Terminal adjoint
        x_T = trajectory[-1]
        terminal_adjoint = self.compute_terminal_adjoint(x_T)
        
        # Solve adjoint ODE
        adjoint_traj = solve_adjoint_ode_control(
            base_model=self.base_model,
            control_net=self.control_net,
            trajectory=trajectory,
            times=times,
            y=y,
            terminal_adjoint=terminal_adjoint,
            cfg_scale=self.cfg_scale
        )
        
        # Sample random time
        t_idx = torch.randint(1, self.num_sampling_steps, (1,)).item()
        t = times[t_idx]
        t_batch = t.expand(batch_size).to(device)
        
        x_t = trajectory[t_idx].detach()
        adjoint_t = adjoint_traj[t_idx].detach()
        
        # Compute loss (only control_net gets gradients)
        self.control_net.train()
        loss_dict = self.loss_fn(
            control_net=self.control_net,
            x_t=x_t,
            t=t_batch,
            y=y,
            adjoint=adjoint_t
        )
        
        return loss_dict
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        latent_size: int = 32,
        num_classes: int = 1000,
        use_control: bool = True,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples with or without control.
        
        Args:
            use_control: If True, use v_base + u_θ. If False, use v_base only.
        
        Useful for A/B comparison.
        """
        device = next(self.control_net.parameters()).device
        
        z_0 = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        
        if class_labels is not None:
            y = class_labels.to(device)
        else:
            y = torch.randint(0, num_classes, (batch_size,), device=device)
        
        trajectory, _ = self.generate_trajectory(z_0, y, use_control=use_control)
        
        return trajectory[-1]  # Return final samples


# ==============================================================================
# Factory function
# ==============================================================================

def create_control_network(
    arch: str = "conv",
    latent_channels: int = 4,
    spatial_size: int = 32,
    **kwargs
) -> nn.Module:
    """
    Create control network by name.
    
    Args:
        arch: "mlp", "conv", or "lora"
        latent_channels: Number of latent channels (4 for SD VAE)
        spatial_size: Spatial size of latents (32 for 256px images)
    
    Returns:
        Control network module
    """
    if arch == "mlp":
        return ControlNetworkMLP(
            latent_channels=latent_channels,
            spatial_size=spatial_size,
            **kwargs
        )
    elif arch == "conv":
        return ControlNetworkConv(
            latent_channels=latent_channels,
            **kwargs
        )
    elif arch == "lora":
        return ControlNetworkLoRA(
            latent_channels=latent_channels,
            spatial_size=spatial_size,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")