"""
Adjoint Matching for SiT Fine-tuning
=====================================

This module integrates with SiT's transport module to provide reward-based
fine-tuning via Adjoint Matching (Domingo-Enrich et al., ICLR 2025).

Usage:
    Place this file in SiT/finetune/adjoint_matching.py
    
    from finetune.adjoint_matching import AdjointMatchingTrainer
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Dict, Tuple
import copy

# ==============================================================================
# Import from SiT's transport module (assumes running from SiT root)
# ==============================================================================
try:
    from transport import create_transport
    from transport.path import ICPlan, VPCPlan, GVPCPlan
    SIT_AVAILABLE = True
except ImportError:
    SIT_AVAILABLE = False
    print("Warning: SiT transport module not found. Using standalone definitions.")


# ==============================================================================
# Standalone Interpolant (fallback if SiT not available, or for reference)
# ==============================================================================
def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand time tensor to match x dimensions."""
    if t.dim() == 0:
        t = t.unsqueeze(0)
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t.expand_as(x)


class Interpolant:
    """
    Interpolant definitions matching SiT's transport/path.py
    
    I_t = α(t) * x_1 + σ(t) * x_0
    
    where x_0 is noise, x_1 is data (SiT convention: t=0 is noise, t=1 is data)
    """
    
    def __init__(self, path_type: str = "Linear"):
        """
        Args:
            path_type: One of "Linear", "GVP", "VP" (matching SiT's --path-type)
        """
        self.path_type = path_type
    
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Data coefficient α(t). At t=1, α=1 (pure data)."""
        if self.path_type == "Linear":
            return t
        elif self.path_type == "GVP":
            return torch.sqrt(t.clamp(min=1e-8))
        elif self.path_type == "VP":
            return torch.sqrt((1 - (1 - t) ** 2).clamp(min=1e-8))
        return t
    
    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Noise coefficient σ(t). At t=0, σ=1 (pure noise)."""
        if self.path_type == "Linear":
            return 1 - t
        elif self.path_type == "GVP":
            return torch.sqrt((1 - t).clamp(min=1e-8))
        elif self.path_type == "VP":
            return 1 - t
        return 1 - t
    
    def d_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative dα/dt."""
        if self.path_type == "Linear":
            return torch.ones_like(t)
        elif self.path_type == "GVP":
            return 0.5 / torch.sqrt(t.clamp(min=1e-8))
        elif self.path_type == "VP":
            return (1 - t) / torch.sqrt((1 - (1 - t) ** 2).clamp(min=1e-8))
        return torch.ones_like(t)
    
    def d_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Derivative dσ/dt."""
        if self.path_type == "Linear":
            return -torch.ones_like(t)
        elif self.path_type == "GVP":
            return -0.5 / torch.sqrt((1 - t).clamp(min=1e-8))
        elif self.path_type == "VP":
            return -torch.ones_like(t)
        return -torch.ones_like(t)
    
    def interpolate(self, x_0: torch.Tensor, x_1: torch.Tensor, 
                    t: torch.Tensor) -> torch.Tensor:
        """
        Compute x_t = α(t) * x_1 + σ(t) * x_0
        
        Args:
            x_0: Noise samples [B, C, H, W]
            x_1: Data samples [B, C, H, W]  
            t: Time [B] or scalar
        """
        t = expand_t_like_x(t, x_0)
        return self.alpha(t) * x_1 + self.sigma(t) * x_0
    
    def velocity(self, x_0: torch.Tensor, x_1: torch.Tensor,
                 t: torch.Tensor) -> torch.Tensor:
        """
        Compute target velocity: v = dα/dt * x_1 + dσ/dt * x_0
        """
        t = expand_t_like_x(t, x_0)
        return self.d_alpha(t) * x_1 + self.d_sigma(t) * x_0




# ==============================================================================
# Memoryless Schedule (required for SOC formulation)
# ==============================================================================
class MemorylessSchedule:
    """
    Memoryless noise schedule for Adjoint Matching.
    
    From the AM paper (Prop. 1): DDIM with η ≥ 1 is memoryless.
    η = 1 gives the distinguished memoryless schedule.
    
    For flow matching, we need to add diffusion to get an SDE.
    """
    
    def __init__(self, eta: float = 1.0):
        """
        Args:
            eta: DDIM η parameter, must be >= 1 for memoryless property
        """
        if eta < 1.0:
            raise ValueError(f"eta must be >= 1.0 for memoryless schedule, got {eta}")
        self.eta = eta
    
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Get diffusion coefficient at time t."""
        # For flow matching with linear path: σ_sde(t) = η * (1 - t)
        return self.eta * (t)


# ==============================================================================
# Lean Adjoint Computation (memory-efficient via VJP)
# ==============================================================================
def compute_lean_adjoint(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    y: Optional[torch.Tensor],
    adjoint: torch.Tensor,
    cfg_scale: float = 1.0
) -> torch.Tensor:
    """
    Compute Vector-Jacobian Product: adjoint^T @ (∂v/∂x)
    
    This is the key operation for adjoint ODE:
        da/dt = -a^T @ (∂v/∂x)
    
    Memory: O(dim) instead of O(dim²) for full Jacobian
    
    Args:
        model: Velocity model v_θ(x, t, y)
        x_t: Current state [B, C, H, W]
        t: Time [B]
        y: Class labels [B] (optional)
        adjoint: Current adjoint vector [B, C, H, W]
        cfg_scale: Classifier-free guidance scale
    
    Returns:
        VJP result: a^T @ (∂v/∂x), same shape as x_t
    """
    x_t = x_t.detach().requires_grad_(True)
    
    # Forward pass to get velocity
    if cfg_scale > 1.0 and y is not None:
        # CFG: v = v_uncond + scale * (v_cond - v_uncond)
        v_cond = model(x_t, t, y)
        v_uncond = model(x_t, t, torch.zeros_like(y))
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
    else:
        v = model(x_t, t, y)
    
    # Compute VJP: adjoint^T @ (∂v/∂x)
    vjp = torch.autograd.grad(
        outputs=v,
        inputs=x_t,
        grad_outputs=adjoint,
        create_graph=False,
        retain_graph=False
    )[0]
    
    return vjp


def solve_adjoint_ode(
    model: nn.Module,
    trajectory: torch.Tensor,  # [T+1, B, C, H, W]
    times: torch.Tensor,       # [T+1]
    y: Optional[torch.Tensor],
    terminal_adjoint: torch.Tensor,  # [B, C, H, W] = ∇r(x_T)
    cfg_scale: float = 1.0
) -> torch.Tensor:
    """
    Solve adjoint ODE backward from t=1 to t=0.
    
    da/dt = -a^T @ (∂v/∂x)
    a(T) = ∇r(x_T)  (terminal condition)
    
    Args:
        model: Velocity model
        trajectory: Forward trajectory states
        times: Time points [T+1]
        y: Class labels
        terminal_adjoint: ∇_x r(x_T) at terminal time
        cfg_scale: CFG scale
        
    Returns:
        adjoint_trajectory: [T+1, B, C, H, W] adjoint at each time
    """
    num_steps = len(times) - 1
    batch_size = trajectory.shape[1]
    
    # Initialize adjoint trajectory
    adjoint_traj = torch.zeros_like(trajectory)
    adjoint_traj[-1] = terminal_adjoint  # a(T) = ∇r(x_T)
    
    # Backward Euler integration
    for i in range(num_steps - 1, -1, -1):
        t_curr = times[i + 1]
        t_prev = times[i]
        dt = t_curr - t_prev  # positive since times are increasing
        
        x_curr = trajectory[i + 1]
        a_curr = adjoint_traj[i + 1]
        
        # Time tensor for batch
        t_batch = t_curr.expand(batch_size)
        
        # Compute VJP
        vjp = compute_lean_adjoint(model, x_curr, t_batch, y, a_curr, cfg_scale)
        
        # Backward Euler: a(t-dt) = a(t) + dt * vjp
        # (note: we're going backward, so we add)
        adjoint_traj[i] = a_curr + dt * vjp
    
    return adjoint_traj


# ==============================================================================
# Adjoint Matching Loss
# ==============================================================================
class AdjointMatchingLoss(nn.Module):
    """
    Adjoint Matching loss for fine-tuning.
    
    L_AM = || (v_θ - v_base) - u* ||²
    
    where u* = σ(t)² · λ · adjoint is the optimal control
    
    This trains the model to match the optimal control that maximizes reward.
    """
    
    def __init__(
        self,
        interpolant: Interpolant,
        schedule: MemorylessSchedule,
        reward_scale: float = 10.0
    ):
        super().__init__()
        self.interpolant = interpolant
        self.schedule = schedule
        self.reward_scale = reward_scale  # λ in the paper
    
    def forward(
        self,
        model: nn.Module,
        base_model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        adjoint: torch.Tensor,
        cfg_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute AM loss.
        
        Args:
            model: Fine-tuned model (trainable)
            base_model: Base model (frozen)
            x_t: Interpolated state at time t
            t: Time
            y: Class labels
            adjoint: Adjoint vector at time t
            cfg_scale: CFG scale
            
        Returns:
            Dict with 'loss' and diagnostic values
        """
        # Get velocities from both models
        with torch.no_grad():
            if cfg_scale > 1.0 and y is not None:
                v_base_cond = base_model(x_t, t, y)
                v_base_uncond = base_model(x_t, t, torch.zeros_like(y))
                v_base = v_base_uncond + cfg_scale * (v_base_cond - v_base_uncond)
            else:
                v_base = base_model(x_t, t, y)
        
        if cfg_scale > 1.0 and y is not None:
            v_model_cond = model(x_t, t, y)
            v_model_uncond = model(x_t, t, torch.zeros_like(y))
            v_model = v_model_uncond + cfg_scale * (v_model_cond - v_model_uncond)
        else:
            v_model = model(x_t, t, y)
        
        # Control: u_θ = v_θ - v_base
        control = v_model - v_base
        
        # Optimal control: u* = σ(t)² · λ · adjoint
        t_expanded = expand_t_like_x(t, x_t)
        sigma_t = self.schedule.get_sigma(t_expanded)
        optimal_control = (sigma_t ** 2) * self.reward_scale * adjoint
        
        # MSE loss
        loss = torch.mean((control - optimal_control) ** 2)
        
        return {
            'loss': loss,
            'control_norm': control.norm().item(),
            'optimal_norm': optimal_control.norm().item(),
            'adjoint_norm': adjoint.norm().item()
        }


# ==============================================================================
# Main Trainer Class
# ==============================================================================
class AdjointMatchingTrainer:
    """
    Complete Adjoint Matching trainer for SiT.
    
    Integrates with SiT's transport module and provides the full training loop.
    
    Usage:
        trainer = AdjointMatchingTrainer(
            model=model,
            base_model=base_model,
            reward_fn=aesthetic_score,
            path_type="Linear"  # Match your SiT checkpoint
        )
        
        for step in range(num_steps):
            loss_dict = trainer.training_step(batch_size=4, latent_size=32)
            loss_dict['loss'].backward()
            optimizer.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_model: nn.Module,
        reward_fn: Callable[[torch.Tensor], torch.Tensor],
        path_type: str = "Linear",
        reward_scale: float = 10.0,
        cfg_scale: float = 4.0,
        num_sampling_steps: int = 50,
        sigma: int = 1.0,
        eta: float = 1.0,
        device: str = 'cuda'
    ):
        """
        Args:
            model: Fine-tuned SiT model (will be trained)
            base_model: Pre-trained SiT model (frozen reference)
            reward_fn: r(x) -> [B] reward function on generated images
            path_type: "Linear", "GVP", or "VP" (must match checkpoint)
            reward_scale: λ multiplier for reward gradient
            cfg_scale: Classifier-free guidance scale
            num_sampling_steps: Steps for generating trajectory
            eta: DDIM η (>=1 for memoryless)
            device: cuda or cpu
        """
        self.model = model
        self.base_model = base_model
        self.reward_fn = reward_fn
        self.device = device
        self.cfg_scale = cfg_scale
        self.num_sampling_steps = num_sampling_steps
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.base_model.eval()
        
        # Setup interpolant and schedule
        self.interpolant = Interpolant(path_type)
        self.schedule = MemorylessSchedule(eta=eta)
        
        # Loss function
        self.loss_fn = AdjointMatchingLoss(
            interpolant=self.interpolant,
            schedule=self.schedule,
            reward_scale=reward_scale
        )
    
    @torch.no_grad()
    def generate_trajectory(
        self,
        z_0: torch.Tensor,
        y: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate forward trajectory from noise to data using base model.
        
        Args:
            z_0: Initial noise [B, C, H, W]
            y: Class labels [B]
            
        Returns:
            trajectory: [T+1, B, C, H, W] states at each time
            times: [T+1] time values
        """
        batch_size = z_0.shape[0]
        device = z_0.device
        
        # Time discretization: t ∈ [0, 1], increasing
        times = torch.linspace(0, 1, self.num_sampling_steps + 1, device=device)
        dt = 1.0 / self.num_sampling_steps
        
        # Initialize trajectory storage
        trajectory = torch.zeros(
            self.num_sampling_steps + 1, *z_0.shape, device=device
        )
        trajectory[0] = z_0
        
        x = z_0.clone()
        
        # Forward Euler integration
        for i, t in enumerate(times[:-1]):
            t_batch = t.expand(batch_size)
            
            # Get velocity from base model
            if self.cfg_scale > 1.0 and y is not None:
                v_cond = self.base_model(x, t_batch, y)
                v_uncond = self.base_model(x, t_batch, torch.zeros_like(y))
                v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            else:
                v = self.base_model(x, t_batch, y)
            
            # Euler–Maruyama SDE step
            t_scalar = t  # scalar tensor
            sigma = self.schedule.get_sigma(t_scalar)          # shape []
            noise = torch.randn_like(x)

            # x = x + dt * v + (dt ** 0.5) * (self.loss_fn.reward_scale ** 0.5) * sigma * noise
            x = x + dt * v + (dt ** 0.5) * (1.0 ** 0.5) * sigma * noise
            trajectory[i + 1].copy_(x)
        
        return trajectory, times
    
    @torch.no_grad()
    def generate_trajectory_with_model(self, z_0, y):
        batch_size = z_0.shape[0]
        device = z_0.device
        times = torch.linspace(0, 1, self.num_sampling_steps + 1, device=device)
        dt = 1.0 / self.num_sampling_steps

        trajectory = torch.zeros(self.num_sampling_steps + 1, *z_0.shape, device=device)
        trajectory[0] = z_0
        x = z_0.clone()

        for i, t in enumerate(times[:-1]):
            t_batch = t.expand(batch_size)
            # use the *trainable* model here
            if self.cfg_scale > 1.0 and y is not None:
                v_cond = self.model(x, t_batch, y)
                v_uncond = self.model(x, t_batch, torch.zeros_like(y))
                v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(x, t_batch, y)
            
            # Euler–Maruyama SDE step
            t_scalar = t  # scalar tensor
            sigma = self.schedule.get_sigma(t_scalar)          # shape []
            noise = torch.randn_like(x)
            # x = x + dt * v + (dt ** 0.5) * (self.loss_fn.reward_scale ** 0.5) * sigma * noise
            x = x + dt * v + (dt ** 0.5) * (1.0 ** 0.5) * sigma * noise
            trajectory[i + 1].copy_(x)

        return trajectory, times

    
    def compute_terminal_adjoint(self, x_T: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute terminal adjoint: a(T) = ∇_x r(x_T)
        
        Args:
            x_T: Final generated samples [B, C, H, W]
            
        Returns:
            Terminal adjoint [B, C, H, W]
        """

        '''
        x_T = x_T.detach().requires_grad_(True)
        reward = self.reward_fn(x_T)  # [B]
        
        # Sum rewards and backprop
        total_reward = reward.sum()
        terminal_adjoint = torch.autograd.grad(
            total_reward, x_T, create_graph=False
        )[0]

        # test = torch.randn_like(x_T)
        # r_test = self.reward_fn(test)
        # print("r_test:", r_test.mean().item(), r_test.min().item(), r_test.max().item())

        # print("aT stats:",
            # terminal_adjoint.mean().item(),
            # terminal_adjoint.abs().max().item(),
            # terminal_adjoint.flatten(1).norm(dim=1).mean().item())
        
        return terminal_adjoint.detach()
        
        # Always make a leaf for terminal gradient
        with torch.inference_mode(False):
            with torch.enable_grad():
                x = x_T.detach().requires_grad_(True)

                # --- ONLY for ImageReward (pixel-space reward) when x is latent ---
                if getattr(self.reward_fn, "__class__", None).__name__ == "ImageRewardScorer" and x.dim() == 4 and x.shape[1] == 4:
                    # decode latent -> pixel (differentiable wrt x)
                    for p in self.vae.parameters():
                        p.requires_grad_(False)
                    img = self.vae.decode(x).sample          # diffusers-style
                    img = (img / 2 + 0.5).clamp(0, 1)
                    r = self.reward_fn(img, None)  # [B]
                else:
                    r = self.reward_fn(x)    # [B]

                (g,) = torch.autograd.grad(r.sum(), x, create_graph=False, retain_graph=False)

        return g.detach()
        '''
        with torch.inference_mode(False):
            with torch.enable_grad():
                x = x_T.detach().requires_grad_(True)
                try:
                    r = self.reward_fn(x, class_labels=y)  # <-- keyword
                except TypeError:
                    r = self.reward_fn(x)                  # for rewards that don't accept it
                (g,) = torch.autograd.grad(r.sum(), x, create_graph=False, retain_graph=False)
        return g.detach()


    
    
    def training_step(
        self,
        batch_size: int,
        latent_size: int = 32,
        num_classes: int = 398,
        return_intermediates: bool = True  # ADD THIS
    ) -> Dict[str, torch.Tensor]:
        """
        Complete training step.
        
        Args:
            batch_size: Number of samples
            latent_size: Spatial size of latent (e.g., 32 for 256px with VAE)
            num_classes: Number of classes for conditioning
            
        Returns:
            Dict with 'loss' and diagnostics
        """
        device = next(self.model.parameters()).device
        
        # 1. Sample initial noise and class labels
        z_0 = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
        
        # 2. Generate trajectory using current model
        trajectory, times = self.generate_trajectory_with_model(z_0, y)
        
        # 3. Compute terminal adjoint: ∇r(x_T)
        x_T = trajectory[-1]
        terminal_adjoint = self.compute_terminal_adjoint(x_T, y=y)

        # --- reward stats under *current* model ---
        with torch.no_grad():
            traj_model, _ = self.generate_trajectory_with_model(z_0, y)
            x_T_model = traj_model[-1]
            rewards = self.reward_fn(x_T_model, class_labels=y)
            reward_mean = rewards.mean()
            reward_std = rewards.std(unbiased=False)
        
        # 4. Solve adjoint ODE backward
        adjoint_traj = solve_adjoint_ode(
            # model=self.base_model,
            model=self.model,
            trajectory=trajectory,
            times=times,
            y=y,
            terminal_adjoint=terminal_adjoint,
            cfg_scale=self.cfg_scale
        )
        
        # 5. Sample random time and compute loss
        t_idx = torch.randint(1, self.num_sampling_steps, (1,)).item()
        t = times[t_idx]
        t_batch = t.expand(batch_size).to(device)
        
        x_t = trajectory[t_idx].detach()
        adjoint_t = adjoint_traj[t_idx].detach()
        
        # 6. Compute AM loss
        self.model.train()
        loss_dict = self.loss_fn(
            model=self.model,
            base_model=self.base_model,
            x_t=x_t,
            t=t_batch,
            y=y,
            adjoint=adjoint_t,
            cfg_scale=self.cfg_scale
        )
        
        # --- NEW: add reward stats into loss_dict ---
        loss_dict["reward_mean"] = reward_mean.detach()
        loss_dict["reward_std"] = reward_std.detach()

        if return_intermediates:
            loss_dict["trajectory"] = trajectory.detach()
            loss_dict["adjoint_traj"] = adjoint_traj.detach()
            loss_dict["times"] = times.detach()
            loss_dict["y"] = y.detach()

        '''
        # pick one step/time
        x = trajectory[-1].detach().requires_grad_(True)
        t_batch = times[-1].expand(x.shape[0])
        a = terminal_adjoint.detach()

        # build v the same way your compute_lean_adjoint does (CFG or not)
        v = self.base_model(x, t_batch, y)  # or CFG-composed v

        vjp_autograd = torch.autograd.grad((v * a).sum(), x, create_graph=False)[0]  # J^T a
        vjp_lean = compute_lean_adjoint(self.base_model, x.detach(), t_batch, y, a, self.cfg_scale)

        print("max|lean - autograd|:", (vjp_lean - vjp_autograd).abs().max().item())
        print("max|lean + autograd|:", (vjp_lean + vjp_autograd).abs().max().item())
        '''

        # adj_base  = solve_adjoint_ode(self.base_model, trajectory, times, y, terminal_adjoint, cfg_scale=self.cfg_scale)
        # adj_model = solve_adjoint_ode(self.model,      trajectory, times, y, terminal_adjoint, cfg_scale=self.cfg_scale)

        # t_idx = torch.randint(1, self.num_sampling_steps, (1,)).item()
        # err = (adj_base[t_idx] - adj_model[t_idx]).abs().max().item()
        # print("max |adj_base - adj_model| at t_idx:", err)     

        return loss_dict


# ==============================================================================
# Convenience function for loading checkpoints
# ==============================================================================
def load_sit_for_finetuning(
    checkpoint_path: str,
    model_name: str = "SiT-XL/2",
    device: str = "cuda"
) -> Tuple[nn.Module, nn.Module]:
    """
    Load SiT checkpoint and create model + frozen base model copy.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        model_name: Model architecture name
        device: Device to load to
        
    Returns:
        (model, base_model): Trainable model and frozen reference
    """
    # Import SiT model definitions
    from models import SiT_models
    
    # Create model
    model = SiT_models[model_name]()
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'ema' in ckpt:
        state_dict = ckpt['ema']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Create frozen copy for base model
    base_model = copy.deepcopy(model)
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    
    return model, base_model