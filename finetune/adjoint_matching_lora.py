"""
Adjoint Matching with LoRA Fine-tuning
=======================================

This injects Low-Rank Adapters (LoRA) into SiT's attention layers,
then fine-tunes only the LoRA parameters using Adjoint Matching.

Key idea:
    W_new = W_frozen + α * (A @ B^T)
    
Where A, B are low-rank matrices (rank << hidden_dim).

Advantages over full fine-tuning:
- ~100x fewer trainable parameters
- Faster training, lower memory
- Can merge back into base model for inference
- Can interpolate: W = W_base + α * ΔW (α ∈ [0, 1])

Advantages over separate control network:
- Modifies model behavior directly (more expressive)
- No architectural assumptions about control signal
- Standard approach in LLM/diffusion fine-tuning

Usage:
    from finetune.adjoint_matching_lora import (
        inject_lora_into_sit,
        AdjointMatchingLoRATrainer
    )
    
    # Inject LoRA into SiT
    model, lora_params = inject_lora_into_sit(model, rank=16)
    
    # Train only LoRA parameters
    trainer = AdjointMatchingLoRATrainer(model, lora_params, reward_fn)
    optimizer = AdamW(lora_params, lr=1e-4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Dict, List, Tuple
import math
import copy


# ==============================================================================
# LoRA Layer
# ==============================================================================

class LoRALayer(nn.Module):
    """
    Low-Rank Adapter layer.
    
    Replaces a linear layer with: output = W_base @ x + α * (A @ B^T) @ x
    
    During forward pass, this is equivalent to:
        output = W_base @ x + α * A @ (B^T @ x)
    
    Which is more efficient when rank << hidden_dim.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Dropout (applied to LoRA path only)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize A with Kaiming, B with zeros
        # This means LoRA contribution starts at zero
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base layer output (frozen)
        base_out = self.base_layer(x)
        
        # LoRA output: α/r * (B @ A) @ x = α/r * B @ (A @ x)
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A)  # [*, rank]
        lora_out = F.linear(lora_out, self.lora_B)  # [*, out_features]
        lora_out = self.scaling * lora_out
        
        return base_out + lora_out
    
    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights back into base layer.
        
        Returns a new Linear layer with merged weights.
        Useful for inference (no overhead).
        """
        merged = nn.Linear(
            self.base_layer.in_features,
            self.base_layer.out_features,
            bias=self.base_layer.bias is not None
        )
        
        # W_merged = W_base + α/r * B @ A
        delta_W = self.scaling * (self.lora_B @ self.lora_A)
        merged.weight.data = self.base_layer.weight.data + delta_W
        
        if self.base_layer.bias is not None:
            merged.bias.data = self.base_layer.bias.data.clone()
        
        return merged


# ==============================================================================
# LoRA Injection into SiT
# ==============================================================================

def inject_lora_into_sit(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 1.0,
    target_modules: List[str] = None,
    dropout: float = 0.0
) -> Tuple[nn.Module, List[nn.Parameter]]:
    """
    Inject LoRA layers into SiT model.
    
    Args:
        model: SiT model (will be modified in-place)
        rank: LoRA rank (lower = fewer params, less expressive)
        alpha: LoRA scaling factor
        target_modules: Which modules to inject into. 
                       Default: ['qkv', 'proj', 'fc1', 'fc2'] (attention + MLP)
        dropout: Dropout on LoRA path
    
    Returns:
        model: Modified model with LoRA layers
        lora_params: List of trainable LoRA parameters
    """
    if target_modules is None:
        # Default: attention QKV, projection, and MLP layers
        target_modules = ['qkv', 'proj', 'fc1', 'fc2']
    
    lora_params = []
    
    def _inject_lora(module: nn.Module, prefix: str = ''):
        """Recursively inject LoRA into target modules."""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, nn.Linear):
                # Check if this is a target module
                should_inject = any(target in name for target in target_modules)
                
                if should_inject:
                    # Replace with LoRA layer
                    lora_layer = LoRALayer(
                        base_layer=child,
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout
                    )
                    setattr(module, name, lora_layer)
                    
                    # Collect LoRA parameters
                    lora_params.append(lora_layer.lora_A)
                    lora_params.append(lora_layer.lora_B)
                    
                    print(f"  Injected LoRA into: {full_name} "
                          f"[{child.in_features}→{child.out_features}, rank={rank}]")
            else:
                # Recurse into children
                _inject_lora(child, full_name)
    
    print(f"Injecting LoRA (rank={rank}, alpha={alpha}) into SiT...")
    _inject_lora(model)
    
    # Freeze all non-LoRA parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze LoRA parameters
    for param in lora_params:
        param.requires_grad = True
    
    total_params = sum(p.numel() for p in model.parameters())
    lora_param_count = sum(p.numel() for p in lora_params)
    
    print(f"\nTotal model params: {total_params:,}")
    print(f"LoRA params: {lora_param_count:,} ({lora_param_count/total_params*100:.2f}%)")
    
    return model, lora_params


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights back into base layers.
    
    After merging, the model has the same architecture as original
    but with updated weights. No inference overhead.
    """
    def _merge_recursive(module: nn.Module):
        for name, child in module.named_children():
            if isinstance(child, LoRALayer):
                merged = child.merge_weights()
                setattr(module, name, merged)
            else:
                _merge_recursive(child)
    
    _merge_recursive(model)
    return model


def set_lora_scale(model: nn.Module, scale: float):
    """
    Adjust LoRA contribution scale for all layers.
    
    Useful for interpolating between base and fine-tuned model:
    - scale=0: Pure base model
    - scale=1: Full LoRA contribution
    - scale=0.5: Halfway blend
    """
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.scaling = scale * module.alpha / module.rank


# ==============================================================================
# Helper classes (same as other files)
# ==============================================================================

def expand_t_like_x(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if t.dim() == 0:
        t = t.unsqueeze(0)
    while t.dim() < x.dim():
        t = t.unsqueeze(-1)
    return t.expand_as(x)


class Interpolant:
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
    
    def interpolate(self, x_0, x_1, t):
        t = expand_t_like_x(t, x_0)
        return self.alpha(t) * x_1 + self.sigma(t) * x_0


class MemorylessSchedule:
    def __init__(self, eta: float = 1.0):
        if eta < 1.0:
            raise ValueError(f"eta must be >= 1.0, got {eta}")
        self.eta = eta
    
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.eta * (1 - t)


# ==============================================================================
# Adjoint Matching Loss for LoRA
# ==============================================================================

class AdjointMatchingLoRALoss(nn.Module):
    """
    AM loss for LoRA fine-tuning.
    
    Same as full model version:
        L = ||(v_lora - v_base) - u*||²
    
    But now v_lora is the model with LoRA adapters,
    and v_base is computed by setting LoRA scale to 0.
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
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: Optional[torch.Tensor],
        adjoint: torch.Tensor,
        cfg_scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute AM loss.
        
        We compute v_base by temporarily setting LoRA scale to 0.
        """
        # Get base velocity (LoRA scale = 0)
        with torch.no_grad():
            set_lora_scale(model, 0.0)
            if cfg_scale > 1.0 and y is not None:
                v_base_cond = model(x_t, t, y)
                v_base_uncond = model(x_t, t, torch.zeros_like(y))
                v_base = v_base_uncond + cfg_scale * (v_base_cond - v_base_uncond)
            else:
                v_base = model(x_t, t, y)
        
        # Get LoRA velocity (LoRA scale = 1)
        set_lora_scale(model, 1.0)
        if cfg_scale > 1.0 and y is not None:
            v_lora_cond = model(x_t, t, y)
            v_lora_uncond = model(x_t, t, torch.zeros_like(y))
            v_lora = v_lora_uncond + cfg_scale * (v_lora_cond - v_lora_uncond)
        else:
            v_lora = model(x_t, t, y)
        
        # Control: difference from base
        control = v_lora - v_base
        
        # Optimal control
        t_expanded = expand_t_like_x(t, x_t)
        sigma_t = self.schedule.get_sigma(t_expanded)
        optimal_control = (sigma_t ** 2) * self.reward_scale * adjoint
        
        # Loss
        loss = F.mse_loss(control, optimal_control)
        
        return {
            'loss': loss,
            'control_norm': control.norm().item(),
            'optimal_norm': optimal_control.norm().item(),
            'adjoint_norm': adjoint.norm().item()
        }


# ==============================================================================
# Lean Adjoint for LoRA
# ==============================================================================

def compute_lean_adjoint_lora(
    model: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    y: Optional[torch.Tensor],
    adjoint: torch.Tensor,
    cfg_scale: float = 1.0
) -> torch.Tensor:
    """Compute VJP with LoRA-enabled model."""
    x_t = x_t.detach().requires_grad_(True)
    
    # Ensure LoRA is active
    set_lora_scale(model, 1.0)
    
    if cfg_scale > 1.0 and y is not None:
        v_cond = model(x_t, t, y)
        v_uncond = model(x_t, t, torch.zeros_like(y))
        v = v_uncond + cfg_scale * (v_cond - v_uncond)
    else:
        v = model(x_t, t, y)
    
    vjp = torch.autograd.grad(
        outputs=v,
        inputs=x_t,
        grad_outputs=adjoint,
        create_graph=False
    )[0]
    
    return vjp


def solve_adjoint_ode_lora(
    model: nn.Module,
    trajectory: torch.Tensor,
    times: torch.Tensor,
    y: Optional[torch.Tensor],
    terminal_adjoint: torch.Tensor,
    cfg_scale: float = 1.0
) -> torch.Tensor:
    """Solve adjoint ODE for LoRA model."""
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
        
        vjp = compute_lean_adjoint_lora(model, x_curr, t_batch, y, a_curr, cfg_scale)
        adjoint_traj[i] = a_curr + dt * vjp
    
    return adjoint_traj


# ==============================================================================
# Main Trainer
# ==============================================================================

class AdjointMatchingLoRATrainer:
    """
    AM trainer with LoRA fine-tuning.
    
    Only LoRA parameters (A, B matrices) are updated.
    Base model weights remain frozen.
    
    Usage:
        # Load and inject LoRA
        model = load_sit_model(...)
        model, lora_params = inject_lora_into_sit(model, rank=16)
        
        # Create trainer
        trainer = AdjointMatchingLoRATrainer(
            model=model,
            lora_params=lora_params,
            reward_fn=reward_fn
        )
        
        # ONLY optimize LoRA params
        optimizer = AdamW(lora_params, lr=1e-4)
        
        for step in range(1000):
            loss = trainer.training_step(batch_size=8)
            loss['loss'].backward()
            optimizer.step()
    """
    
    def __init__(
        self,
        model: nn.Module,
        lora_params: List[nn.Parameter],
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
            model: SiT model with LoRA layers injected
            lora_params: List of LoRA parameters (from inject_lora_into_sit)
            reward_fn: Reward function
            path_type: Interpolant type
            reward_scale: λ multiplier
            cfg_scale: CFG scale
            num_sampling_steps: Sampling steps
            eta: DDIM eta
            device: Device
        """
        self.model = model
        self.lora_params = lora_params
        self.reward_fn = reward_fn
        self.device = device
        self.cfg_scale = cfg_scale
        self.num_sampling_steps = num_sampling_steps
        
        self.interpolant = Interpolant(path_type)
        self.schedule = MemorylessSchedule(eta=eta)
        self.loss_fn = AdjointMatchingLoRALoss(
            schedule=self.schedule,
            reward_scale=reward_scale
        )
    
    @torch.no_grad()
    def generate_trajectory(
        self,
        z_0: torch.Tensor,
        y: Optional[torch.Tensor],
        use_lora: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectory with or without LoRA."""
        batch_size = z_0.shape[0]
        device = z_0.device
        
        # Set LoRA scale
        set_lora_scale(self.model, 1.0 if use_lora else 0.0)
        
        times = torch.linspace(0, 1, self.num_sampling_steps + 1, device=device)
        dt = 1.0 / self.num_sampling_steps
        
        trajectory = torch.zeros(
            self.num_sampling_steps + 1, *z_0.shape, device=device
        )
        trajectory[0] = z_0
        
        x = z_0.clone()
        
        for i, t in enumerate(times[:-1]):
            t_batch = t.expand(batch_size)
            
            if self.cfg_scale > 1.0 and y is not None:
                v_cond = self.model(x, t_batch, y)
                v_uncond = self.model(x, t_batch, torch.zeros_like(y))
                v = v_uncond + self.cfg_scale * (v_cond - v_uncond)
            else:
                v = self.model(x, t_batch, y)
            
            x = x + dt * v
            trajectory[i + 1] = x
        
        return trajectory, times
    
    def compute_terminal_adjoint(self, x_T: torch.Tensor) -> torch.Tensor:
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
        """Training step - only LoRA params get gradients."""
        device = next(self.model.parameters()).device
        
        z_0 = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
        
        # Generate trajectory with LoRA
        trajectory, times = self.generate_trajectory(z_0, y, use_lora=True)
        
        # Terminal adjoint
        x_T = trajectory[-1]
        terminal_adjoint = self.compute_terminal_adjoint(x_T)
        
        # Solve adjoint ODE
        adjoint_traj = solve_adjoint_ode_lora(
            model=self.model,
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
        
        # Compute loss
        self.model.train()
        loss_dict = self.loss_fn(
            model=self.model,
            x_t=x_t,
            t=t_batch,
            y=y,
            adjoint=adjoint_t,
            cfg_scale=self.cfg_scale
        )
        
        return loss_dict
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        latent_size: int = 32,
        num_classes: int = 1000,
        lora_scale: float = 1.0,
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate samples with adjustable LoRA strength.
        
        Args:
            lora_scale: 0.0 = base model, 1.0 = full LoRA, 0.5 = blend
        """
        device = next(self.model.parameters()).device
        
        set_lora_scale(self.model, lora_scale)
        
        z_0 = torch.randn(batch_size, 4, latent_size, latent_size, device=device)
        
        if class_labels is not None:
            y = class_labels.to(device)
        else:
            y = torch.randint(0, num_classes, (batch_size,), device=device)
        
        trajectory, _ = self.generate_trajectory(z_0, y, use_lora=True)
        
        return trajectory[-1]
    
    def save_lora_weights(self, path: str):
        """Save only LoRA weights (very small file)."""
        lora_state = {}
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer):
                lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
                lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
        
        torch.save(lora_state, path)
        print(f"Saved LoRA weights to {path}")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights."""
        lora_state = torch.load(path, map_location='cpu')
        
        for name, module in self.model.named_modules():
            if isinstance(module, LoRALayer):
                if f"{name}.lora_A" in lora_state:
                    module.lora_A.data = lora_state[f"{name}.lora_A"].to(
                        module.lora_A.device
                    )
                    module.lora_B.data = lora_state[f"{name}.lora_B"].to(
                        module.lora_B.device
                    )
        
        print(f"Loaded LoRA weights from {path}")


# ==============================================================================
# Convenience function
# ==============================================================================

def setup_lora_finetuning(
    model: nn.Module,
    reward_fn: Callable,
    rank: int = 16,
    alpha: float = 1.0,
    path_type: str = "Linear",
    reward_scale: float = 10.0,
    cfg_scale: float = 4.0,
    lr: float = 1e-4,
    device: str = 'cuda'
) -> Tuple[AdjointMatchingLoRATrainer, torch.optim.Optimizer]:
    """
    One-liner setup for LoRA fine-tuning.
    
    Returns:
        trainer: Ready-to-use trainer
        optimizer: AdamW optimizer for LoRA params only
    """
    # Inject LoRA
    model, lora_params = inject_lora_into_sit(model, rank=rank, alpha=alpha)
    model = model.to(device)
    
    # Create trainer
    trainer = AdjointMatchingLoRATrainer(
        model=model,
        lora_params=lora_params,
        reward_fn=reward_fn,
        path_type=path_type,
        reward_scale=reward_scale,
        cfg_scale=cfg_scale,
        device=device
    )
    
    # Optimizer for LoRA params only
    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    
    return trainer, optimizer