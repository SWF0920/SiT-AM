import argparse
import os
import sys
import copy
import json
from datetime import datetime
from glob import glob

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ==============================================================================
# Distributed Utilities
# ==============================================================================

def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    
    return rank, world_size, device


def cleanup_distributed():
    """Clean up distributed training."""
    dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return dist.get_rank() == 0


def print_rank0(msg):
    """Print only from rank 0."""
    if is_main_process():
        print(msg)


def save_checkpoint_rank0(state, path):
    """Save checkpoint only from rank 0."""
    if is_main_process():
        torch.save(state, path)
        print(f"Saved checkpoint to {path}")


def all_reduce_mean(tensor):
    """Average tensor across all processes."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return tensor


# ==============================================================================
# Arguments
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune SiT with Adjoint Matching (Multi-GPU)"
    )
    
    # Model settings
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to SiT checkpoint")
    parser.add_argument("--model", type=str, default="SiT-XL/2",
                        choices=["SiT-XL/2", "SiT-L/2", "SiT-B/2", 
                                "SiT-XL/4", "SiT-L/4", "SiT-B/4"],
                        help="Model architecture")
    parser.add_argument("--path-type", type=str, default="Linear",
                        choices=["Linear", "GVP", "VP"],
                        help="Interpolant path type (must match checkpoint)")
    
    # Fine-tuning method
    parser.add_argument("--method", type=str, default="full",
                        choices=["full", "lora", "control_net", "stochastic"],
                        help="Fine-tuning method")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (only for --method lora)")
    
    # Reward settings
    parser.add_argument("--reward", type=str, default="aesthetic",
                        choices=["aesthetic", "brightness", "quadratic", 
                                "imagereward", "composite", "linear"],
                        help="Reward function to optimize")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Reward scale λ")
    
    # Training settings
    parser.add_argument("--num-iterations", type=int, default=5000,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size PER GPU")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping norm")
    parser.add_argument("--warmup-iters", type=int, default=25,
                        help="Linear warmup iterations")
    
    # Sampling settings
    parser.add_argument("--cfg-scale", type=float, default=4.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num-sampling-steps", type=int, default=50,
                        help="Number of steps for trajectory generation")
    parser.add_argument("--eta", type=float, default=1.0,
                        help="DDIM eta (>=1 for memoryless)")
    
    # Image settings
    parser.add_argument("--image-size", type=int, default=256,
                        help="Image size (256 or 512)")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="Number of classes")
    
    # Logging and checkpoints
    parser.add_argument("--output-dir", type=str, default="./finetune_results",
                        help="Output directory")
    parser.add_argument("--log-every", type=int, default=1,
                        help="Log every N iterations")
    parser.add_argument("--save-every", type=int, default=1000,
                        help="Save checkpoint every N iterations")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging (rank 0 only)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--log-images-every", type=int, default=1)
    parser.add_argument("--num-images-to-log", type=int, default=4)
    parser.add_argument("--log-intermediates-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=100,
                        help="Compute eval_reward every N iterations")
    parser.add_argument("--num-eval-images", type=int, default=16,
                        help="Number of images for eval_reward computation")
    parser.add_argument("--eval-cfg-scale", type=float, default=4.5,
                        help="CFG scale for eval sampling")
    
    return parser.parse_args()


# ==============================================================================
# Model Loading
# ==============================================================================

def load_model(args, device):
    """Load SiT model for distributed training."""
    from models import SiT_models
    
    print_rank0(f"Loading model {args.model} from {args.ckpt}")
    
    latent_size = args.image_size // 8
    
    # Create model
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    )
    
    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location='cpu')
    if 'ema' in ckpt:
        print_rank0("Loading EMA weights")
        state_dict = ckpt['ema']
    elif 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    # Create frozen base model (on same device)
    base_model = copy.deepcopy(model)
    for param in base_model.parameters():
        param.requires_grad = False
    base_model.eval()
    
    print_rank0(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, base_model


def setup_finetuning_method(model, base_model, args, device):
    """Setup fine-tuning method (full, LoRA, or control net)."""
    
    if args.method == "full":
        # Full model fine-tuning
        from finetune.adjoint_matching import AdjointMatchingTrainer
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[device], find_unused_parameters=False)
        
        trainable_params = list(model.parameters())
        
        trainer_class = AdjointMatchingTrainer
        trainer_kwargs = {
            'model': model,
            'base_model': base_model,
        }

    elif args.method == "stochastic":
        # Full model fine-tuning
        from finetune.adjoint_matching_stochastic import AdjointMatchingTrainer
        
        # Wrap model with DDP
        model = DDP(model, device_ids=[device], find_unused_parameters=False)
        
        trainable_params = list(model.parameters())
        
        trainer_class = AdjointMatchingTrainer
        trainer_kwargs = {
            'model': model,
            'base_model': base_model,
        }
        
    elif args.method == "lora":
        # LoRA fine-tuning
        from finetune.adjoint_matching_lora import (
            inject_lora_into_sit,
            AdjointMatchingLoRATrainer
        )
        
        # Inject LoRA (before DDP)
        model, lora_params = inject_lora_into_sit(
            model, 
            rank=args.lora_rank,
            alpha=1.0
        )
        
        # Wrap with DDP
        model = DDP(model, device_ids=[device], find_unused_parameters=True)
        
        trainable_params = lora_params
        
        trainer_class = AdjointMatchingLoRATrainer
        trainer_kwargs = {
            'model': model.module,  # Unwrap for trainer
            'lora_params': lora_params,
        }
        
    elif args.method == "control_net":
        # Control network fine-tuning
        from finetune.adjoint_matching_control_net import (
            AdjointMatchingControlTrainer,
            create_control_network
        )
        
        # Create control network
        latent_size = args.image_size // 8
        control_net = create_control_network(
            arch="conv",
            latent_channels=4,
            spatial_size=latent_size
        ).to(device)
        
        # Wrap control net with DDP
        control_net = DDP(control_net, device_ids=[device])
        
        trainable_params = list(control_net.parameters())
        
        trainer_class = AdjointMatchingControlTrainer
        trainer_kwargs = {
            'control_net': control_net.module,
            'base_model': base_model,
        }
    
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    return model, trainable_params, trainer_class, trainer_kwargs


def load_vae(device):
    """Load VAE for pixel-space rewards."""
    try:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-ema"
        ).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False
        print_rank0("VAE loaded")
        return vae
    except Exception as e:
        print_rank0(f"Warning: Could not load VAE: {e}")
        return None


def get_reward_fn(args, vae, device):
    """Create reward function."""
    from finetune.rewards import (
        AestheticReward,
        QuadraticReward,
        BrightnessReward,
        LatentSpaceReward,
        ImageRewardScorer,
        CompositeReward,
        LinearReward
    )
    
    if args.reward == "aesthetic":
        pixel_reward = AestheticReward(device=device)
        if vae is not None:
            return LatentSpaceReward(pixel_reward, vae)
        return pixel_reward
    
    elif args.reward == "brightness":
        pixel_reward = BrightnessReward(target_brightness=0.6)
        if vae is not None:
            return LatentSpaceReward(pixel_reward, vae)
        return pixel_reward
    
    elif args.reward == "imagereward":
        pixel_reward = ImageRewardScorer(device=device)
        if vae is not None:
            return LatentSpaceReward(pixel_reward, vae)
        return pixel_reward
    
    elif args.reward == "quadratic":
        return QuadraticReward()

    elif args.reward == "linear":
        return LinearReward()
    
    elif args.reward == "composite":
        # Aesthetic + Brightness
        rewards = [
            (AestheticReward(device=device), 1.0),
            (BrightnessReward(target_brightness=0.5), 0.3),
        ]
        pixel_reward = CompositeReward(rewards)
        if vae is not None:
            return LatentSpaceReward(pixel_reward, vae)
        return pixel_reward
    
    else:
        raise ValueError(f"Unknown reward: {args.reward}")


# ==============================================================================
# Learning Rate Schedule
# ==============================================================================

def get_lr(step, args):
    """Learning rate with linear warmup and cosine decay."""
    if step < args.warmup_iters:
        # Linear warmup
        return args.lr * step / args.warmup_iters
    else:
        # Cosine decay
        # progress = (step - args.warmup_iters) / (args.num_iterations - args.warmup_iters)
        # return args.lr * 0.5 * (1 + math.cos(math.pi * progress))
        return args.lr


def set_lr(optimizer, lr):
    """Set learning rate for optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

'''

def decode_and_log_images(trainer, vae, batch_size, latent_size, num_classes, num_images, step, device):
    """Generate and log images to wandb."""
    import wandb
    import numpy as np
    
    z_0 = torch.randn(num_images, 4, latent_size, latent_size, device=device)
    y = torch.randint(0, num_classes, (num_images,), device=device)
    
    with torch.no_grad():
        traj, _ = trainer.generate_trajectory_with_model(z_0, y)
        x_T = traj[-1]
        
        if vae is not None:
            # CRITICAL: Scale latents before decoding (SiT convention)
            imgs = vae.decode(x_T / 0.18215).sample
            
            # Normalize from [-1, 1] to [0, 1], then to [0, 255]
            imgs = ((imgs / 2 + 0.5).clamp(0, 1) * 255).byte()
            imgs = imgs.cpu().permute(0, 2, 3, 1).numpy()
            wandb.log({"samples": [wandb.Image(img) for img in imgs], "step": step})

'''

# ==============================================================================
# Image Logging with Diagnostics
# ==============================================================================

def decode_and_log_images(trainer, vae, batch_size, latent_size, num_classes, num_images, step, device):
    """Generate and log images to wandb with diagnostic comparison."""
    import wandb
    import numpy as np
    
    z_0 = torch.randn(num_images, 4, latent_size, latent_size, device=device)
    y = torch.randint(0, num_classes, (num_images,), device=device)
    
    with torch.no_grad():
        # === Sample from BASE MODEL ===
        traj_base, _ = trainer.generate_trajectory(z_0, y)  # uses base_model
        x_T_base = traj_base[-1]
        
        # === Sample from FINE-TUNED MODEL (same z_0 and y) ===
        traj_model, _ = trainer.generate_trajectory_with_model(z_0, y)
        x_T_model = traj_model[-1]
        
        # === Print diagnostics ===
        print(f"[Diagnostics at step {step}]")
        print(f"  Base x_T: min={x_T_base.min():.3f}, max={x_T_base.max():.3f}, mean={x_T_base.mean():.3f}, std={x_T_base.std():.3f}")
        print(f"  Model x_T: min={x_T_model.min():.3f}, max={x_T_model.max():.3f}, mean={x_T_model.mean():.3f}, std={x_T_model.std():.3f}")
        print(f"  cfg_scale: {trainer.cfg_scale}, num_steps: {trainer.num_sampling_steps}")
        print(f"  Classes: {y.tolist()}")
        
        if vae is not None:
            # Decode base model samples
            imgs_base = vae.decode(x_T_base / 0.18215).sample
            imgs_base = ((imgs_base / 2 + 0.5).clamp(0, 1) * 255).byte()
            imgs_base = imgs_base.cpu().permute(0, 2, 3, 1).numpy()
            
            # Decode fine-tuned model samples  
            imgs_model = vae.decode(x_T_model / 0.18215).sample
            imgs_model = ((imgs_model / 2 + 0.5).clamp(0, 1) * 255).byte()
            imgs_model = imgs_model.cpu().permute(0, 2, 3, 1).numpy()
            
            # === Log side-by-side comparisons ===
            comparison_images = []
            for i in range(num_images):
                # Concatenate base and finetuned horizontally for direct comparison
                combined = np.concatenate([imgs_base[i], imgs_model[i]], axis=1)
                comparison_images.append(
                    wandb.Image(combined, caption=f"class={y[i].item()} | Left: Base, Right: Finetuned")
                )
            
            wandb.log({
                "samples_comparison": comparison_images,
                "samples_base": [wandb.Image(img, caption=f"class={y[i].item()}") for i, img in enumerate(imgs_base)],
                "samples_finetuned": [wandb.Image(img, caption=f"class={y[i].item()}") for i, img in enumerate(imgs_model)],
                "step": step
            })
            
            print(f"  Logged {len(imgs_base)} comparison images")


def compute_eval_reward_and_log_images(trainer, reward_fn, vae, latent_size, num_classes, 
                                        num_images, step, device, eval_cfg_scale=4.5):
    """
    Compute eval_reward with deterministic sampling (eta=0) and specified CFG scale.
    Also log the generated images.
    
    Args:
        trainer: AdjointMatchingTrainer instance
        reward_fn: Reward function to evaluate
        vae: VAE decoder (can be None for latent-space rewards)
        latent_size: Spatial size of latents
        num_classes: Number of classes for conditioning
        num_images: Number of images to generate for evaluation
        step: Current training step
        device: Device to generate on
        eval_cfg_scale: CFG scale for evaluation (default 4.5)
    
    Returns:
        eval_reward_mean: Mean reward across generated images
        eval_reward_std: Std of reward across generated images
    """
    import wandb
    import numpy as np
    
    z_0 = torch.randn(num_images, 4, latent_size, latent_size, device=device)
    y = torch.randint(0, num_classes, (num_images,), device=device)
    
    # Store original settings
    original_cfg_scale = trainer.cfg_scale
    original_eta = trainer.eta
    
    # Set eval settings: deterministic sampling (eta=0) and specified CFG
    trainer.cfg_scale = eval_cfg_scale
    trainer.eta = 0.0  # Deterministic sampling
    
    with torch.no_grad():
        # Generate with fine-tuned model using eval settings
        traj_eval, _ = trainer.generate_trajectory_with_model(z_0, y)
        x_T_eval = traj_eval[-1]
        
        # Compute reward on final latents
        reward_values = reward_fn(x_T_eval)
        eval_reward_mean = reward_values.mean().item()
        eval_reward_std = reward_values.std().item()
        
        print(f"[Eval at step {step}]")
        print(f"  eval_cfg_scale: {eval_cfg_scale}, eta: 0.0 (deterministic)")
        print(f"  eval_reward_mean: {eval_reward_mean:.4f}, eval_reward_std: {eval_reward_std:.4f}")
        print(f"  x_T_eval: min={x_T_eval.min():.3f}, max={x_T_eval.max():.3f}, mean={x_T_eval.mean():.3f}")
        
        # Log images
        if vae is not None:
            imgs_eval = vae.decode(x_T_eval / 0.18215).sample
            imgs_eval = ((imgs_eval / 2 + 0.5).clamp(0, 1) * 255).byte()
            imgs_eval = imgs_eval.cpu().permute(0, 2, 3, 1).numpy()
            
            eval_images = [
                wandb.Image(img, caption=f"class={y[i].item()}, r={reward_values[i].item():.3f}") 
                for i, img in enumerate(imgs_eval)
            ]
            
            wandb.log({
                "eval_samples_cfg4.5": eval_images,
                "eval_reward_mean": eval_reward_mean,
                "eval_reward_std": eval_reward_std,
                "step": step
            })
            
            print(f"  Logged {len(imgs_eval)} eval images (CFG {eval_cfg_scale})")
        else:
            # No VAE, just log the reward metrics
            wandb.log({
                "eval_reward_mean": eval_reward_mean,
                "eval_reward_std": eval_reward_std,
                "step": step
            })
    
    # Restore original settings
    trainer.cfg_scale = original_cfg_scale
    trainer.eta = original_eta
    
    return eval_reward_mean, eval_reward_std


def log_intermediates(loss_dict, step):
    """Log intermediate trajectory/adjoint stats to wandb."""
    import wandb
    import numpy as np
    
    if "trajectory" not in loss_dict:
        return
    
    traj = loss_dict["trajectory"]      # [T+1, B, C, H, W]
    adj = loss_dict["adjoint_traj"]     # [T+1, B, C, H, W]
    times = loss_dict["times"]          # [T+1]
    
    log_data = {"step": step}
    for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = (times - t_val).abs().argmin().item()
        t_actual = times[idx].item()
        log_data[f"x_norm_t{t_actual:.2f}"] = traj[idx].norm().item()
        log_data[f"adj_norm_t{t_actual:.2f}"] = adj[idx].norm().item()
        log_data[f"adj_mean_t{t_actual:.2f}"] = adj[idx].mean().item()
        
        # Adjoint heatmap (first sample, channel-mean)
        heatmap = adj[idx, 0].mean(0).cpu().numpy()
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        log_data[f"adj_heatmap_t{t_actual:.2f}"] = wandb.Image((heatmap * 255).astype(np.uint8))
    
    wandb.log(log_data)


# ==============================================================================
# Main Training Loop
# ==============================================================================

import math

def main():
    args = parse_args()
    
    # Setup distributed
    rank, world_size, device = setup_distributed()
    
    print_rank0("=" * 60)
    print_rank0("SiT + Adjoint Matching (Distributed Training)")
    print_rank0("=" * 60)
    print_rank0(f"World size: {world_size} GPUs")
    print_rank0(f"Batch size per GPU: {args.batch_size}")
    print_rank0(f"Global batch size: {args.batch_size * world_size}")
    print_rank0(f"Method: {args.method}")
    print_rank0(f"Reward: {args.reward}, Scale: {args.reward_scale}")
    print_rank0(f"Eval every: {args.eval_every} iters, CFG: {args.eval_cfg_scale}, Images: {args.num_eval_images}")
    
    # Create output directory
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Wait for directory creation
    dist.barrier()
    
    # Initialize wandb (rank 0 only)
    if args.wandb and is_main_process():
        import wandb
        wandb.init(
            project="sit-adjoint-matching",
            config=vars(args),
            name=f"{args.method}_{args.reward}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Load model
    model, base_model = load_model(args, device)
    
    # Setup fine-tuning method
    model, trainable_params, trainer_class, trainer_kwargs = setup_finetuning_method(
        model, base_model, args, device
    )
    
    print_rank0(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    
    # Load VAE and reward function
    vae = load_vae(device) if args.reward in ["aesthetic", "brightness", "imagereward", "composite"] else None
    reward_fn = get_reward_fn(args, vae, device)
    
    # Create trainer
    latent_size = args.image_size // 8
    
    trainer = trainer_class(
        **trainer_kwargs,
        reward_fn=reward_fn,
        path_type=args.path_type,
        reward_scale=args.reward_scale,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        eta=args.eta,
        device=device
    )
    
    # Optimizer
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        print_rank0(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=f'cuda:{device}')
        
        # Load model state
        if args.method == "full":
            model.module.load_state_dict(ckpt['model'])
        elif args.method == "lora":
            # Load LoRA weights
            for name, param in model.module.named_parameters():
                if 'lora' in name and name in ckpt['model']:
                    param.data = ckpt['model'][name]
        
        optimizer.load_state_dict(ckpt['optimizer'])
        start_step = ckpt['step']
        print_rank0(f"Resumed at step {start_step}")
    
    # Training loop
    print_rank0(f"\nStarting training from step {start_step + 1}")
    print_rank0("-" * 60)
    
    model.train() if args.method == "full" else None
    running_loss = 0.0
    log_steps = 0
    
    for step in range(start_step + 1, args.num_iterations + 1):
        # Update learning rate
        lr = get_lr(step, args)
        set_lr(optimizer, lr)
        
        optimizer.zero_grad()
        
        # Training step
        '''
        loss_dict = trainer.training_step(
            batch_size=args.batch_size,
            latent_size=latent_size,
            num_classes=args.num_classes
        )
        '''

        # Training step
        loss_dict = trainer.training_step(
            batch_size=args.batch_size,
            latent_size=latent_size,
            num_classes=args.num_classes,
            return_intermediates=(args.wandb and step % args.log_intermediates_every == 0)  # ADD THIS
        )
        
        loss = loss_dict['loss']
        loss.backward()
        
        # Gradient clipping
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        
        optimizer.step()
        
        # Accumulate loss for logging
        loss_tensor = loss.detach()
        all_reduce_mean(loss_tensor)
        running_loss += loss_tensor.item()
        log_steps += 1
        
        # Logging
        if step % args.log_every == 0:
            avg_loss = running_loss / log_steps
            
            if is_main_process():
                # --- GPU stats (rank 0 device only) ---
                gpu_mem = torch.cuda.memory_allocated(device) / 1e9
                gpu_reserved = torch.cuda.memory_reserved(device) / 1e9

                log_str = f"Step {step}/{args.num_iterations}: loss={avg_loss:.6f}, lr={lr:.2e}"
                log_str += f", ctrl={loss_dict['control_norm']:.4f}"
                log_str += f", adj={loss_dict['adjoint_norm']:.4f}"

                # reward stats if present
                if "reward_mean" in loss_dict:
                    log_str += f", r_mean={loss_dict['reward_mean']:.4f}"
                if "reward_std" in loss_dict:
                    log_str += f", r_std={loss_dict['reward_std']:.4f}"

                print(log_str)
                
                if args.wandb:
                    import wandb
                    log_data = {
                        "loss": avg_loss,
                        "lr": lr,
                        "control_norm": loss_dict['control_norm'],
                        "adjoint_norm": loss_dict['adjoint_norm'],
                        "gpu_mem_gb": gpu_mem,
                        "gpu_reserved_gb": gpu_reserved,
                        "step": step,
                    }
                    if "reward_mean" in loss_dict:
                        log_data["reward_mean"] = float(loss_dict["reward_mean"])
                    if "reward_std" in loss_dict:
                        log_data["reward_std"] = float(loss_dict["reward_std"])

                    # Log training images (with current cfg_scale)
                    if step % args.log_images_every == 0:
                        decode_and_log_images(trainer, vae, args.batch_size, latent_size, 
                                            args.num_classes, args.num_images_to_log, step, device)
                    
                    # Log intermediates
                    if step % args.log_intermediates_every == 0:
                        log_intermediates(loss_dict, step)    

                    wandb.log(log_data)
            
            running_loss = 0.0
            log_steps = 0
        
        # Eval reward computation (deterministic sampling, CFG 4.5)
        if step % args.eval_every == 0 and is_main_process() and args.wandb:
            compute_eval_reward_and_log_images(
                trainer=trainer,
                reward_fn=reward_fn,
                vae=vae,
                latent_size=latent_size,
                num_classes=args.num_classes,
                num_images=args.num_eval_images,
                step=step,
                device=device,
                eval_cfg_scale=args.eval_cfg_scale
            )
        
        # Save checkpoint
        if step % args.save_every == 0:
            if args.method == "full":
                model_state = model.module.state_dict()
            elif args.method == "lora":
                # Save only LoRA weights
                model_state = {
                    k: v for k, v in model.module.state_dict().items()
                    if 'lora' in k
                }
            elif args.method == "control_net":
                model_state = trainer.control_net.state_dict()
            
            save_checkpoint_rank0({
                'step': step,
                'model': model_state,
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
            }, os.path.join(args.output_dir, f'checkpoint_{step:06d}.pt'))
        
        # Synchronize
        dist.barrier()
    
    # Save final model
    if args.method == "full":
        model_state = model.module.state_dict()
    elif args.method == "lora":
        model_state = {
            k: v for k, v in model.module.state_dict().items()
            if 'lora' in k
        }
    elif args.method == "control_net":
        model_state = trainer.control_net.state_dict()
    
    save_checkpoint_rank0({
        'step': args.num_iterations,
        'model': model_state,
        'args': vars(args),
    }, os.path.join(args.output_dir, 'final_model.pt'))
    
    if args.wandb and is_main_process():
        import wandb
        wandb.finish()
    
    print_rank0("\nTraining complete!")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()