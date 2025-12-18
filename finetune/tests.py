"""
Sanity Test for SiT + Adjoint Matching
=======================================

This script runs a systematic evaluation:
1. Evaluate pre-trained model (baseline)
2. Run 1-step AM fine-tuning + evaluate
3. Run 100-step AM fine-tuning + evaluate  
4. Hyperparameter sweep (optional)

Usage:
    python finetune/tests.py \
        --ckpt ~/SiT/pretrained_models/SiT-XL-2-256x256.pt \
        --num-eval-samples 64 \
        --output-dir ./sanity_results
"""

import argparse
import os
import sys
import copy
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune.adjoint_matching import (
    AdjointMatchingTrainer,
    Interpolant,
)
from finetune.rewards import (
    AestheticReward,
    QuadraticReward,
    BrightnessReward,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Sanity test for SiT + AM")
    
    # Model
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="SiT-XL/2")
    parser.add_argument("--path-type", type=str, default="Linear")
    parser.add_argument("--image-size", type=int, default=256)
    
    # Evaluation
    parser.add_argument("--num-eval-samples", type=int, default=256,
                        help="Samples for evaluation (more = more stable)")
    parser.add_argument("--eval-batch-size", type=int, default=4)
    
    # Training
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)

    # Sampling settings
    parser.add_argument("--num-sampling-steps", type=int, default=20,
                        help="Number of steps for trajectory generation")
    
    # Hyperparam sweep
    parser.add_argument("--sweep", action="store_true",
                        help="Run hyperparameter sweep")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="./sanity_results")
    parser.add_argument("--device", type=str, default="cuda")
    
    # Quick test mode
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with fewer samples")
    
    return parser.parse_args()


class Evaluator:
    """Evaluate model on multiple reward functions."""
    
    def __init__(self, device='cuda', use_vae=True):
        self.device = device
        self.use_vae = use_vae
        
        # Initialize reward functions
        self.rewards = {
            'aesthetic': AestheticReward(device=device),
            'brightness': BrightnessReward(target_brightness=0.5),
            'quadratic': QuadraticReward(),
        }
        
        # Load VAE if needed
        self.vae = None
        if use_vae:
            try:
                from diffusers.models import AutoencoderKL
                self.vae = AutoencoderKL.from_pretrained(
                    "stabilityai/sd-vae-ft-ema"
                ).to(device)
                self.vae.eval()
                print("VAE loaded for pixel-space evaluation")
            except Exception as e:
                print(f"Warning: Could not load VAE: {e}")
                print("Will evaluate in latent space only")
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode VAE latents to images."""
        if self.vae is None:
            return latents
        
        with torch.no_grad():
            # Scale factor for SD VAE
            latents = latents / 0.18215
            images = self.vae.decode(latents).sample
        return images
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        num_samples: int,
        batch_size: int,
        latent_size: int,
        num_classes: int = 1000,
        cfg_scale: float = 4.0,
        num_steps: int = 20
    ) -> dict:
        """
        Generate samples and compute all reward metrics.
        
        Returns:
            dict with reward statistics
        """
        model.eval()
        
        all_rewards = {name: [] for name in self.rewards}
        
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for _ in tqdm(range(num_batches), desc="Evaluating"):
            curr_batch = min(batch_size, num_samples - len(all_rewards['quadratic']))
            
            # Sample noise and labels
            z = torch.randn(curr_batch, 4, latent_size, latent_size, device=self.device)
            y = torch.randint(0, num_classes, (curr_batch,), device=self.device)
            
            # Generate samples (simple Euler)
            x = z.clone()
            dt = 1.0 / num_steps
            
            for i in range(num_steps):
                t = torch.full((curr_batch,), i * dt, device=self.device)
                
                # CFG
                if cfg_scale > 1.0:
                    v_cond = model(x, t, y)
                    v_uncond = model(x, t, torch.zeros_like(y))
                    v = v_uncond + cfg_scale * (v_cond - v_uncond)
                else:
                    v = model(x, t, y)
                
                x = x + dt * v
            
            # Compute rewards
            # Latent-space rewards
            all_rewards['quadratic'].extend(
                self.rewards['quadratic'](x).cpu().tolist()
            )
            
            # Pixel-space rewards (if VAE available)
            if self.vae is not None:
                images = self.decode_latents(x)
                all_rewards['aesthetic'].extend(
                    self.rewards['aesthetic'](images).cpu().tolist()
                )
                all_rewards['brightness'].extend(
                    self.rewards['brightness'](images).cpu().tolist()
                )
            else:
                # Dummy values
                all_rewards['aesthetic'].extend([0.0] * curr_batch)
                all_rewards['brightness'].extend([0.0] * curr_batch)
        
        # Compute statistics
        results = {}
        for name, values in all_rewards.items():
            values = np.array(values[:num_samples])
            results[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }
        
        return results


def load_model(args):
    """Load SiT model."""
    from models import SiT_models
    
    latent_size = args.image_size // 8
    model = SiT_models[args.model](input_size=latent_size, num_classes=1000)
    
    ckpt = torch.load(args.ckpt, map_location='cpu')
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    
    return model.to(args.device)


def train_steps(trainer, optimizer, num_steps, latent_size, batch_size):
    """Run N training steps."""
    trainer.model.train()
    losses = []
    
    for _ in tqdm(range(num_steps), desc=f"Training {num_steps} steps"):
        optimizer.zero_grad()
        loss_dict = trainer.training_step(
            batch_size=batch_size,
            latent_size=latent_size
        )
        loss_dict['loss'].backward()
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss_dict['loss'].item())
    
    return losses


def run_sanity_test(args):
    """Main sanity test routine."""
    
    os.makedirs(args.output_dir, exist_ok=True)
    results = {'args': vars(args), 'experiments': []}
    
    latent_size = args.image_size // 8
    
    # Quick mode adjustments
    if args.quick:
        args.num_eval_samples = 64
        args.eval_batch_size = 8
        print("Quick mode: using 64 eval samples")
    
    # Initialize evaluator
    evaluator = Evaluator(device=args.device)
    
    # =========================================================================
    # Experiment 1: Baseline (pre-trained model)
    # =========================================================================
    print("\n" + "="*60)
    print("Experiment 1: Baseline Evaluation")
    print("="*60)
    
    model = load_model(args)
    
    baseline_results = evaluator.evaluate(
        model=model,
        num_samples=args.num_eval_samples,
        batch_size=args.eval_batch_size,
        latent_size=latent_size,
        cfg_scale=args.cfg_scale
    )
    
    print("\nBaseline Results:")
    for name, stats in baseline_results.items():
        print(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    results['experiments'].append({
        'name': 'baseline',
        'steps': 0,
        'results': baseline_results
    })
    
    # =========================================================================
    # Experiment 2: 1-step AM
    # =========================================================================
    print("\n" + "="*60)
    print("Experiment 2: 1-step AM Fine-tuning")
    print("="*60)
    
    # Reload fresh model
    model = load_model(args)
    base_model = copy.deepcopy(model)
    for p in base_model.parameters():
        p.requires_grad = False
    
    # Use quadratic reward for fast testing (no VAE decode needed)
    reward_fn = QuadraticReward()
    
    trainer = AdjointMatchingTrainer(
        model=model,
        base_model=base_model,
        reward_fn=reward_fn,
        path_type=args.path_type,
        reward_scale=args.reward_scale,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        device=args.device
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Train 1 step
    losses_1 = train_steps(trainer, optimizer, 1, latent_size, args.train_batch_size)
    
    # Evaluate
    results_1step = evaluator.evaluate(
        model=model,
        num_samples=args.num_eval_samples,
        batch_size=args.eval_batch_size,
        latent_size=latent_size,
        cfg_scale=args.cfg_scale
    )
    
    print("\n1-step AM Results:")
    for name, stats in results_1step.items():
        baseline_mean = baseline_results[name]['mean']
        delta = stats['mean'] - baseline_mean
        print(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f} (Δ={delta:+.4f})")
    
    results['experiments'].append({
        'name': '1_step_am',
        'steps': 1,
        'losses': losses_1,
        'results': results_1step
    })
    
    # =========================================================================
    # Experiment 3: 100-step AM
    # =========================================================================
    print("\n" + "="*60)
    print("Experiment 3: 100-step AM Fine-tuning")
    print("="*60)
    
    # Reload fresh model
    model = load_model(args)
    base_model = copy.deepcopy(model)
    for p in base_model.parameters():
        p.requires_grad = False
    
    trainer = AdjointMatchingTrainer(
        model=model,
        base_model=base_model,
        reward_fn=reward_fn,
        path_type=args.path_type,
        reward_scale=args.reward_scale,
        cfg_scale=args.cfg_scale,
        num_sampling_steps=args.num_sampling_steps,
        device=args.device
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Train 100 steps
    losses_100 = train_steps(trainer, optimizer, 100, latent_size, args.train_batch_size)
    
    # Evaluate
    results_100step = evaluator.evaluate(
        model=model,
        num_samples=args.num_eval_samples,
        batch_size=args.eval_batch_size,
        latent_size=latent_size,
        cfg_scale=args.cfg_scale
    )
    
    print("\n100-step AM Results:")
    for name, stats in results_100step.items():
        baseline_mean = baseline_results[name]['mean']
        delta = stats['mean'] - baseline_mean
        print(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f} (Δ={delta:+.4f})")
    
    results['experiments'].append({
        'name': '100_step_am',
        'steps': 100,
        'losses': losses_100,
        'results': results_100step
    })
    
    # =========================================================================
    # Experiment 4: Hyperparameter Sweep (optional)
    # =========================================================================
    if args.sweep:
        print("\n" + "="*60)
        print("Experiment 4: Hyperparameter Sweep")
        print("="*60)
        
        sweep_configs = [
            {'lr': 1e-6, 'reward_scale': 1.0},
            {'lr': 1e-5, 'reward_scale': 1.0},
            {'lr': 1e-5, 'reward_scale': 10.0},
            {'lr': 1e-5, 'reward_scale': 100.0},
            {'lr': 1e-4, 'reward_scale': 10.0},
        ]
        
        for config in sweep_configs:
            print(f"\nSweep: lr={config['lr']}, reward_scale={config['reward_scale']}")
            
            # Fresh model
            model = load_model(args)
            base_model = copy.deepcopy(model)
            for p in base_model.parameters():
                p.requires_grad = False
            
            trainer = AdjointMatchingTrainer(
                model=model,
                base_model=base_model,
                reward_fn=reward_fn,
                path_type=args.path_type,
                reward_scale=config['reward_scale'],
                cfg_scale=args.cfg_scale,
                device=args.device
            )
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
            
            # Train 100 steps
            losses = train_steps(trainer, optimizer, 100, latent_size, args.train_batch_size)
            
            # Evaluate
            sweep_results = evaluator.evaluate(
                model=model,
                num_samples=args.num_eval_samples,
                batch_size=args.eval_batch_size,
                latent_size=latent_size,
                cfg_scale=args.cfg_scale
            )
            
            # Quick summary
            quad_delta = sweep_results['quadratic']['mean'] - baseline_results['quadratic']['mean']
            print(f"  → quadratic: Δ={quad_delta:+.4f}")
            
            results['experiments'].append({
                'name': f"sweep_lr{config['lr']}_rs{config['reward_scale']}",
                'config': config,
                'steps': 100,
                'losses': losses,
                'results': sweep_results
            })
    
    # =========================================================================
    # Save Results
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.output_dir, f"sanity_test_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    print("\nQuadratic Reward (higher = better, target is 0):")
    for exp in results['experiments']:
        name = exp['name']
        mean = exp['results']['quadratic']['mean']
        if name == 'baseline':
            print(f"  {name:30s}: {mean:.4f}")
        else:
            delta = mean - results['experiments'][0]['results']['quadratic']['mean']
            print(f"  {name:30s}: {mean:.4f} (Δ={delta:+.4f})")
    
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    args = parse_args()
    
    print("="*60)
    print("SiT + Adjoint Matching Sanity Test")
    print("="*60)
    print(f"Checkpoint: {args.ckpt}")
    print(f"Model: {args.model}")
    print(f"Eval samples: {args.num_eval_samples}")
    print(f"Training batch size: {args.train_batch_size}")
    print(f"LR: {args.lr}, Reward scale: {args.reward_scale}")
    
    results = run_sanity_test(args)
    
    print("\nSanity test complete!")


if __name__ == "__main__":
    main()