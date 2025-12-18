"""
SiT Fine-tuning with Adjoint Matching
======================================

This module provides reward-based fine-tuning for SiT models using
the Adjoint Matching algorithm.

Usage:
    # From SiT root directory
    from finetune import AdjointMatchingTrainer, AestheticReward
    
    trainer = AdjointMatchingTrainer(
        model=model,
        base_model=base_model,
        reward_fn=AestheticReward()
    )

Or run directly:
    python finetune/finetune_am.py --ckpt path/to/model.pt --reward aesthetic
"""

from .adjoint_matching import (
    AdjointMatchingTrainer,
    AdjointMatchingLoss,
    Interpolant,
    MemorylessSchedule,
    compute_lean_adjoint,
    solve_adjoint_ode,
    load_sit_for_finetuning,
)

from .rewards import (
    AestheticReward,
    ImageRewardScorer,
    QuadraticReward,
    BrightnessReward,
    CompositeReward,
    LatentSpaceReward,
)

__all__ = [
    # Core AM components
    'AdjointMatchingTrainer',
    'AdjointMatchingLoss',
    'Interpolant',
    'MemorylessSchedule',
    'compute_lean_adjoint',
    'solve_adjoint_ode',
    'load_sit_for_finetuning',
    # Reward functions
    'AestheticReward',
    'ImageRewardScorer',
    'QuadraticReward',
    'BrightnessReward',
    'CompositeReward',
    'LatentSpaceReward',
]