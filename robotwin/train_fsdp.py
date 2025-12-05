"""
FSDP-enabled training script for Qwen3-VLA on RoboTwin dataset.

Supports distributed training across multiple GPUs using PyTorch FSDP
(Fully Sharded Data Parallelism).

Usage:
    # Single GPU (fallback mode)
    python train_fsdp.py --config config.yaml

    # Multi-GPU with torchrun (recommended)
    torchrun --nproc_per_node=8 train_fsdp.py --config config.yaml

    # Multi-GPU with specific GPUs
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_fsdp.py --config config.yaml
"""

import argparse
import functools
import os
import random
import sys
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
    FullStateDictConfig,
    StateDictType,
)
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.checkpoint.filesystem import FileSystemWriter, FileSystemReader
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.optim import AdamW

try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import random_split
from transformers import get_cosine_schedule_with_warmup, AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLDecoderLayer
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from train_config import TrainingConfig
from robotwin_dataset import RoboTwinVLADataset
from data_collator import VLADataCollator
from model_with_state_history import (
    Qwen3VLAModelWithStateHistory,
    Qwen3VLAWithStateHistoryConfig,
)
from state_encoder import StateEncoderConfig


def create_state_encoder_config(config: TrainingConfig) -> StateEncoderConfig:
    """Create StateEncoderConfig from TrainingConfig."""
    return StateEncoderConfig(
        encoder_type=config.state_encoder_type,
        history_len=config.state_history_len,
        state_dim=14,  # Fixed: 2*6 DoF + 2 grippers for dual-arm robot
        hidden_dim=config.state_encoder_hidden_dim,
        n_output_tokens=config.state_encoder_n_output_tokens,
        dropout=config.state_encoder_dropout,
        conv_channels=config.state_encoder_conv_channels,
        conv_kernel_size=config.state_encoder_conv_kernel_size,
        n_heads=config.state_encoder_n_heads,
        n_layers=config.state_encoder_n_layers,
        rnn_type=config.state_encoder_rnn_type,
        bidirectional=config.state_encoder_bidirectional,
        rnn_layers=config.state_encoder_rnn_layers,
    )


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank():
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


def print_rank0(message):
    """Print only from rank 0."""
    if is_main_process():
        print(message)


def set_seed(seed: int, rank: int = 0):
    """Set seed for reproducibility, with rank offset for different randomness per GPU."""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_distributed():
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        # Single GPU fallback
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        # Set device before init_process_group to avoid warnings
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}")
        )

    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_fsdp_config(config: TrainingConfig):
    """Create FSDP configuration from training config."""
    # Sharding strategy
    sharding_strategies = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
    }
    sharding_strategy = sharding_strategies.get(
        config.fsdp_sharding_strategy, ShardingStrategy.FULL_SHARD
    )

    # Backward prefetch
    backward_prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        None: None,
        "None": None,
    }
    backward_prefetch = backward_prefetch_map.get(
        config.fsdp_backward_prefetch, BackwardPrefetch.BACKWARD_PRE
    )

    # Mixed precision policy for bfloat16
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    # CPU offload
    cpu_offload = CPUOffload(offload_params=True) if config.fsdp_cpu_offload else None

    return {
        "sharding_strategy": sharding_strategy,
        "mixed_precision": mixed_precision_policy,
        "backward_prefetch": backward_prefetch,
        "cpu_offload": cpu_offload,
        "sync_module_states": config.fsdp_sync_module_states,
        "use_orig_params": config.fsdp_use_orig_params,
        "limit_all_gathers": config.fsdp_limit_all_gathers,
    }


def get_fsdp_wrap_policy():
    """Create FSDP auto wrap policy for Qwen2-VL model."""
    # Wrap at the decoder layer level for transformer models
    # This is the standard approach for LLMs
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2VLDecoderLayer},
    )
    return wrap_policy


def wrap_model_with_fsdp(model: nn.Module, config: TrainingConfig, local_rank: int):
    """Wrap model with FSDP for distributed training.

    If NO_SHARD is selected, uses DDP instead (as recommended since NO_SHARD is deprecated).
    For single-GPU training, skips wrapping entirely.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # For single GPU, just move to device without wrapping
    if world_size == 1:
        print_rank0("Single GPU mode - skipping distributed wrapper")
        model = model.to(f"cuda:{local_rank}")
        return model

    # Use DDP instead of FSDP with NO_SHARD (deprecated)
    if config.fsdp_sharding_strategy == "NO_SHARD":
        print_rank0("Using DistributedDataParallel (NO_SHARD is deprecated)")

        # Apply activation checkpointing before moving to GPU (for memory efficiency)
        if config.fsdp_activation_checkpointing:
            # For DDP/regular models, use HuggingFace's built-in gradient checkpointing
            # Handle both raw HuggingFace models and Qwen3VLAModelWithStateHistory wrapper
            inner_model = getattr(model, 'model', model)  # Get inner model if wrapped
            if hasattr(inner_model, 'gradient_checkpointing_enable'):
                inner_model.gradient_checkpointing_enable()
                print_rank0("Enabled model's built-in gradient checkpointing")
                # Verify it's actually enabled
                if hasattr(inner_model, 'is_gradient_checkpointing'):
                    print_rank0(f"  is_gradient_checkpointing: {inner_model.is_gradient_checkpointing}")
                if hasattr(inner_model.config, 'use_cache'):
                    print_rank0(f"  config.use_cache: {inner_model.config.use_cache}")
            else:
                print_rank0("Warning: Model does not support gradient_checkpointing_enable()")

        # Move model to GPU
        model = model.to(f"cuda:{local_rank}")
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # More efficient if all params are used
            gradient_as_bucket_view=True,  # Memory optimization: reuse gradient memory for buckets
        )
        return ddp_model

    fsdp_config = get_fsdp_config(config)
    wrap_policy = get_fsdp_wrap_policy()

    print_rank0(f"Wrapping model with FSDP (strategy: {config.fsdp_sharding_strategy})")

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        device_id=local_rank,
        **fsdp_config,
    )

    # Apply activation checkpointing if requested
    if config.fsdp_activation_checkpointing:
        # For FSDP, use the distributed checkpoint wrapper
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper,
            CheckpointImpl,
            apply_activation_checkpointing,
        )

        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: isinstance(submodule, Qwen2VLDecoderLayer),
        )

    return fsdp_model


def create_distributed_dataloaders(config: TrainingConfig, processor, rank: int, world_size: int):
    """Create training and validation dataloaders with distributed samplers."""
    print_rank0("\nCreating datasets...")

    # Determine state history length (0 if state encoder not enabled)
    state_history_len = config.state_history_len if getattr(config, 'use_state_encoder', False) else 0

    # Create full dataset
    full_dataset = RoboTwinVLADataset(
        dataset_root=config.dataset_root,
        norm_stats_path=config.norm_stats_path,
        valid_timesteps_path=config.valid_timesteps_path,
        action_horizon=config.action_horizon,
        image_size=config.image_size,
        robot_types=config.robot_types,
        variants=config.variants,
        tasks=config.tasks,
        cache_size=10,
        enable_augmentation=config.enable_augmentation,
        max_num_transforms=config.max_num_transforms,
        random_order=config.random_order,
        pad_action_horizon=config.pad_action_horizon,
        tokenizer_type=config.tokenizer_type,
        n_bins=config.n_bins,
        symmetric_delta_norm=config.symmetric_delta_norm,
        binarize_grippers=config.binarize_grippers,
        gripper_open_threshold=config.gripper_open_threshold,
        gripper_closed_threshold=config.gripper_closed_threshold,
        state_dropout_prob=config.state_dropout_prob,
        state_dropout_full_prob=config.state_dropout_full_prob,
        bspline_n_control_points=config.bspline_n_control_points,
        bspline_degree=config.bspline_degree,
        bspline_bounds=config.bspline_bounds,
        bspline_token_order=config.bspline_token_order,
        state_history_len=state_history_len,
        state_history_filter_valid_prob=config.state_history_filter_valid_prob,
    )

    # Split into train/val
    total_size = len(full_dataset)
    val_size = int(total_size * config.validation_split)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )

    print_rank0(f"  Train samples: {len(train_dataset)}")
    print_rank0(f"  Val samples: {len(val_dataset)}")

    # Create validation dataset without augmentation
    val_dataset_no_aug = RoboTwinVLADataset(
        dataset_root=config.dataset_root,
        norm_stats_path=config.norm_stats_path,
        valid_timesteps_path=config.valid_timesteps_path,
        action_horizon=config.action_horizon,
        image_size=config.image_size,
        robot_types=config.robot_types,
        variants=config.variants,
        tasks=config.tasks,
        cache_size=10,
        enable_augmentation=config.val_use_augmentation,
        max_num_transforms=config.max_num_transforms,
        random_order=config.random_order,
        pad_action_horizon=config.pad_action_horizon,
        tokenizer_type=config.tokenizer_type,
        n_bins=config.n_bins,
        symmetric_delta_norm=config.symmetric_delta_norm,
        binarize_grippers=config.binarize_grippers,
        gripper_open_threshold=config.gripper_open_threshold,
        gripper_closed_threshold=config.gripper_closed_threshold,
        state_dropout_prob=0.0,
        state_dropout_full_prob=0.0,
        bspline_n_control_points=config.bspline_n_control_points,
        bspline_degree=config.bspline_degree,
        bspline_bounds=config.bspline_bounds,
        bspline_token_order=config.bspline_token_order,
        state_history_len=state_history_len,
        state_history_filter_valid_prob=0.0,  # No filtering during validation
    )

    # Use same indices for validation
    val_dataset_no_aug = torch.utils.data.Subset(val_dataset_no_aug, val_dataset.indices)

    # Create collator using the passed processor
    collator = VLADataCollator(
        processor=processor,
        tokenizer_type=config.tokenizer_type,
        n_bins=config.n_bins,
        state_reconstruction=config.state_reconstruction,
        state_reconstruction_only_on_dropout=config.state_reconstruction_only_on_dropout,
        include_text_state_in_prompt=config.include_text_state_in_prompt,
        image_dropout_all_prob=config.image_dropout_all_prob,
        image_dropout_prob=config.image_dropout_prob,
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
    ) if world_size > 1 else None

    val_sampler = DistributedSampler(
        val_dataset_no_aug,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if world_size > 1 else None

    # Create dataloaders
    # persistent_workers=True keeps workers alive between batches (faster for heavy collate_fn)
    use_persistent_workers = config.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches for consistent batch sizes across GPUs
        persistent_workers=use_persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset_no_aug,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.num_workers,
        collate_fn=collator,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
        pin_memory=True,
        persistent_workers=use_persistent_workers,
    )

    return train_loader, val_loader, train_sampler


def compute_gradient_norms(model, device, use_fsdp: bool = False) -> dict:
    """
    Compute gradient norms for different parameter groups.

    For FSDP models, gradients are sharded across ranks, so we need to
    sum the squared norms across all ranks before taking the square root.
    For DDP, each rank has full gradients, so no communication is needed.

    Groups tracked:
    - vision: Vision encoder (parameters with "visual" in name)
    - state_encoder: State history encoder (parameters with "state_encoder" in name)
    - embed: Embedding layers (parameters with "embed" or "lm_head" in name)
    - lm: Language model (everything else)

    Args:
        model: The model (can be FSDP, DDP, or unwrapped)
        device: Device for tensor operations
        use_fsdp: Whether the model uses FSDP (requires all_reduce for correct norms)
    """
    vision_grad_sq = 0.0
    lm_grad_sq = 0.0
    embed_grad_sq = 0.0
    state_encoder_grad_sq = 0.0
    vision_param_count = 0
    lm_param_count = 0
    embed_param_count = 0
    state_encoder_param_count = 0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue

        grad_norm_sq = param.grad.detach().norm().item() ** 2

        if "state_encoder" in name:
            state_encoder_grad_sq += grad_norm_sq
            state_encoder_param_count += 1
        elif "visual" in name:
            vision_grad_sq += grad_norm_sq
            vision_param_count += 1
        elif "embed" in name or "lm_head" in name:
            embed_grad_sq += grad_norm_sq
            embed_param_count += 1
        else:
            lm_grad_sq += grad_norm_sq
            lm_param_count += 1

    # For FSDP only: sum squared norms across all ranks (gradients are sharded)
    # For DDP: each rank has full gradients after backward(), no communication needed
    if use_fsdp and dist.is_initialized():
        grad_sq_tensor = torch.tensor(
            [vision_grad_sq, lm_grad_sq, embed_grad_sq, state_encoder_grad_sq],
            device=device
        )
        dist.all_reduce(grad_sq_tensor, op=dist.ReduceOp.SUM)
        vision_grad_sq, lm_grad_sq, embed_grad_sq, state_encoder_grad_sq = grad_sq_tensor.tolist()

    total_grad_sq = vision_grad_sq + lm_grad_sq + embed_grad_sq + state_encoder_grad_sq

    return {
        "grad_norm/vision": vision_grad_sq ** 0.5,
        "grad_norm/lm": lm_grad_sq ** 0.5,
        "grad_norm/embed": embed_grad_sq ** 0.5,
        "grad_norm/state_encoder": state_encoder_grad_sq ** 0.5,
        "grad_norm/total": total_grad_sq ** 0.5,
        "grad_params/vision": vision_param_count,
        "grad_params/lm": lm_param_count,
        "grad_params/embed": embed_param_count,
        "grad_params/state_encoder": state_encoder_param_count,
    }


def train_step(model, batch, device, use_amp, debug=False):
    """Single training step. Returns (loss, embed_norms) where embed_norms may be None."""
    rank = get_rank()

    if debug:
        print(f"[Rank {rank}] Moving batch to device {device}")
        print(f"[Rank {rank}] GPU memory before batch move: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1e9:.2f} GB reserved")

    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    if debug:
        print(f"[Rank {rank}] Batch moved. Starting forward pass...")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"[Rank {rank}]   {k}: shape={v.shape}, device={v.device}, dtype={v.dtype}")

    with autocast('cuda', enabled=use_amp, dtype=torch.bfloat16 if use_amp else torch.float32):
        if debug:
            print(f"[Rank {rank}] Inside autocast, calling model...")
        outputs = model(**batch)
        loss = outputs["loss"]
        if debug:
            print(f"[Rank {rank}] Forward done, loss={loss.item():.4f}")

    if debug:
        print(f"[Rank {rank}] Starting backward...")
        print(f"[Rank {rank}] GPU memory before backward: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1e9:.2f} GB reserved")
    loss.backward()
    if debug:
        print(f"[Rank {rank}] Backward done")
        print(f"[Rank {rank}] GPU memory after backward: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1e9:.2f} GB reserved")

    # Extract embed norms if present
    embed_norms = outputs.get("embed_norms", None)

    return loss.item(), embed_norms


@torch.no_grad()
def validate(model, val_loader, config, device):
    """Run validation across all ranks."""
    model.eval()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation", leave=False) if is_main_process() else val_loader
    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= config.max_val_batches:
            break

        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16 if config.use_amp else torch.float32):
            outputs = model(**batch)
            loss = outputs["loss"]

        total_loss += loss.item()
        num_batches += 1

        if is_main_process() and hasattr(pbar, 'set_postfix'):
            pbar.set_postfix({"val_loss": total_loss / num_batches})

    # Average across all ranks
    if dist.is_initialized():
        total_loss_tensor = torch.tensor([total_loss], device=device)
        num_batches_tensor = torch.tensor([num_batches], device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)
        total_loss = total_loss_tensor.item()
        num_batches = num_batches_tensor.item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    model.train()
    return avg_loss


class AppState(Stateful):
    """Wrapper for model and optimizer state that implements the Stateful protocol for DCP."""

    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def state_dict(self):
        """Get state dict using the new DCP API for proper FSDP FQN handling."""
        if self.optimizer is not None:
            # get_state_dict expects model and optimizers (can be single or list)
            model_state, optimizer_state = get_state_dict(self.model, [self.optimizer])
            return {"model": model_state, "optim": optimizer_state}
        else:
            model_state, _ = get_state_dict(self.model, [])
            return {"model": model_state}

    def load_state_dict(self, state_dict):
        """Load state dict using the new DCP API."""
        if self.optimizer is not None:
            set_state_dict(
                self.model,
                [self.optimizer],
                model_state_dict=state_dict["model"],
                optim_state_dict=state_dict.get("optim", {}),
            )
        else:
            set_state_dict(
                self.model,
                [],
                model_state_dict=state_dict["model"],
                optim_state_dict={},
            )


def save_checkpoint_dcp(model, optimizer, scheduler, step, config, processor=None, is_best=False):
    """
    Save checkpoint using PyTorch Distributed Checkpoint (DCP).

    This is the modern approach that:
    - Each rank saves its own shard (no gathering to rank 0)
    - Supports automatic resharding when loading with different world sizes
    - Much faster for FSDP models since there's no all-gather communication

    For DDP/single-GPU models, saves in HuggingFace format using save_pretrained().
    """
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Handle DDP wrapped models - save in HuggingFace format
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if is_main_process():
            model_path = checkpoint_dir / f"step_{step}"
            model_path.mkdir(parents=True, exist_ok=True)

            # Save model using HuggingFace save_pretrained
            print(f"Saving model to {model_path} (HuggingFace format)...")
            model.module.save_pretrained(model_path)

            # Save processor if available
            if processor is not None:
                processor.save_pretrained(model_path)

            # Save training config as YAML for easy loading during eval
            config.to_yaml(str(model_path / "training_config.yaml"))

            # Save training state (optimizer, scheduler, step)
            torch.save({
                "step": step,
                "config": config.to_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, model_path / "training_state.pt")

            print(f"Checkpoint saved to {model_path}")

            if is_best:
                best_path = checkpoint_dir / "best_model"
                best_path.mkdir(parents=True, exist_ok=True)

                print(f"Saving best model to {best_path} (HuggingFace format)...")
                model.module.save_pretrained(best_path)

                if processor is not None:
                    processor.save_pretrained(best_path)

                # Save training config for best model
                config.to_yaml(str(best_path / "training_config.yaml"))

                torch.save({
                    "step": step,
                    "config": config.to_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, best_path / "training_state.pt")
                print(f"Best model saved to {best_path}")

        if dist.is_initialized():
            dist.barrier()
        return

    # Handle regular (non-wrapped) models - for single GPU mode
    if not isinstance(model, FSDP):
        if is_main_process():
            model_path = checkpoint_dir / f"step_{step}"
            model_path.mkdir(parents=True, exist_ok=True)

            # Save model using HuggingFace save_pretrained
            print(f"Saving model to {model_path} (HuggingFace format)...")
            model.save_pretrained(model_path)

            # Save processor if available
            if processor is not None:
                processor.save_pretrained(model_path)

            # Save training config as YAML for easy loading during eval
            config.to_yaml(str(model_path / "training_config.yaml"))

            # Save training state
            torch.save({
                "step": step,
                "config": config.to_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, model_path / "training_state.pt")

            print(f"Checkpoint saved to {model_path}")

            if is_best:
                best_path = checkpoint_dir / "best_model"
                best_path.mkdir(parents=True, exist_ok=True)

                print(f"Saving best model to {best_path} (HuggingFace format)...")
                model.save_pretrained(best_path)

                if processor is not None:
                    processor.save_pretrained(best_path)

                # Save training config for best model
                config.to_yaml(str(best_path / "training_config.yaml"))

                torch.save({
                    "step": step,
                    "config": config.to_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, best_path / "training_state.pt")
                print(f"Best model saved to {best_path}")
        return

    # Handle FSDP wrapped models using DCP
    # DCP saves sharded checkpoints - each rank saves its own shard
    model_path = checkpoint_dir / f"step_{step}"

    print_rank0(f"Saving FSDP checkpoint to {model_path} using DCP...")

    # Create the AppState wrapper for proper state handling
    app_state = AppState(model, optimizer)

    # Save using DCP - all ranks participate, each saves its own shard
    state_dict = {"app": app_state}
    storage_writer = FileSystemWriter(str(model_path), overwrite=True)
    dcp.save(state_dict, storage_writer=storage_writer)

    # Save scheduler and training metadata (only rank 0)
    if is_main_process():
        # Save training config as YAML for easy loading during eval
        config.to_yaml(str(model_path / "training_config.yaml"))

        torch.save({
            "step": step,
            "config": config.to_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }, model_path / "training_state.pt")

        print(f"Checkpoint saved to {model_path}")

    # Handle best model
    if is_best:
        best_path = checkpoint_dir / "best_model"
        print_rank0(f"Saving best model to {best_path}...")

        # Save best model using DCP
        best_storage_writer = FileSystemWriter(str(best_path), overwrite=True)
        dcp.save(state_dict, storage_writer=best_storage_writer)

        if is_main_process():
            # Save training config for best model
            config.to_yaml(str(best_path / "training_config.yaml"))

            torch.save({
                "step": step,
                "config": config.to_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }, best_path / "training_state.pt")
            print(f"Best model saved to {best_path}")

    # Barrier to ensure all ranks complete saving
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """
    Load checkpoint from various formats.

    Supports:
    - HuggingFace format (saved with save_pretrained())
    - DCP format (PyTorch Distributed Checkpoint) with automatic resharding
    - Legacy format (single model_state.pt file)
    """
    checkpoint_path = Path(checkpoint_path)

    # Check checkpoint format:
    # 1. HuggingFace format (has config.json)
    # 2. DCP format (has .metadata)
    # 3. Legacy format (has model_state.pt)
    is_hf_checkpoint = (checkpoint_path / "config.json").exists()
    is_dcp_checkpoint = (checkpoint_path / ".metadata").exists()

    if is_hf_checkpoint:
        # HuggingFace format - use from_pretrained()
        print_rank0(f"Loading HuggingFace checkpoint from {checkpoint_path}")

        if isinstance(model, FSDP):
            raise ValueError(
                "Cannot load HuggingFace checkpoint into FSDP model. "
                "Use DCP checkpoints for FSDP training."
            )

        # Get the actual model (unwrap DDP if needed)
        actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

        # Check if this is a Qwen3VLAModelWithStateHistory checkpoint
        state_encoder_path = checkpoint_path / "state_encoder.pt"
        is_state_history_checkpoint = state_encoder_path.exists()

        if is_state_history_checkpoint and isinstance(actual_model, Qwen3VLAModelWithStateHistory):
            # Use the model's load_checkpoint method for efficient in-place loading
            print_rank0("Loading Qwen3VLAModelWithStateHistory checkpoint...")
            actual_model.load_checkpoint(str(checkpoint_path))
        else:
            # Standard HuggingFace checkpoint loading
            loaded_model = AutoModelForImageTextToText.from_pretrained(
                str(checkpoint_path),
                dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            actual_model.load_state_dict(loaded_model.state_dict())
            del loaded_model

        # Load training state if available
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location="cpu")
            if optimizer is not None and "optimizer_state_dict" in training_state:
                optimizer.load_state_dict(training_state["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in training_state:
                try:
                    scheduler.load_state_dict(training_state["scheduler_state_dict"])
                except (IndexError, KeyError) as e:
                    print_rank0(f"Warning: Could not load scheduler state: {e}")
            step = training_state.get("step", 0)
            print_rank0(f"Loaded checkpoint from step {step}")
            return step

        print_rank0("No training_state.pt found, starting from step 0")
        return 0

    if not is_dcp_checkpoint:
        # Legacy format - single model_state.pt file
        print_rank0(f"Loading legacy checkpoint from {checkpoint_path}")

        if isinstance(model, FSDP):
            raise ValueError(
                "Cannot load legacy (non-DCP) checkpoint into FSDP model. "
                "Please convert the checkpoint or use a non-FSDP model."
            )

        state_dict_path = checkpoint_path / "model_state.pt"
        if state_dict_path.exists():
            state_dict = torch.load(state_dict_path, map_location="cpu")
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)

        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location="cpu")
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
            return training_state.get("step", 0)

        return 0

    # DCP checkpoint - use DCP to load
    print_rank0(f"Loading DCP checkpoint from {checkpoint_path}")

    # Create AppState wrapper with model and optimizer
    app_state = AppState(model, optimizer)
    state_dict = {"app": app_state}

    # Load using DCP - handles automatic resharding
    storage_reader = FileSystemReader(str(checkpoint_path))
    dcp.load(state_dict, storage_reader=storage_reader)

    # Load scheduler state (saved separately)
    training_state_path = checkpoint_path / "training_state.pt"
    step = 0
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location="cpu")
        # Only load scheduler state if scheduler is provided and compatible
        if scheduler is not None and "scheduler_state_dict" in training_state:
            try:
                scheduler.load_state_dict(training_state["scheduler_state_dict"])
            except (IndexError, KeyError) as e:
                print_rank0(f"Warning: Could not load scheduler state: {e}")
        step = training_state.get("step", 0)

    print_rank0(f"Loaded checkpoint from step {step}")
    return step


def export_checkpoint(config: TrainingConfig, checkpoint_path: str, output_path: str):
    """
    Export a DCP checkpoint to HuggingFace format using save_pretrained().

    This function:
    1. Loads the model and wraps it with FSDP
    2. Loads the DCP checkpoint into the FSDP model
    3. Gathers the full state dict on rank 0
    4. Saves using HuggingFace's save_pretrained()

    Usage:
        torchrun --nproc_per_node=1 train_fsdp.py --config config.yaml \\
            --export checkpoints/step_1000 --export-output exported_model/
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    if not checkpoint_path.exists():
        print_rank0(f"Error: Checkpoint not found at {checkpoint_path}")
        return

    if not (checkpoint_path / ".metadata").exists():
        print_rank0(f"Error: {checkpoint_path} is not a DCP checkpoint (no .metadata file)")
        return

    print_rank0(f"\n{'='*60}")
    print_rank0("Exporting DCP checkpoint to HuggingFace format")
    print_rank0(f"{'='*60}")
    print_rank0(f"Checkpoint: {checkpoint_path}")
    print_rank0(f"Output: {output_path}")

    # Get local rank
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Load model and wrap with FSDP (same as training)
    print_rank0("\nLoading model...")
    model, processor = load_model_for_fsdp(config)

    # Wrap with FSDP
    print_rank0("Wrapping model with FSDP...")
    model = wrap_model_with_fsdp(model, config, local_rank)

    # Create a dummy optimizer (needed for DCP loading)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Create a dummy scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1)

    # Load the DCP checkpoint
    print_rank0(f"\nLoading checkpoint from {checkpoint_path}...")
    step = load_checkpoint(model, optimizer, scheduler, str(checkpoint_path))
    print_rank0(f"Loaded checkpoint from step {step}")

    # Now gather the full state dict and save
    print_rank0("\nGathering full state dict (this may take a moment)...")

    # Check if model is wrapped with FSDP
    if isinstance(model, FSDP):
        # Configure FSDP to gather full state dict on rank 0
        full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
            state_dict = model.state_dict()
    else:
        # Non-FSDP model (single GPU mode) - just get state dict directly
        state_dict = model.state_dict()

    # Only rank 0 saves
    if is_main_process():
        print(f"State dict has {len(state_dict)} keys")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Load a fresh model on CPU to load the state dict into
        print("Loading fresh model for export...")
        export_model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Extend vocabulary to match the trained model
        export_model.resize_token_embeddings(
            new_num_tokens=config.new_vocab_size,
            pad_to_multiple_of=64,
            mean_resizing=True
        )

        # Load the trained weights
        print("Loading trained weights into export model...")
        export_model.load_state_dict(state_dict)

        # Save with HuggingFace's save_pretrained
        print(f"Saving model to {output_path}...")
        export_model.save_pretrained(output_path)

        # Save processor too
        print(f"Saving processor to {output_path}...")
        processor.save_pretrained(output_path)

        # Save training metadata
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            import shutil
            shutil.copy(training_state_path, output_path / "training_state.pt")

        print(f"\n{'='*60}")
        print(f"Export complete!")
        print(f"{'='*60}")
        print(f"Model saved to: {output_path}")
        print(f"\nTo use the exported model:")
        print(f"  from transformers import AutoModelForImageTextToText, AutoProcessor")
        print(f"  model = AutoModelForImageTextToText.from_pretrained('{output_path}')")
        print(f"  processor = AutoProcessor.from_pretrained('{output_path}')")

    # Barrier to ensure all ranks complete
    if dist.is_initialized():
        dist.barrier()

    cleanup_distributed()


def load_model_for_fsdp(config: TrainingConfig):
    """Load model in a way that's compatible with FSDP wrapping."""
    # Each rank loads the model independently (same weights from HF cache)
    # No barriers needed here since we're not syncing yet
    print(f"[Rank {get_rank()}] Loading {config.model_name}...")

    # Load model on CPU first, FSDP will handle device placement
    # Using low_cpu_mem_usage to reduce peak memory during loading
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",  # Use PyTorch's Scaled Dot Product Attention
        # Don't use device_map - FSDP will handle sharding
    )

    # Extend vocabulary (skip if already at target size)
    original_vocab_size = config.original_vocab_size
    new_vocab_size = config.new_vocab_size
    action_vocab_size = new_vocab_size - original_vocab_size

    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    print_rank0(f"Loaded model vocabulary size: {current_vocab_size}")

    if current_vocab_size != new_vocab_size:
        print_rank0(f"Extending vocabulary with {action_vocab_size} action tokens...")
        model.resize_token_embeddings(
            new_num_tokens=new_vocab_size,
            pad_to_multiple_of=64,
            mean_resizing=True
        )
        print_rank0(f"New vocabulary size: {new_vocab_size}")
    else:
        print_rank0(f"Vocabulary already at target size ({new_vocab_size}), skipping resize")

    # Apply LoRA if requested (before FSDP wrapping)
    if config.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        print_rank0(f"Applying LoRA (r={config.lora_r}, alpha={config.lora_alpha})...")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        if is_main_process():
            model.print_trainable_parameters()

    print(f"[Rank {get_rank()}] Model loaded successfully")

    # Load processor (only need on rank 0 but load on all for simplicity)
    processor = AutoProcessor.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )

    return model, processor


def train(config: TrainingConfig):
    """Main FSDP training loop."""
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print_rank0("=" * 60)
    print_rank0("Qwen3-VLA FSDP Training on RoboTwin Dataset")
    print_rank0(f"World size: {world_size} GPUs")
    print_rank0("=" * 60)

    # Set seed
    set_seed(config.seed, rank)

    # Initialize WandB only on main process
    if config.enable_wandb and WANDB_AVAILABLE and is_main_process():
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=config.to_dict(),
        )

    # Load model and processor first (processor needed for dataloaders)
    print_rank0("\nInitializing model...")

    use_state_encoder = getattr(config, 'use_state_encoder', False)

    # Always load fresh model - checkpoint loading happens after FSDP wrapping
    if use_state_encoder:
        # Create Qwen3VLAModelWithStateHistory with state encoder
        state_encoder_config = create_state_encoder_config(config)
        model_config = Qwen3VLAWithStateHistoryConfig(
            model_name=config.model_name,
            new_vocab_size=config.new_vocab_size,
            original_vocab_size=config.original_vocab_size,
            state_encoder_config=state_encoder_config,
        )
        # Use device_map=None so FSDP/DDP can handle device placement
        model_wrapper = Qwen3VLAModelWithStateHistory(model_config, device_map=None)
        model = model_wrapper  # The wrapper will be wrapped by FSDP/DDP
        processor = model_wrapper.processor
        print_rank0(f"State encoder enabled: {config.state_encoder_type}")
        print_rank0(f"  History length: {config.state_history_len}")
        print_rank0(f"  Output tokens: {config.state_encoder_n_output_tokens}")
    else:
        model, processor = load_model_for_fsdp(config)

    # Create dataloaders (needs processor for collator)
    train_loader, val_loader, train_sampler = create_distributed_dataloaders(
        config, processor, rank, world_size
    )

    # Count parameters before FSDP
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank0(f"\nModel parameters:")
    print_rank0(f"  Total: {total_params:,}")
    print_rank0(f"  Trainable: {trainable_params:,}")
    print_rank0(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Wrap with FSDP
    if world_size > 1 or config.use_fsdp:
        model = wrap_model_with_fsdp(model, config, local_rank)
    else:
        model = model.to(device)  # type: ignore[arg-type]

    # Create optimizer
    # Select optimizer class
    use_8bit = getattr(config, 'use_8bit_optimizer', False)
    if use_8bit:
        if not BNB_AVAILABLE:
            print_rank0("Warning: bitsandbytes not available, falling back to standard AdamW")
            print_rank0("  Install with: pip install bitsandbytes")
            optimizer_cls = AdamW
        else:
            print_rank0("Using 8-bit AdamW optimizer (4x less optimizer memory)")
            optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = AdamW

    if config.vision_lr is not None:
        print_rank0(f"\nUsing separate learning rates:")
        print_rank0(f"  Vision tower LR: {config.vision_lr:.2e}")
        print_rank0(f"  Other modules LR: {config.learning_rate:.2e}")

        vision_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "visual" in name:
                vision_params.append(param)
            else:
                other_params.append(param)

        optimizer_grouped_parameters = [
            {"params": other_params, "lr": config.learning_rate, "weight_decay": config.weight_decay},
            {"params": vision_params, "lr": config.vision_lr, "weight_decay": config.weight_decay},
        ]
        optimizer = optimizer_cls(optimizer_grouped_parameters)
    else:
        optimizer = optimizer_cls(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    # Create scheduler
    num_training_steps = config.max_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Resume training state if specified
    start_step = 0
    if config.resume_from_checkpoint:
        print_rank0(f"\nResuming from checkpoint: {config.resume_from_checkpoint}")
        start_step = load_checkpoint(model, optimizer, scheduler, config.resume_from_checkpoint)
        print_rank0(f"Resumed from step {start_step}")

    # Synchronize all ranks before starting training loop
    # This ensures all ranks have completed model/optimizer setup
    if dist.is_initialized():
        print(f"[Rank {rank}] Waiting at barrier before training loop...")
        dist.barrier()
        print(f"[Rank {rank}] Passed barrier, starting training")

    # Training loop
    print_rank0("\n" + "=" * 60)
    print_rank0("Starting training...")
    print_rank0("=" * 60)

    model.train()
    global_step = start_step
    running_loss = 0
    micro_step = 0
    best_val_loss = float('inf')
    epoch = 0

    # Running averages for embed norms
    running_state_embed_norm = 0.0
    running_text_embed_norm = 0.0
    embed_norm_count = 0

    pbar = tqdm(initial=start_step, total=config.max_steps, desc="Training") if is_main_process() else None

    while global_step < config.max_steps:
        # Set epoch for distributed sampler (ensures different shuffling each epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            if global_step >= config.max_steps:
                break

            # Training step (debug first 3 steps)
            debug_step = micro_step < 3

            # For DDP: skip gradient sync during accumulation steps (sync only on last step)
            # This reduces communication overhead by gradient_accumulation_steps factor
            # Note: requires more memory as gradients accumulate locally
            is_accumulation_step = (micro_step + 1) % config.gradient_accumulation_steps != 0
            is_ddp = isinstance(model, torch.nn.parallel.DistributedDataParallel)

            if is_ddp and is_accumulation_step:
                with model.no_sync():
                    loss, embed_norms = train_step(model, batch, device, config.use_amp, debug=debug_step)
            else:
                loss, embed_norms = train_step(model, batch, device, config.use_amp, debug=debug_step)

            running_loss += loss
            micro_step += 1

            # Accumulate embed norms for logging
            if embed_norms is not None:
                running_state_embed_norm += embed_norms.get("state_embed_norm", 0.0)
                running_text_embed_norm += embed_norms.get("text_embed_norm", 0.0)
                embed_norm_count += 1

            # Gradient accumulation complete
            if micro_step % config.gradient_accumulation_steps == 0:
                if debug_step:
                    print(f"[Rank {rank}] Gradient accumulation complete")
                    print(f"[Rank {rank}] Computing gradient norms...")

                # Compute gradient norms before clipping (for logging)
                # use_fsdp=True only for actual FSDP (not DDP with NO_SHARD)
                grad_norms = compute_gradient_norms(model, device, use_fsdp=isinstance(model, FSDP))

                if debug_step:
                    print(f"[Rank {rank}] Gradient norms computed, clipping...")

                # Gradient clipping
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(config.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                if debug_step:
                    print(f"[Rank {rank}] Gradients clipped, stepping optimizer...")
                    print(f"[Rank {rank}] GPU memory before optimizer.step(): {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1e9:.2f} GB reserved")

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if debug_step:
                    print(f"[Rank {rank}] Optimizer step complete")
                    print(f"[Rank {rank}] GPU memory after optimizer.step(): {torch.cuda.memory_allocated(device) / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved(device) / 1e9:.2f} GB reserved")

                global_step += 1
                if pbar:
                    pbar.update(1)

                # Logging
                if global_step % config.wandb_log_interval == 0:
                    num_microbatches = config.wandb_log_interval * config.gradient_accumulation_steps
                    avg_loss = running_loss / num_microbatches

                    # Average loss across ranks
                    if dist.is_initialized():
                        avg_loss_tensor = torch.tensor([avg_loss], device=device)
                        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.AVG)
                        avg_loss = avg_loss_tensor.item()

                    lrs = scheduler.get_last_lr()

                    if pbar:
                        postfix = {
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lrs[0]:.2e}",
                            "gnorm": f"{grad_norms['grad_norm/total']:.2f}",
                        }
                        if len(lrs) > 1:
                            postfix["vlr"] = f"{lrs[1]:.2e}"
                        pbar.set_postfix(postfix)

                    # Print detailed grad norms periodically
                    if is_main_process() and global_step % 5 == 0:
                        print(f"\n  Grad norms - total: {grad_norms['grad_norm/total']:.4f}, "
                              f"vision: {grad_norms['grad_norm/vision']:.4f}, "
                              f"lm: {grad_norms['grad_norm/lm']:.4f}, "
                              f"embed: {grad_norms['grad_norm/embed']:.4f}, "
                              f"state_enc: {grad_norms['grad_norm/state_encoder']:.4f}")

                    if config.enable_wandb and WANDB_AVAILABLE and is_main_process():
                        log_dict = {
                            "train/loss": avg_loss,
                            "train/step": global_step,
                            "train/epoch": epoch,
                            # Gradient norms for debugging different model components
                            "grad_norm/vision": grad_norms["grad_norm/vision"],
                            "grad_norm/lm": grad_norms["grad_norm/lm"],
                            "grad_norm/embed": grad_norms["grad_norm/embed"],
                            "grad_norm/state_encoder": grad_norms["grad_norm/state_encoder"],
                            "grad_norm/total": grad_norms["grad_norm/total"],
                        }
                        if len(lrs) > 1:
                            log_dict["train/learning_rate_other"] = lrs[0]
                            log_dict["train/learning_rate_vision"] = lrs[1]
                        else:
                            log_dict["train/learning_rate"] = lrs[0]

                        # Add embed norms if available
                        if embed_norm_count > 0:
                            log_dict["embed_norm/state"] = running_state_embed_norm / embed_norm_count
                            log_dict["embed_norm/text"] = running_text_embed_norm / embed_norm_count
                            log_dict["embed_norm/ratio"] = (running_state_embed_norm / embed_norm_count) / max(running_text_embed_norm / embed_norm_count, 1e-8)

                        wandb.log(log_dict, step=global_step)

                    running_loss = 0
                    # Reset embed norm accumulators
                    running_state_embed_norm = 0.0
                    running_text_embed_norm = 0.0
                    embed_norm_count = 0

                # Validation
                if global_step % config.val_interval == 0:
                    val_loss = validate(model, val_loader, config, device)
                    print_rank0(f"\nStep {global_step}: Validation loss = {val_loss:.4f}")

                    if config.enable_wandb and WANDB_AVAILABLE and is_main_process():
                        wandb.log({"val/loss": val_loss}, step=global_step)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_checkpoint_dcp(model, optimizer, scheduler, global_step, config, processor=processor, is_best=True)

                # Save checkpoint
                if global_step % config.save_interval == 0:
                    save_checkpoint_dcp(model, optimizer, scheduler, global_step, config, processor=processor)

        epoch += 1

    if pbar:
        pbar.close()

    # Final save
    print_rank0("\nTraining complete!")
    save_checkpoint_dcp(model, optimizer, scheduler, global_step, config, processor=processor)

    if config.enable_wandb and WANDB_AVAILABLE and is_main_process():
        wandb.finish()

    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VLA on RoboTwin with FSDP")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--export", type=str, default=None,
                        help="Export a DCP checkpoint to HuggingFace format (path to DCP checkpoint)")
    parser.add_argument("--export-output", type=str, default=None,
                        help="Output directory for exported model (required with --export)")
    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        config = TrainingConfig.from_yaml(args.config)
    else:
        print(f"Config file {args.config} not found, using default config")
        config = TrainingConfig()

    # Enable FSDP by default for this script
    config.use_fsdp = True

    # Handle export mode
    if args.export:
        if not args.export_output:
            print("Error: --export-output is required when using --export")
            sys.exit(1)

        # Initialize distributed (needed for FSDP even with 1 GPU)
        setup_distributed()

        export_checkpoint(config, args.export, args.export_output)
        return

    if args.resume:
        config.resume_from_checkpoint = args.resume

    if is_main_process():
        print("\nTraining Configuration:")
        print("-" * 60)
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
        print("-" * 60)

    train(config)


if __name__ == "__main__":
    main()
