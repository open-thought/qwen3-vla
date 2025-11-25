"""
Training script for Qwen3-VLA on RoboTwin dataset.

Trains a vision-language-action model by extending Qwen3-VL with FAST action tokens.
"""

import argparse
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

from train_config import TrainingConfig
from model import Qwen3VLAModel
from robotwin_dataset import RoboTwinVLADataset
from data_collator import VLADataCollator


def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_dataloaders(config: TrainingConfig):
    """Create training and validation dataloaders."""
    print("\nCreating datasets...")

    # Create full dataset
    full_dataset = RoboTwinVLADataset(
        dataset_root=config.dataset_root,
        norm_stats_path=config.norm_stats_path,
        episode_lengths_path=config.episode_lengths_path,
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

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Create validation dataset without augmentation
    val_dataset_no_aug = RoboTwinVLADataset(
        dataset_root=config.dataset_root,
        norm_stats_path=config.norm_stats_path,
        episode_lengths_path=config.episode_lengths_path,
        action_horizon=config.action_horizon,
        image_size=config.image_size,
        robot_types=config.robot_types,
        variants=config.variants,
        tasks=config.tasks,
        cache_size=10,
        enable_augmentation=config.val_use_augmentation,  # Usually False
        max_num_transforms=config.max_num_transforms,
        random_order=config.random_order,
        pad_action_horizon=config.pad_action_horizon,
    )

    # Use same indices for validation
    val_dataset_no_aug = torch.utils.data.Subset(val_dataset_no_aug, val_dataset.indices)

    # Create processor and collator
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)
    collator = VLADataCollator(processor=processor)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset_no_aug,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
        prefetch_factor=config.prefetch_factor,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_step(model, batch, use_amp):
    """Single training step."""
    # Move batch to device
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Forward pass with mixed precision (bfloat16)
    with torch.amp.autocast('cuda', enabled=use_amp, dtype=torch.bfloat16 if use_amp else torch.float32):
        outputs = model(**batch)
        loss = outputs["loss"]

    # Backward pass (no scaler needed for bfloat16)
    loss.backward()

    return loss.item()


@torch.no_grad()
def validate(model, val_loader, config, step):
    """Run validation."""
    model.eval()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(val_loader, desc="Validation", leave=False)
    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= config.max_val_batches:
            break

        # Move batch to device
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        with torch.amp.autocast('cuda', enabled=config.use_amp, dtype=torch.bfloat16 if config.use_amp else torch.float32):
            outputs = model(**batch)
            loss = outputs["loss"]

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"val_loss": total_loss / num_batches})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    model.train()
    return avg_loss


def save_checkpoint(model, optimizer, scheduler, step, config, is_best=False):
    """Save training checkpoint."""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = checkpoint_dir / f"step_{step}"
    model.save_pretrained(str(model_path))

    # Save training state
    state_path = checkpoint_dir / f"step_{step}_state.pt"
    torch.save({
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config.to_dict(),
    }, state_path)

    print(f"Checkpoint saved to {model_path}")

    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model"
        model.save_pretrained(str(best_path))
        print(f"Best model saved to {best_path}")


def train(config: TrainingConfig):
    """Main training loop."""
    print("=" * 60)
    print("Qwen3-VLA Training on RoboTwin Dataset")
    print("=" * 60)

    # Set seed
    set_seed(config.seed)

    # Initialize WandB
    if config.enable_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=config.to_dict(),
        )

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)

    # Create model
    print("\nInitializing model...")
    if config.resume_from_checkpoint:
        # Load model from checkpoint
        print(f"Loading model from checkpoint: {config.resume_from_checkpoint}")
        model = Qwen3VLAModel.from_pretrained(config.resume_from_checkpoint)
    else:
        model = Qwen3VLAModel(
            model_name=config.model_name,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
        )

    # Count parameters
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Create optimizer with parameter groups
    # Use different learning rates for vision tower if specified
    if config.vision_lr is not None:
        print(f"\nUsing separate learning rates:")
        print(f"  Vision tower LR: {config.vision_lr:.2e}")
        print(f"  Other modules LR: {config.learning_rate:.2e}")

        # Separate vision parameters from other parameters
        vision_params = []
        other_params = []

        for name, param in model.model.named_parameters():
            if not param.requires_grad:
                continue
            if "visual" in name:
                vision_params.append(param)
            else:
                other_params.append(param)

        vision_count = sum(p.numel() for p in vision_params)
        other_count = sum(p.numel() for p in other_params)
        print(f"  Vision parameters: {vision_count:,}")
        print(f"  Other parameters: {other_count:,}")

        # Create parameter groups with different learning rates
        optimizer_grouped_parameters = [
            {
                "params": other_params,
                "lr": config.learning_rate,
                "weight_decay": config.weight_decay,
            },
            {
                "params": vision_params,
                "lr": config.vision_lr,
                "weight_decay": config.weight_decay,
            },
        ]

        optimizer = AdamW(optimizer_grouped_parameters)
    else:
        # Use single learning rate for all parameters
        optimizer = AdamW(
            model.model.parameters(),
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

    # Resume from checkpoint if specified
    start_step = 0
    if config.resume_from_checkpoint:
        print(f"\nResuming from checkpoint: {config.resume_from_checkpoint}")
        checkpoint_path = Path(config.resume_from_checkpoint)
        # State file is saved as step_{step}_state.pt alongside the checkpoint directory
        state_path = checkpoint_path.parent / f"{checkpoint_path.name}_state.pt"
        if state_path.exists():
            state = torch.load(state_path)
            start_step = state["step"]
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            print(f"Resumed optimizer/scheduler from step {start_step}")
        else:
            # Try to extract step from checkpoint directory name
            match = re.search(r'step_(\d+)', checkpoint_path.name)
            if match:
                start_step = int(match.group(1))
                print(f"Warning: State file {state_path} not found, starting from step {start_step} without optimizer state")
            else:
                print(f"Warning: Could not determine step from checkpoint path, starting from step 0")

    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    model.model.train()
    global_step = start_step
    running_loss = 0
    best_val_loss = float('inf')

    # Infinite data iterator
    train_iterator = iter(train_loader)

    pbar = tqdm(range(start_step, config.max_steps), initial=start_step, total=config.max_steps)

    for step in pbar:
        # Get next batch
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        # Training step
        loss = train_step(model.model, batch, config.use_amp)
        running_loss += loss

        # Gradient accumulation
        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.model.parameters(), config.max_grad_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1

        # Logging
        if (step + 1) % config.wandb_log_interval == 0:
            avg_loss = running_loss / config.wandb_log_interval
            lrs = scheduler.get_last_lr()

            # Display primary learning rate in progress bar
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{lrs[0]:.2e}",
            })

            if config.enable_wandb and WANDB_AVAILABLE:
                log_dict = {
                    "train/loss": avg_loss,
                    "train/step": global_step,
                }

                # Log all learning rates if using parameter groups
                if len(lrs) > 1:
                    log_dict["train/learning_rate_other"] = lrs[0]
                    log_dict["train/learning_rate_vision"] = lrs[1]
                else:
                    log_dict["train/learning_rate"] = lrs[0]

                wandb.log(log_dict, step=global_step)

            running_loss = 0

        # Validation
        if (step + 1) % config.val_interval == 0:
            val_loss = validate(model.model, val_loader, config, global_step)
            print(f"\nStep {global_step}: Validation loss = {val_loss:.4f}")

            if config.enable_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "val/loss": val_loss,
                }, step=global_step)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, scheduler, global_step, config, is_best=True)

        # Save checkpoint
        if (step + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, scheduler, global_step, config)

    # Final save
    print("\nTraining complete!")
    save_checkpoint(model, optimizer, scheduler, global_step, config)

    if config.enable_wandb and WANDB_AVAILABLE:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VLA on RoboTwin")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        config = TrainingConfig.from_yaml(args.config)
    else:
        print(f"Config file {args.config} not found, using default config")
        config = TrainingConfig()

    # Override resume path if provided
    if args.resume:
        config.resume_from_checkpoint = args.resume

    # Print configuration
    print("\nTraining Configuration:")
    print("-" * 60)
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    print("-" * 60)

    # Start training
    train(config)


if __name__ == "__main__":
    main()
