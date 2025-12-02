#!/usr/bin/env python3
"""
Convert DDP/single-GPU checkpoints (model_state.pt) to HuggingFace format.

DDP checkpoints saved by train_fsdp.py in NO_SHARD mode consist of:
- model_state.pt: Raw PyTorch state dict from model.module.state_dict()
- training_state.pt: Training metadata (optimizer, scheduler, step, config)

This script converts them to HuggingFace format using save_pretrained(),
which creates the standard HuggingFace model structure (config.json, model.safetensors, etc.).

Usage:
    python convert_checkpoint.py checkpoints/step_1000 exported_model/

    # With explicit config
    python convert_checkpoint.py checkpoints/step_1000 exported_model/ --config config.yaml

    # Batch convert multiple checkpoints
    python convert_checkpoint.py checkpoints/step_* exported_models/ --batch
"""

import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_config_from_checkpoint(checkpoint_path: Path) -> dict:
    """Load config from training_state.pt if available."""
    training_state_path = checkpoint_path / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        return training_state.get("config", {})
    return {}


def convert_ddp_checkpoint(
    checkpoint_path: str,
    output_path: str,
    model_name: str = None,
    new_vocab_size: int = None,
    copy_training_state: bool = True,
):
    """
    Convert a DDP checkpoint to HuggingFace format.

    Args:
        checkpoint_path: Path to checkpoint directory containing model_state.pt
        output_path: Output directory for HuggingFace model
        model_name: Base model name (e.g., "Qwen/Qwen3-VL-2B-Instruct")
        new_vocab_size: Extended vocabulary size (if different from base model)
        copy_training_state: Whether to copy training_state.pt to output
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)

    # Validate checkpoint
    model_state_path = checkpoint_path / "model_state.pt"
    if not model_state_path.exists():
        raise FileNotFoundError(
            f"model_state.pt not found in {checkpoint_path}. "
            "This script is for DDP/single-GPU checkpoints, not FSDP DCP checkpoints. "
            "For DCP checkpoints, use: python train_fsdp.py --export <path> --export-output <output>"
        )

    # Check if this is a DCP checkpoint (should not be)
    if (checkpoint_path / ".metadata").exists():
        raise ValueError(
            f"{checkpoint_path} appears to be a DCP checkpoint (has .metadata file). "
            "Use train_fsdp.py --export for DCP checkpoints instead."
        )

    # Load config from checkpoint if available
    ckpt_config = load_config_from_checkpoint(checkpoint_path)

    # Use provided values or fall back to checkpoint config
    if model_name is None:
        model_name = ckpt_config.get("model_name", "Qwen/Qwen3-VL-2B-Instruct")
    if new_vocab_size is None:
        new_vocab_size = ckpt_config.get("new_vocab_size", 153984)

    print(f"\n{'='*60}")
    print("Converting DDP checkpoint to HuggingFace format")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Base model: {model_name}")
    print(f"Vocab size: {new_vocab_size}")

    # Load the checkpoint state dict
    print(f"\nLoading checkpoint from {model_state_path}...")
    state_dict = torch.load(model_state_path, map_location="cpu", weights_only=True)
    print(f"  Loaded state dict with {len(state_dict)} keys")

    # Load base model
    print(f"\nLoading base model: {model_name}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Check if we need to resize embeddings
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"  Base model vocab size: {current_vocab_size}")

    # Determine actual vocab size from checkpoint
    # Look for the lm_head weight to determine actual size
    lm_head_key = None
    for key in state_dict.keys():
        if "lm_head.weight" in key:
            lm_head_key = key
            break

    if lm_head_key:
        ckpt_vocab_size = state_dict[lm_head_key].shape[0]
        print(f"  Checkpoint vocab size (from lm_head): {ckpt_vocab_size}")
        if ckpt_vocab_size != new_vocab_size:
            print(f"  Adjusting new_vocab_size from {new_vocab_size} to {ckpt_vocab_size}")
            new_vocab_size = ckpt_vocab_size

    if current_vocab_size != new_vocab_size:
        print(f"\nResizing embeddings: {current_vocab_size} -> {new_vocab_size}")
        model.resize_token_embeddings(
            new_num_tokens=new_vocab_size,
            pad_to_multiple_of=64,
            mean_resizing=True
        )

    # Load the trained weights
    print("\nLoading trained weights...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"  Warning: Missing keys ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"  Warning: Unexpected keys ({len(unexpected_keys)}): {unexpected_keys[:5]}...")

    if not missing_keys and not unexpected_keys:
        print("  All keys matched perfectly!")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Save with HuggingFace's save_pretrained
    print(f"\nSaving model to {output_path}...")
    model.save_pretrained(output_path)

    # Save processor
    print(f"Saving processor to {output_path}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor.save_pretrained(output_path)

    # Copy training metadata if requested
    if copy_training_state:
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            import shutil
            shutil.copy(training_state_path, output_path / "training_state.pt")
            print(f"Copied training_state.pt to {output_path}")

    # Print step info if available
    training_state_path = checkpoint_path / "training_state.pt"
    if training_state_path.exists():
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        step = training_state.get("step", "unknown")
        print(f"\nCheckpoint was from training step: {step}")

    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"{'='*60}")
    print(f"\nTo use the converted model:")
    print(f"  from transformers import AutoModelForImageTextToText, AutoProcessor")
    print(f"  model = AutoModelForImageTextToText.from_pretrained('{output_path}')")
    print(f"  processor = AutoProcessor.from_pretrained('{output_path}')")

    return output_path


def batch_convert(
    checkpoint_paths: list[str],
    output_dir: str,
    model_name: str = None,
    new_vocab_size: int = None,
):
    """Convert multiple checkpoints to a single output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ckpt_path in checkpoint_paths:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.is_dir():
            print(f"Skipping {ckpt_path}: not a directory")
            continue
        if not (ckpt_path / "model_state.pt").exists():
            print(f"Skipping {ckpt_path}: no model_state.pt found")
            continue

        # Create output subdirectory with same name
        output_path = output_dir / ckpt_path.name

        try:
            convert_ddp_checkpoint(
                str(ckpt_path),
                str(output_path),
                model_name=model_name,
                new_vocab_size=new_vocab_size,
            )
        except Exception as e:
            print(f"Error converting {ckpt_path}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Convert DDP checkpoints to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert single checkpoint (auto-detects model_name and vocab_size from checkpoint)
    python convert_checkpoint.py checkpoints/step_1000 exported_model/

    # With config file (uses model_name and vocab_size from config)
    python convert_checkpoint.py checkpoints/step_1000 exported_model/ --config config.yaml

    # Batch convert (creates subdirectories in output)
    python convert_checkpoint.py checkpoints/step_* exported_models/ --batch
        """,
    )
    parser.add_argument(
        "checkpoint_path",
        nargs="+",
        help="Path(s) to checkpoint directory containing model_state.pt",
    )
    parser.add_argument(
        "output_path",
        help="Output directory for converted model(s)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml to read model_name and vocab_size (optional, auto-detected from checkpoint if not provided)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override base model name (default: from config or checkpoint)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Override vocabulary size (default: from config or checkpoint)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode: convert multiple checkpoints to subdirectories",
    )

    args = parser.parse_args()

    # Priority: command-line args > config file > checkpoint's training_state.pt
    model_name = args.model_name
    new_vocab_size = args.vocab_size

    if args.config:
        from train_config import TrainingConfig
        config = TrainingConfig.from_yaml(args.config)
        if model_name is None:
            model_name = config.model_name
        if new_vocab_size is None:
            new_vocab_size = config.new_vocab_size

    if args.batch or len(args.checkpoint_path) > 1:
        batch_convert(
            args.checkpoint_path,
            args.output_path,
            model_name=model_name,
            new_vocab_size=new_vocab_size,
        )
    else:
        convert_ddp_checkpoint(
            args.checkpoint_path[0],
            args.output_path,
            model_name=model_name,
            new_vocab_size=new_vocab_size,
        )


if __name__ == "__main__":
    main()
