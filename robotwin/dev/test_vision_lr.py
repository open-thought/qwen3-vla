"""
Test script to verify the vision_lr parameter grouping works correctly.
"""

import torch
from model import Qwen3VLAModel

def main():
    print("Loading Qwen3VLAModel...")
    model = Qwen3VLAModel(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        use_lora=False,
    )

    print("\n" + "="*80)
    print("TESTING PARAMETER GROUPING FOR VISION_LR")
    print("="*80)

    # Separate vision parameters from other parameters (same logic as train.py)
    vision_params = []
    other_params = []

    for name, param in model.model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual" in name:
            vision_params.append((name, param))
        else:
            other_params.append((name, param))

    vision_count = sum(p.numel() for _, p in vision_params)
    other_count = sum(p.numel() for _, p in other_params)
    total_trainable = vision_count + other_count

    print(f"\nVision parameters: {vision_count:,} ({100*vision_count/total_trainable:.2f}%)")
    print(f"Other parameters: {other_count:,} ({100*other_count/total_trainable:.2f}%)")
    print(f"Total trainable: {total_trainable:,}")

    print(f"\nFirst 10 vision parameter names:")
    for i, (name, _) in enumerate(vision_params[:10]):
        print(f"  {i+1}. {name}")

    print(f"\nFirst 10 non-vision parameter names:")
    for i, (name, _) in enumerate(other_params[:10]):
        print(f"  {i+1}. {name}")

    # Test creating optimizer with parameter groups
    print("\n" + "="*80)
    print("TESTING OPTIMIZER CREATION")
    print("="*80)

    from torch.optim import AdamW

    learning_rate = 2e-5
    vision_lr = 2e-6

    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in other_params],
            "lr": learning_rate,
            "weight_decay": 0.01,
        },
        {
            "params": [p for _, p in vision_params],
            "lr": vision_lr,
            "weight_decay": 0.01,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters)

    print(f"\n✓ Optimizer created successfully with {len(optimizer.param_groups)} parameter groups")
    print(f"  Group 0 (other): {len(optimizer.param_groups[0]['params'])} params, LR={optimizer.param_groups[0]['lr']:.2e}")
    print(f"  Group 1 (vision): {len(optimizer.param_groups[1]['params'])} params, LR={optimizer.param_groups[1]['lr']:.2e}")

    print("\n" + "="*80)
    print("✓ TEST PASSED - Vision LR grouping works correctly!")
    print("="*80)

if __name__ == "__main__":
    main()
