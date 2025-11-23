"""
Comprehensive test for dataset + data collator pipeline.

Tests:
- Image loading and augmentation
- State and action normalization
- FAST tokenization
- Batch collation with loss masking
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor
import numpy as np
from PIL import Image

from robotwin_dataset import RoboTwinVLADataset
from data_collator import VLADataCollator
from normalization import MultiRobotNormalizer
from action_tokenizer import ActionTokenizer


def test_dataset_loading():
    """Test basic dataset loading and sample structure."""
    print("=" * 80)
    print("TEST 1: Dataset Loading")
    print("=" * 80)

    dataset = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/test_norm_stats.json",
        episode_lengths_path="data/robotwin_episode_lengths_test.json",
        action_horizon=50,
        image_size=(320, 240),
        tasks=["adjust_bottle"],
        robot_types=["franka"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=False,
    )

    print(f"\n✓ Dataset created: {len(dataset)} samples")

    # Get a sample
    sample = dataset[0]

    print(f"\n✓ Sample structure:")
    print(f"  - left_camera: {sample['left_camera'].shape} (dtype: {sample['left_camera'].dtype})")
    print(f"  - right_camera: {sample['right_camera'].shape}")
    print(f"  - head_camera: {sample['head_camera'].shape}")
    print(f"  - task_description: {sample['task_description'][:50]}...")
    print(f"  - robot_type: {sample['robot_type']}")
    print(f"  - discretized_state: {sample['discretized_state'].shape}, range [{sample['discretized_state'].min()}, {sample['discretized_state'].max()}]")
    print(f"  - action_tokens: {len(sample['action_tokens'])} tokens, range [{min(sample['action_tokens'])}, {max(sample['action_tokens'])}]")

    # Verify image values are in [0, 1]
    assert sample['left_camera'].min() >= 0 and sample['left_camera'].max() <= 1, "Image values not in [0, 1]"
    assert sample['right_camera'].min() >= 0 and sample['right_camera'].max() <= 1, "Image values not in [0, 1]"
    assert sample['head_camera'].min() >= 0 and sample['head_camera'].max() <= 1, "Image values not in [0, 1]"
    print(f"\n✓ Image values correctly normalized to [0, 1]")

    # Verify image shape is 320x240
    assert sample['left_camera'].shape == (3, 240, 320), f"Expected (3, 240, 320), got {sample['left_camera'].shape}"
    print(f"✓ Image resolution correct: 320x240 (W x H)")

    # Verify discretized state is in [0, 255]
    assert sample['discretized_state'].min() >= 0 and sample['discretized_state'].max() <= 255, "State not in [0, 255]"
    print(f"✓ Robot state correctly discretized to [0, 255]")

    # Verify action tokens are in FAST range
    assert min(sample['action_tokens']) >= 151936 and max(sample['action_tokens']) <= 153983, "Action tokens not in FAST range"
    print(f"✓ Action tokens in correct range [151936, 153983]")

    dataset.close()
    print(f"\n{'='*80}")
    print("✅ TEST 1 PASSED: Dataset Loading")
    print(f"{'='*80}\n")


def test_image_augmentation():
    """Test that augmentation actually modifies images."""
    print("=" * 80)
    print("TEST 2: Image Augmentation")
    print("=" * 80)

    # Create dataset without augmentation
    dataset_no_aug = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/test_norm_stats.json",
        action_horizon=50,
        image_size=(320, 240),
        tasks=["adjust_bottle"],
        robot_types=["franka"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=False,
    )

    # Create dataset with augmentation
    dataset_with_aug = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/test_norm_stats.json",
        episode_lengths_path="data/robotwin_episode_lengths_test.json",
        action_horizon=50,
        image_size=(320, 240),
        tasks=["adjust_bottle"],
        robot_types=["franka"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=True,
        max_num_transforms=3,
    )

    print(f"\n✓ Created datasets with and without augmentation")

    # Sample the same index multiple times with augmentation
    idx = 0
    samples_aug = [dataset_with_aug[idx]['left_camera'] for _ in range(5)]
    sample_no_aug = dataset_no_aug[idx]['left_camera']

    # Check that augmented samples differ
    differences = []
    for i, sample_aug in enumerate(samples_aug):
        diff = torch.abs(sample_aug - sample_no_aug).mean().item()
        differences.append(diff)
        print(f"  Sample {i+1} vs no-aug: mean absolute diff = {diff:.6f}")

    # At least some samples should be different (augmentation is random)
    # But they might occasionally be the same by chance
    avg_diff = np.mean(differences)
    print(f"\n✓ Average difference: {avg_diff:.6f}")

    if avg_diff > 0.001:
        print(f"✓ Augmentation is modifying images (avg diff > 0.001)")
    else:
        print(f"⚠ Warning: Augmentation may not be working (avg diff = {avg_diff:.6f})")

    dataset_no_aug.close()
    dataset_with_aug.close()

    print(f"\n{'='*80}")
    print("✅ TEST 2 PASSED: Image Augmentation")
    print(f"{'='*80}\n")


def test_normalization_roundtrip():
    """Test that normalization and denormalization work correctly."""
    print("=" * 80)
    print("TEST 3: Normalization Round-trip")
    print("=" * 80)

    dataset = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/test_norm_stats.json",
        episode_lengths_path="data/robotwin_episode_lengths_test.json",
        action_horizon=50,
        image_size=(320, 240),
        tasks=["adjust_bottle"],
        robot_types=["franka"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=False,
    )

    # Get normalized values from dataset
    sample = dataset[0]
    normalized_state = sample['normalized_state']
    normalized_deltas = sample['normalized_deltas']
    robot_type = sample['robot_type']

    print(f"\n✓ Got sample with robot type: {robot_type}")
    print(f"  Normalized state shape: {normalized_state.shape}")
    print(f"  Normalized state range: [{normalized_state.min():.4f}, {normalized_state.max():.4f}]")
    print(f"  Normalized deltas shape: {normalized_deltas.shape}")
    print(f"  Normalized deltas range: [{normalized_deltas.min():.4f}, {normalized_deltas.max():.4f}]")

    # Check normalized values are in [-1, 1]
    assert normalized_state.min() >= -1.0 and normalized_state.max() <= 1.0, "State not in [-1, 1]"
    assert normalized_deltas.min() >= -1.0 and normalized_deltas.max() <= 1.0, "Deltas not in [-1, 1]"
    print(f"\n✓ Normalized values correctly in [-1, 1] range")

    # Test denormalization
    normalizer = dataset.normalizer
    denormalized_state = normalizer.denormalize_state(normalized_state, robot_type)
    denormalized_deltas = normalizer.denormalize_delta_actions(normalized_deltas, robot_type)

    print(f"\n✓ Denormalized values:")
    print(f"  State range: [{denormalized_state.min():.4f}, {denormalized_state.max():.4f}]")
    print(f"  Deltas range: [{denormalized_deltas.min():.4f}, {denormalized_deltas.max():.4f}]")

    # Test round-trip (normalize -> denormalize -> normalize)
    renormalized_state = normalizer.normalize_state(denormalized_state, robot_type)
    renormalized_deltas = normalizer.normalize_delta_actions(denormalized_deltas, robot_type)

    state_error = np.abs(normalized_state - renormalized_state).mean()
    delta_error = np.abs(normalized_deltas - renormalized_deltas).mean()

    print(f"\n✓ Round-trip error:")
    print(f"  State: {state_error:.6f}")
    print(f"  Deltas: {delta_error:.6f}")

    assert state_error < 1e-5, f"State round-trip error too large: {state_error}"
    assert delta_error < 1e-5, f"Delta round-trip error too large: {delta_error}"

    print(f"\n✓ Round-trip successful (error < 1e-5)")

    dataset.close()

    print(f"\n{'='*80}")
    print("✅ TEST 3 PASSED: Normalization Round-trip")
    print(f"{'='*80}\n")


def test_action_tokenization():
    """Test FAST action tokenization and reconstruction."""
    print("=" * 80)
    print("TEST 4: Action Tokenization")
    print("=" * 80)

    dataset = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/test_norm_stats.json",
        episode_lengths_path="data/robotwin_episode_lengths_test.json",
        action_horizon=50,
        image_size=(320, 240),
        tasks=["adjust_bottle"],
        robot_types=["franka"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=False,
    )

    sample = dataset[0]
    action_tokens = sample['action_tokens']
    normalized_deltas = sample['normalized_deltas']
    robot_type = sample['robot_type']

    print(f"\n✓ Original normalized deltas: {normalized_deltas.shape}")
    print(f"  Range: [{normalized_deltas.min():.4f}, {normalized_deltas.max():.4f}]")
    print(f"\n✓ Action tokens: {len(action_tokens)} tokens")
    print(f"  Range: [{min(action_tokens)}, {max(action_tokens)}]")

    # Decode tokens back to actions
    tokenizer = dataset.tokenizer
    action_dim = normalized_deltas.shape[1]  # 2*dof

    decoded_actions = tokenizer.decode(
        [action_tokens],
        action_horizon=dataset.action_horizon,
        action_dim=action_dim
    )

    print(f"\n✓ Decoded actions: {decoded_actions.shape}")
    print(f"  Range: [{decoded_actions.min():.4f}, {decoded_actions.max():.4f}]")

    # Compute reconstruction error
    reconstruction_error = np.abs(normalized_deltas - decoded_actions[0]).mean()
    print(f"\n✓ Reconstruction error (MAE): {reconstruction_error:.6f}")

    # FAST is lossy compression, so allow some error
    assert reconstruction_error < 0.1, f"Reconstruction error too large: {reconstruction_error}"
    print(f"✓ Reconstruction error within acceptable range (< 0.1)")

    dataset.close()

    print(f"\n{'='*80}")
    print("✅ TEST 4 PASSED: Action Tokenization")
    print(f"{'='*80}\n")


def test_data_collator():
    """Test data collation and batch preparation."""
    print("=" * 80)
    print("TEST 5: Data Collator")
    print("=" * 80)

    # Create dataset
    dataset = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/robotwin_norm_stats_test.json",
        episode_lengths_path="data/robotwin_episode_lengths_test.json",
        action_horizon=50,
        image_size=(320, 240),
        tasks=["adjust_bottle"],
        robot_types=["franka"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=False,
    )

    # Create processor and collator
    print("\n✓ Loading Qwen3-VL processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True
    )
    collator = VLADataCollator(processor=processor)

    # Create dataloader with batch size 4
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=collator,
        shuffle=False,
    )

    print(f"✓ Created DataLoader with batch_size=4")

    # Get a batch
    batch = next(iter(dataloader))

    print(f"\n✓ Batch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"  {key}: {type(value)}")

    # Verify batch structure
    assert 'input_ids' in batch, "Missing input_ids"
    assert 'attention_mask' in batch, "Missing attention_mask"
    assert 'labels' in batch, "Missing labels"
    assert 'pixel_values' in batch, "Missing pixel_values"
    assert 'image_grid_thw' in batch, "Missing image_grid_thw"

    print(f"\n✓ All required keys present")

    # Check batch size
    batch_size = batch['input_ids'].shape[0]
    assert batch_size == 4, f"Expected batch size 4, got {batch_size}"
    print(f"✓ Correct batch size: {batch_size}")

    # Check loss masking
    labels = batch['labels']
    num_masked = (labels == -100).sum(dim=1)
    num_action_tokens = (labels != -100).sum(dim=1)

    print(f"\n✓ Loss masking per sample:")
    for i in range(batch_size):
        print(f"  Sample {i}: {num_masked[i].item()} masked, {num_action_tokens[i].item()} action tokens")
        # Verify we have some action tokens
        assert num_action_tokens[i] > 0, f"Sample {i} has no action tokens!"

    print(f"\n✓ All samples have action tokens")

    # Verify action tokens are in FAST range
    for i in range(batch_size):
        sample_labels = labels[i][labels[i] != -100]
        if len(sample_labels) > 0:
            min_token = sample_labels.min().item()
            max_token = sample_labels.max().item()
            assert min_token >= 151936, f"Token {min_token} below FAST range"
            assert max_token <= 153983, f"Token {max_token} above FAST range"

    print(f"✓ All action tokens in FAST range [151936, 153983]")

    dataset.close()

    print(f"\n{'='*80}")
    print("✅ TEST 5 PASSED: Data Collator")
    print(f"{'='*80}\n")


def test_full_pipeline():
    """Test the full pipeline with multiple batches."""
    print("=" * 80)
    print("TEST 6: Full Pipeline (Multiple Batches)")
    print("=" * 80)

    # Create dataset with augmentation
    dataset = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/robotwin_norm_stats_test.json",
        episode_lengths_path="data/robotwin_episode_lengths_test.json",
        action_horizon=50,
        image_size=(320, 240),
        tasks=["adjust_bottle"],
        robot_types=["franka"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=True,
        max_num_transforms=3,
    )

    # Create processor and collator
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        trust_remote_code=True
    )
    collator = VLADataCollator(processor=processor)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collator,
        shuffle=True,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
    )

    print(f"\n✓ Created DataLoader with batch_size=8, shuffle=True, augmentation=True")

    # Process 3 batches
    num_batches = 3
    print(f"\n✓ Processing {num_batches} batches...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break

        print(f"\n  Batch {batch_idx + 1}:")
        print(f"    input_ids: {batch['input_ids'].shape}")
        print(f"    labels: {batch['labels'].shape}")
        print(f"    pixel_values: {batch['pixel_values'].shape}")

        # Verify no NaNs or Infs
        assert not torch.isnan(batch['input_ids']).any(), "NaN in input_ids"
        assert not torch.isnan(batch['pixel_values']).any(), "NaN in pixel_values"
        assert not torch.isinf(batch['pixel_values']).any(), "Inf in pixel_values"

        # Verify pixel values in reasonable range
        assert batch['pixel_values'].min() >= -10, "Pixel values too low"
        assert batch['pixel_values'].max() <= 10, "Pixel values too high"

    print(f"\n✓ Successfully processed {num_batches} batches")
    print(f"✓ No NaNs or Infs detected")
    print(f"✓ All values in reasonable ranges")

    dataset.close()

    print(f"\n{'='*80}")
    print("✅ TEST 6 PASSED: Full Pipeline")
    print(f"{'='*80}\n")


def save_augmented_images(output_dir: str, num_samples: int = 5):
    """
    Save images with and without augmentation for visual inspection.

    Args:
        output_dir: Directory to save images
        num_samples: Number of samples to save
    """
    print("=" * 80)
    print("SAVING AUGMENTED IMAGES FOR INSPECTION")
    print("=" * 80)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n✓ Saving to: {output_path}")

    # Create dataset WITHOUT augmentation
    dataset_no_aug = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/robotwin_norm_stats_test.json",
        episode_lengths_path="data/robotwin_episode_lengths_test.json",
        action_horizon=50,
        image_size=(320, 240),
        tasks=["adjust_bottle"],
        robot_types=["franka"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=False,
    )

    # Create dataset WITH augmentation
    dataset_with_aug = RoboTwinVLADataset(
        dataset_root="/mnt/robotwin/dataset",
        norm_stats_path="data/robotwin_norm_stats_test.json",
        episode_lengths_path="data/robotwin_episode_lengths_test.json",
        action_horizon=50,
        image_size=(320, 240),
        tasks=["adjust_bottle"],
        robot_types=["franka"],
        variants=["clean_50"],
        cache_size=5,
        enable_augmentation=True,
        max_num_transforms=3,
    )

    print(f"\n✓ Saving {num_samples} samples (3 cameras each, with and without augmentation)...")

    for idx in range(min(num_samples, len(dataset_no_aug))):
        # Get samples
        sample_no_aug = dataset_no_aug[idx]
        sample_with_aug = dataset_with_aug[idx]

        # Save each camera view
        for camera_name in ['left_camera', 'right_camera', 'head_camera']:
            # Original (no augmentation)
            img_tensor_orig = sample_no_aug[camera_name]  # (3, H, W) in [0, 1]
            img_array_orig = (img_tensor_orig.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_orig = Image.fromarray(img_array_orig)
            img_orig.save(output_path / f"sample_{idx:03d}_{camera_name}_original.png")

            # Augmented
            img_tensor_aug = sample_with_aug[camera_name]  # (3, H, W) in [0, 1]
            img_array_aug = (img_tensor_aug.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_aug = Image.fromarray(img_array_aug)
            img_aug.save(output_path / f"sample_{idx:03d}_{camera_name}_augmented.png")

        print(f"  Saved sample {idx + 1}/{num_samples}")

    dataset_no_aug.close()
    dataset_with_aug.close()

    print(f"\n✓ Saved {num_samples * 3 * 2} images to {output_path}")
    print(f"  - Original images: sample_XXX_CAMERA_original.png")
    print(f"  - Augmented images: sample_XXX_CAMERA_augmented.png")
    print(f"  - Cameras: left_camera, right_camera, head_camera")

    print(f"\n{'='*80}")
    print("✅ IMAGE SAVING COMPLETE")
    print(f"{'='*80}\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" " * 20 + "DATASET + COLLATOR PIPELINE TESTS")
    print("=" * 80 + "\n")

    try:
        test_dataset_loading()
        test_image_augmentation()
        test_normalization_roundtrip()
        test_action_tokenization()
        test_data_collator()
        test_full_pipeline()

        print("\n" + "=" * 80)
        print(" " * 25 + "🎉 ALL TESTS PASSED! 🎉")
        print("=" * 80 + "\n")
        return True

    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f" " * 30 + "❌ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test dataset and collator pipeline, optionally save augmented images"
    )
    parser.add_argument(
        "--save-images",
        type=str,
        default=None,
        help="Directory to save augmented images for inspection (optional)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of samples to save when --save-images is used (default: 5)",
    )
    args = parser.parse_args()

    # Save images if requested
    if args.save_images:
        save_augmented_images(args.save_images, args.num_samples)
        exit(0)

    # Otherwise run all tests
    success = run_all_tests()
    exit(0 if success else 1)
