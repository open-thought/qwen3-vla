# Quick Test Guide

This guide walks through testing the training pipeline on a small subset of data for rapid validation.

## Test Configuration

The `config/config_test.yaml` file is pre-configured for quick testing:

- **Robot**: Franka only
- **Task**: adjust_bottle only
- **Variant**: clean_50
- **Steps**: 5,000 (vs 100,000 full training)
- **Batch size**: 4 (vs 8, requires ~20GB GPU memory)
- **Validation**: Every 100 steps (vs 1000)
- **Checkpoints**: Every 500 steps (vs 5000)

Expected runtime: ~1 hour (vs ~27 hours for full training)

## Step 1: Generate Test Normalization Statistics

Extract delta actions and compute statistics for the test subset only:

```bash
# Extract delta actions for Franka + adjust_bottle only (~1 minute)
python extract_delta_actions.py \
    --dataset-root /mnt/robotwin/dataset \
    --output data/robotwin_delta_actions_test.hdf5 \
    --episode-lengths-output data/robotwin_episode_lengths_test.json \
    --action-horizon 16 \
    --robot-types franka \
    --variants clean_50 \
    --tasks adjust_bottle

# Compute normalization statistics for the subset (~10 seconds)
python compute_norm_stats.py \
    --delta-actions data/robotwin_delta_actions_test.hdf5 \
    --output data/robotwin_norm_stats_test.json
```

This creates:
- `data/robotwin_delta_actions_test.hdf5` - Extracted delta actions
- `data/robotwin_episode_lengths_test.json` - Actual episode lengths
- `data/robotwin_norm_stats_test.json` - Normalization statistics for Franka only

The test configuration (`config/config_test.yaml`) is already configured to use these files.

## Step 2: Test the Pipeline

Run the comprehensive pipeline tests:

```bash
python test_pipeline.py
```

This verifies:
1. Dataset loading (images, states, actions)
2. Image augmentation
3. Normalization round-trip
4. FAST action tokenization
5. Data collation and batching
6. Full pipeline with multiple batches

All 6 tests should pass.

## Step 3: Start Test Training

Launch training with the test configuration:

```bash
python train.py --config config/config_test.yaml
```

Monitor the logs for:
- Model loads successfully with extended vocabulary
- Dataset builds index and loads samples
- Training loss decreases over steps
- Validation runs every 100 steps
- Checkpoints save every 500 steps to `checkpoints/qwen3-vla-robotwin-test/`

If WandB is enabled, view real-time metrics at wandb.ai.

## Step 4: Verify Checkpoints

After a few hundred steps, verify checkpoints are being saved:

```bash
ls -lh checkpoints/qwen3-vla-robotwin-test/
```

You should see:
- `step_500/` - Model checkpoint
- `step_500_state.pt` - Training state (optimizer, scheduler)

## Full Dataset Training

Once the test run completes successfully, proceed to full training:

```bash
# Generate full normalization statistics
python extract_delta_actions.py \
    --dataset-root /mnt/robotwin/dataset \
    --output data/robotwin_delta_actions.hdf5 \
    --episode-lengths-output data/robotwin_episode_lengths.json \
    --action-horizon 16

python compute_norm_stats.py \
    --delta-actions data/robotwin_delta_actions.hdf5 \
    --output data/robotwin_norm_stats.json

# Generate default config
python train_config.py

# Start full training
python train.py --config config.yaml
```

## Tips

- **Memory Issues**: Reduce `batch_size` to 2 if you get OOM errors
- **Faster Iteration**: Set `max_steps: 1000` in config for even quicker tests
- **Debug Mode**: Set `enable_wandb: false` to disable logging during debugging
- **LoRA Testing**: Set `use_lora: true` to test efficient fine-tuning mode
