# Qwen3-VLA Training on RoboTwin Dataset

Training a Vision-Language-Action model by extending Qwen3-VL with FAST action tokens for the RoboTwin robotics dataset.

## Prerequisites

- Python 3.10+
- CUDA-capable GPU
- RoboTwin dataset at `/mnt/robotwin/dataset`
- Required packages: `torch`, `transformers`, `peft`, `wandb`, `h5py`, `pillow`, `torchvision`

```bash
pip install torch transformers peft wandb h5py pillow torchvision pyyaml tqdm
```

## Quick Start

### 1. Compute Normalization Statistics

First, extract delta actions from the dataset and compute normalization statistics:

```bash
# Extract delta action chunks (takes ~5-10 minutes for full dataset)
python extract_delta_actions.py \
    --dataset-root /mnt/robotwin/dataset \
    --output data/robotwin_delta_actions.hdf5 \
    --action-horizon 50

# Compute normalization statistics per robot type
python compute_norm_stats.py \
    --delta-actions data/robotwin_delta_actions.hdf5 \
    --output data/robotwin_norm_stats.json
```

This creates `data/robotwin_norm_stats.json` with 1%/99% percentile statistics for:
- Robot states (current joint positions)
- Delta actions (future movements)
- Gripper states

Per robot type: `aloha-agilex`, `arx-x5`, `franka`, `ur5`

### 2. Create Training Configuration

Generate a default configuration file:

```bash
python train_config.py
```

This creates `config.yaml`. Edit it to customize:

```yaml
# Model
model_name: "Qwen/Qwen3-VL-2B-Instruct"

# Dataset
dataset_root: "/mnt/robotwin/dataset"
norm_stats_path: "data/robotwin_norm_stats.json"
action_horizon: 50
image_size: [320, 240]  # Native RoboTwin resolution

# Training
batch_size: 8
gradient_accumulation_steps: 4  # Effective batch size: 32
learning_rate: 2.0e-05
max_steps: 100000

# Image Augmentation
enable_augmentation: true
max_num_transforms: 3

# LoRA (optional - for efficient fine-tuning)
use_lora: false
lora_r: 16
lora_alpha: 32

# WandB
enable_wandb: true
wandb_project: "qwen3-vla-robotwin"
```

### 3. Start Training

```bash
# Full training
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/qwen3-vla-robotwin/step_10000
```

Training will:
- Load Qwen3-VL and extend vocabulary (151,936 → 153,984 tokens)
- Train on RoboTwin episodes with 3 camera views
- Predict FAST-tokenized action chunks (50 timesteps)
- Log to WandB and save checkpoints every 5000 steps

## Testing

Test the complete pipeline before training:

```bash
python test_pipeline.py
```

This runs 6 comprehensive tests:
1. Dataset loading (images, states, actions)
2. Image augmentation
3. Normalization round-trip
4. FAST action tokenization
5. Data collation and batching
6. Full pipeline with multiple batches

## Architecture Overview

```
Input: 3 cameras (320×240) + task text + robot state → Qwen3-VL
Output: FAST tokens (151936-153983) → Delta actions (50 timesteps)
```

### Key Components

- **Dataset** (`robotwin_dataset.py`): Loads RoboTwin episodes, normalizes states/actions
- **Collator** (`data_collator.py`): Prepares batches with loss masking (only on action tokens)
- **Model** (`model.py`): Qwen3-VL with +2048 FAST tokens
- **Normalization** (`normalization.py`): Per-robot quantile normalization
- **Tokenizer** (`action_tokenizer.py`): FAST action space tokenizer

### Data Flow

```
Episode → Extract delta actions → Normalize → Discretize state (0-255)
                                            ↓
                                  Tokenize actions (FAST)
                                            ↓
                              Multi-camera images (320×240)
                                            ↓
                                      VLM Training
```

## Directory Structure

```
robotwin/
├── extract_delta_actions.py    # Extract delta action chunks
├── compute_norm_stats.py        # Compute normalization stats
├── normalization.py             # State/action normalization
├── action_tokenizer.py          # FAST tokenizer wrapper
├── robotwin_dataset.py          # PyTorch dataset
├── data_collator.py             # Batch collation
├── model.py                     # Qwen3-VLA model
├── train_config.py              # Training configuration
├── train.py                     # Training loop
├── test_pipeline.py             # Pipeline tests
└── efficient_batch_loader.py    # Episode indexing

data/
├── robotwin_delta_actions.hdf5  # Extracted delta actions
└── robotwin_norm_stats.json     # Normalization statistics

checkpoints/
└── qwen3-vla-robotwin/          # Training checkpoints
    ├── step_5000/
    ├── step_10000/
    └── best_model/
```

## Model Details

- **Base Model**: Qwen3-VL-2B-Instruct (2B parameters)
- **Vocabulary**: 151,936 → 153,984 (+2,048 FAST tokens)
- **Action Horizon**: 50 future timesteps
- **Image Resolution**: 320×240 (width×height)
- **Action Compression**: ~50×12 values → ~33 tokens (FAST)
- **Training**: Mixed precision (bfloat16), gradient accumulation

## Filtering Options

To train on specific subsets, edit `config.yaml`:

```yaml
# Train only on specific robots
robot_types: ["franka", "ur5"]

# Train only on specific variants
variants: ["clean_50"]

# Train only on specific tasks
tasks: ["adjust_bottle", "click_bell"]
```

## Tips

- **Memory**: Batch size 8 requires ~40GB GPU memory. Reduce if needed.
- **Speed**: ~1 step/sec on A100. Full training (~100k steps) takes ~27 hours.
- **Augmentation**: Enable for training, disable for validation (default).
- **LoRA**: Use `use_lora: true` for faster training with less memory.
- **Validation**: Runs every 1000 steps on 5% of data.

## Monitoring

Training metrics logged to WandB:
- `train/loss`: Training loss (on action tokens only)
- `train/learning_rate`: Current learning rate
- `val/loss`: Validation loss

## Troubleshooting

**Out of memory?**
- Reduce `batch_size` to 4 or 2
- Enable LoRA: `use_lora: true`
- Reduce `max_num_transforms` for augmentation

**Loss not decreasing?**
- Check normalization stats are correct
- Verify action tokens in range [151936, 153983]
- Try higher learning rate (e.g., 5e-5)

**Data loading slow?**
- Increase `num_workers` in config
- Increase `prefetch_factor`
- Use faster storage (SSD/NVMe)

## Citation

```bibtex
@article{qwen3vl2024,
  title={Qwen3-VL: Large Vision Language Model},
  author={Qwen Team},
  year={2024}
}

@article{fast2024,
  title={FAST: Frequency Action Space Tokenizer},
  author={Physical Intelligence},
  year={2024}
}
```
