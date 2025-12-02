"""
Training configuration for Qwen3-VLA on RoboTwin dataset.
"""

from dataclasses import dataclass, field
from typing import Optional
import yaml
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for VLA training."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"

    # FSDP / Distributed training configuration
    use_fsdp: bool = False  # Enable Fully Sharded Data Parallel for multi-GPU training
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_cpu_offload: bool = False  # Offload parameters to CPU (saves GPU memory but slower)
    fsdp_backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST, or None
    fsdp_state_dict_type: str = "FULL_STATE_DICT"  # FULL_STATE_DICT, LOCAL_STATE_DICT, SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool = False  # Enable activation checkpointing to save memory
    fsdp_sync_module_states: bool = True  # Sync module states across ranks at init
    fsdp_use_orig_params: bool = True  # Required for optimizer state dict compatibility
    fsdp_limit_all_gathers: bool = True  # Limit concurrent all-gathers for memory efficiency
    original_vocab_size: int = 151936
    new_vocab_size: int = 152191  # 151936 + 255 bins (default for bspline/bin tokenizer)
    tokenizer_type: str = "bspline"  # "bspline" (B-spline trajectory tokenizer) or "bin" (OpenVLA-style bins)
    n_bins: int = 255  # Number of bins for quantization (255 for exact zero with symmetric bounds)
    symmetric_delta_norm: bool = False  # Use symmetric normalization for deltas (0 maps to exactly 0)

    # B-spline tokenizer configuration
    bspline_n_control_points: int = 8  # Number of B-spline control points per DoF
    bspline_degree: int = 4  # B-spline polynomial degree
    bspline_bounds: tuple[float, float] = (-1.0, 1.0)  # Bounds for control point values
    bspline_token_order: str = "basis_first"  # Token ordering: "basis_first" or "joint_first"

    # Gripper binarization (converts continuous gripper values to binary 0/1)
    binarize_grippers: bool = False  # If True, binarize gripper values during training
    gripper_open_threshold: float = 0.95  # Values > this are considered open (1.0)
    gripper_closed_threshold: float = 0.05  # Values < this are considered closed (0.0)

    # Dataset configuration
    dataset_root: str = "/mnt/robotwin/dataset"
    norm_stats_path: str = "data/robotwin_norm_stats.json"
    episode_lengths_path: str = "data/robotwin_episode_lengths.json"
    valid_timesteps_path: Optional[str] = None
    action_horizon: int = 50
    pad_action_horizon: bool = False  # Pad action sequences by repeating final state (True) or use variable-length (False)
    image_size: tuple[int, int] = (320, 240)  # (width, height) - native RoboTwin resolution

    # Data filtering (None = use all)
    robot_types: Optional[list[str]] = None
    variants: Optional[list[str]] = None
    tasks: Optional[list[str]] = None

    # Image augmentation
    enable_augmentation: bool = True
    max_num_transforms: int = 3
    random_order: bool = False
    val_use_augmentation: bool = False  # Disable augmentation for validation

    # State dropout (to prevent shortcut learning via state)
    # When enabled, randomly replaces state values with "?" in the prompt
    # This forces the model to rely on images instead of state shortcuts
    state_dropout_prob: float = 0.0  # Probability of dropping out each state value (0.0 = disabled)
    state_dropout_full_prob: float = 0.0  # Probability of dropping ALL state values at once (0.0 = disabled)

    # State reconstruction (auxiliary task to force visual understanding)
    # When enabled with state dropout, appends the full uncorrupted state after actions
    # Sequence: [masked state prompt] → [actions] → [EOT] → [state tokens] → [EOT]
    # This forces the model to infer state from images when state is dropped out
    state_reconstruction: bool = False  # Enable state reconstruction auxiliary task
    state_reconstruction_only_on_dropout: bool = True  # Only add reconstruction when state was dropped

    # Training split
    validation_split: float = 0.05

    # Training hyperparameters
    batch_size: int = 8
    gradient_accumulation_steps: int = 4  # Effective batch size: 32
    learning_rate: float = 2e-5
    vision_lr: Optional[float] = None  # Optional separate LR for vision tower, defaults to learning_rate
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    max_steps: int = 100000
    warmup_steps: int = 1000
    use_8bit_optimizer: bool = False  # Use 8-bit Adam from bitsandbytes (4x less optimizer memory)

    # LoRA configuration
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])

    # Checkpointing and evaluation
    checkpoint_dir: str = "checkpoints/qwen3-vla-robotwin"
    save_interval: int = 5000
    val_interval: int = 1000
    max_val_batches: int = 100  # Limit validation to prevent long eval

    # DataLoader configuration
    num_workers: int = 4
    prefetch_factor: int = 2

    # WandB logging
    enable_wandb: bool = True
    wandb_project: str = "qwen3-vla-robotwin"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_log_interval: int = 10

    # Mixed precision training (always uses bfloat16)
    use_amp: bool = True

    # Seed for reproducibility
    seed: int = 42

    # Resume training
    resume_from_checkpoint: Optional[str] = None

    def __post_init__(self):
        """Validate and setup configuration."""
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Validate tokenizer type
        if self.tokenizer_type not in ("bspline", "bin"):
            raise ValueError(f"tokenizer_type must be 'bspline' or 'bin', got {self.tokenizer_type}")

        # Validate n_bins
        if self.n_bins <= 0:
            raise ValueError(f"n_bins must be positive, got {self.n_bins}")

        # Compute vocab size: original + action token bins, rounded up to multiple of 64
        # (improves GPU tensor core utilization)
        raw_vocab_size = self.original_vocab_size + self.n_bins
        self.new_vocab_size = ((raw_vocab_size + 63) // 64) * 64

        # Validate action horizon
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")

        # Validate effective batch size
        effective_batch_size = self.batch_size * self.gradient_accumulation_steps
        if effective_batch_size < 8:
            print(f"Warning: Effective batch size is {effective_batch_size}, which may be too small")

    @property
    def effective_batch_size(self) -> int:
        """Get effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        # Convert to dict, handling special types
        config_dict = {}
        for key, value in self.__dict__.items():
            if value is not None:
                config_dict[key] = value

        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


def create_default_config(output_path: str = "config.yaml"):
    """Create a default configuration file."""
    config = TrainingConfig()
    config.to_yaml(output_path)
    print(f"Default configuration saved to {output_path}")


if __name__ == "__main__":
    # Create example configuration
    create_default_config("config.yaml")

    # Test loading
    config = TrainingConfig.from_yaml("config.yaml")
    print("\nLoaded configuration:")
    print(f"  Model: {config.model_name}")
    print(f"  Effective batch size: {config.effective_batch_size}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  Action horizon: {config.action_horizon}")
    print(f"  Image size: {config.image_size}")
    print(f"  Augmentation: {config.enable_augmentation}")
    print(f"  LoRA enabled: {config.use_lora}")
    print(f"  Checkpoint dir: {config.checkpoint_dir}")
