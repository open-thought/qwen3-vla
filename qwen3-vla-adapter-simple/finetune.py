from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import argparse
import os
import random
import torch
import yaml
import numpy as np
from tqdm import tqdm

# Disable tokenizers parallelism to avoid fork warnings with DataLoader
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.transforms import (
    ImageTransforms,
    ImageTransformsConfig,
)
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoProcessor
from qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLConfig,
    Qwen3VLPreTrainedModel,
)
import torch.nn as nn
from bridge_attention import ProprioProjector, ActionHead

from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.optimization import (
    get_cosine_schedule_with_warmup,
)


@dataclass
class ExperimentConfig:
    """Configuration for VLA training experiment"""

    # General configuration
    seed: int = 42

    # Model configuration
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    device: str = "cuda:0"

    # Dataset configuration
    dataset_name: str = "HuggingFaceVLA/libero"
    action_chunk_len: int = 8
    train_split: float = 0.8  # Fraction of episodes for training
    val_batch_size: Optional[int] = None  # If None, uses batch_size

    # Data augmentation
    enable_transforms: bool = True
    max_num_transforms: int = 3
    random_order: bool = False
    val_use_augmentation: bool = False  # Whether to use augmentation on validation set

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    grad_accumulation_steps: int = 1
    max_steps: int = 150_000
    num_warmup_steps: int = 100

    # Checkpoint and validation configuration
    save_checkpoint_interval: int = 5000
    val_interval: int = 1000  # Run validation every N steps
    max_val_batches: int = 256  # Maximum number of batches to evaluate during validation
    output_dir: str = "/data2/checkpoints"
    experiment_name: str = "qwen3_vla_adapter"
    checkpoint_path: Optional[str] = None

    # Wandb configuration
    enable_wandb: bool = True
    wandb_project: str = "qwen3-vla"
    wandb_entity: Optional[str] = None
    wandb_log_interval: int = 10
    print_actions_interval: int = 250  # Print target/predicted actions every N steps

    # DataLoader configuration
    num_workers: int = 1

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExperimentConfig":
        """Load configuration from YAML file"""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        with open(yaml_path, "w") as f:
            yaml.dump(self.__dict__, f, default_flow_style=False, sort_keys=False)


def set_seed(seed: int) -> None:
    """Set seed for reproducibility across all random number generators"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ActionNormalization:
    def __init__(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor):
        self.set_range(lower_bound, upper_bound)

    def set_range(self, lower_bound: torch.Tensor, upper_bound: torch.Tensor) -> None:
        assert lower_bound.le(
            upper_bound
        ).all(), "elements of lower_bound must not be larger than their counterparts in upper_bound"
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.dynamic_range = self.upper_bound - self.lower_bound + 1e-8

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        actions = torch.clamp(actions, self.lower_bound, self.upper_bound)
        return 2 * (actions - self.lower_bound) / self.dynamic_range - 1

    def unnormalize(self, normalized_actions: torch.Tensor) -> torch.Tensor:
        return (normalized_actions + 1) * 0.5 * self.dynamic_range + self.lower_bound


@dataclass
class DatasetCollator:
    dataset: LeRobotDataset
    action_chunk_len: int
    action_normalization: ActionNormalization

    def load_future_actions(self, index: int) -> torch.Tensor:
        episode_index_column = self.dataset.hf_dataset["episode_index"]
        action_column = self.dataset.hf_dataset["action"]

        episode_indices = episode_index_column[index : index + self.action_chunk_len]
        actions = action_column[index : index + self.action_chunk_len]
        assert len(actions) > 0 and len(episode_indices) == len(actions)

        action_chunk_list: list[torch.Tensor] = []

        for i in range(self.action_chunk_len):
            if i < len(episode_indices) and episode_indices[i] == episode_indices[0]:
                last_action = actions[i]
                last_action = self.action_normalization.normalize(last_action)
            # if episode changed or we are at end of dataset, repeat last action
            action_chunk_list.append(last_action)

        return torch.stack(action_chunk_list)

    def __call__(self, xs: list[dict]) -> dict[str, Any]:

        batch = {}

        keys = (
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        )

        for k in keys:
            batch[k] = torch.stack([y[k] for y in xs])

        # special handling for action_chunks
        batch["action_chunk"] = torch.stack(
            [self.load_future_actions(y["index"].item()) for y in xs]
        )

        # fetch task language descriptions
        tasks = []
        for x in xs:
            task_index = x["task_index"].item()
            tasks.append(self.dataset.meta.episodes["tasks"][task_index])
        batch["task"] = tasks

        return batch


def prepare_batch(processor, batch):
    images = batch["observation.images.image"]
    images2 = batch["observation.images.image2"]
    tasks = batch["task"]
    action_chunks = batch["action_chunk"]

    batch_size = len(tasks)
    prompts = []
    for i in range(batch_size):
        msg = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Overview cam:"},
                    {"type": "image", "image": images[i]},
                    {
                        "type": "text",
                        "text": "Wrist cam:",
                    },
                    {"type": "image", "image": images2[i]},
                    {
                        "type": "text",
                        "text": f"What action should the robot take to {tasks[i]}?",
                    },
                ],
            }
        ]
        prompts.append(msg)

    inputs = processor.apply_chat_template(
        prompts,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        do_rescale=False,  # pixel values are already in 0-1 range
    )

    inputs["proprio_state"] = batch["observation.state"]

    return inputs, action_chunks


class Qwen3VLA(Qwen3VLPreTrainedModel):
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.num_task_embeddings = 64
        self.task_embeddings = nn.Embedding(
            num_embeddings=self.num_task_embeddings,
            embedding_dim=config.text_config.hidden_size,
            dtype=config.text_config.dtype,
        )
        hidden_size = config.text_config.hidden_size
        num_hidden_layers = min(config.text_config.num_hidden_layers, 24)
        action_chunk_len = 8
        action_dim = 7
        proprio_dim = 8
        self.proprio_projector = ProprioProjector(
            proprio_dim=proprio_dim, llm_dim=hidden_size
        )
        self.action_head = ActionHead(
            num_layers=num_hidden_layers,
            src_dim=hidden_size,
            hidden_dim=512,
            action_dim=action_dim,
            action_chunk_len=action_chunk_len,
        )

    def predict_action(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        proprio_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        batch_size = input_ids.shape[0]
        device = self.model.device

        task_tokens = (
            self.task_embeddings.forward(
                torch.arange(0, self.num_task_embeddings, device=device)
            )
            .unsqueeze(0)
            .repeat_interleave(repeats=batch_size, dim=0)
        )

        inputs_embeds = torch.cat((inputs_embeds, task_tokens), dim=1)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    size=(batch_size, self.num_task_embeddings),
                    dtype=torch.long,
                    device=device,
                ),
            ),
            dim=1,
        )
        input_ids = torch.cat(
            (
                input_ids,
                torch.zeros(
                    (batch_size, self.num_task_embeddings),
                    dtype=torch.long,
                    device=device,
                ),
            ),
            dim=1,
        )

        position_ids, rope_deltas = self.model.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask=attention_mask,
        )
        position_ids[:, :, -self.num_task_embeddings :] = 500

        self.model.rope_deltas = rope_deltas

        outputs = self.model.forward(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )

        combined_hidden_states = []
        for item in outputs.hidden_states:
            # batch_size, seq_len, dim = item.shape
            layer_task_hidden_state = item[:, -self.num_task_embeddings:].unsqueeze(1)
            combined_hidden_states.append(layer_task_hidden_state)
        hidden_states = torch.cat(
            combined_hidden_states, dim=1
        )  # [batch_size, layers, seq_len, dim]

        out = self.action_head.predict_action(
            hidden_states=hidden_states,
            proprio_projector=self.proprio_projector,
            proprio=proprio_state.to(device=device, dtype=hidden_states.dtype),
        )

        return out


def save_checkpoint(
    output_dir: str,
    experiment_name: str,
    step: int,
    epoche: int,
    model: Qwen3VLA,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    action_normalization: ActionNormalization,
) -> None:
    state = {
        "step": step,
        "epoche": epoche,
        "proprio_projector": model.proprio_projector.state_dict(),
        "task_embeddings": model.task_embeddings.state_dict(),
        "action_head": model.action_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "action_normalization": {
            "lower_bound": action_normalization.lower_bound,
            "upper_bound": action_normalization.upper_bound,
        },
    }

    checkpoint_path = Path(output_dir) / f"{experiment_name}_{step}.pth"

    print(f"Saving: {checkpoint_path} (step={step})")

    torch.save(state, checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: Qwen3VLA,
    optimizer: Optimizer,
    lr_scheduler: LRScheduler,
    action_normalization: ActionNormalization,
) -> tuple[int, int]:

    state = torch.load(checkpoint_path)

    model.proprio_projector.load_state_dict(state["proprio_projector"])
    model.task_embeddings.load_state_dict(state["task_embeddings"])
    model.action_head.load_state_dict(state["action_head"])
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])
    an = state["action_normalization"]
    action_normalization.set_range(an["lower_bound"], an["upper_bound"])
    return state["step"], state.get("epoche") or 0


def format_actions(a: torch.Tensor) -> str:
    return "\n".join(
        tuple(", ".join(tuple(f"{x: 0.4f}" for x in r.tolist())) for r in a)
    )


def validate(
    model: Qwen3VLA,
    val_dataloader: DataLoader,
    processor: AutoProcessor,
    objective: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Run validation and return metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x in tqdm(val_dataloader, desc="Validation", leave=False):
            # Prepare batch
            inputs, target = prepare_batch(processor, x)
            inputs.to(device)
            target = target.to(device)

            # Forward pass
            output = model.predict_action(**inputs)

            # Calculate loss
            loss = objective(output, target)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return {
        "val/loss": avg_loss,
        "val/num_batches": num_batches,
    }


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Qwen3-VLA model")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device to use (e.g., 'cuda:0', 'cuda:1', overrides config)",
    )
    args = parser.parse_args()

    # Load configuration
    config = ExperimentConfig.from_yaml(args.config)
    print(f"Loaded configuration from {args.config}")
    print(f"Experiment: {config.experiment_name}")

    # Override checkpoint path if provided via command line
    if args.checkpoint is not None:
        config.checkpoint_path = args.checkpoint
        print(f"Checkpoint path overridden via command line: {args.checkpoint}")

    # Override device if provided via command line
    if args.device is not None:
        config.device = args.device
        print(f"Device overridden via command line: {args.device}")

    # Set seed for reproducibility
    set_seed(config.seed)
    print(f"Set random seed to {config.seed}")

    # Set up device
    device = torch.device(config.device)

    # Initialize wandb
    if config.enable_wandb:
        try:
            import wandb

            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=config.experiment_name,
                config=config.__dict__,
            )
            print(f"Initialized wandb project: {config.wandb_project}")
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            config.enable_wandb = False

    # Set up image augmentation for training
    train_transforms_config = ImageTransformsConfig(
        enable=config.enable_transforms,
        max_num_transforms=config.max_num_transforms,
        random_order=config.random_order,
    )
    train_transforms = ImageTransforms(train_transforms_config)

    # Set up image augmentation for validation (optional)
    val_transforms_config = ImageTransformsConfig(
        enable=config.val_use_augmentation and config.enable_transforms,
        max_num_transforms=config.max_num_transforms,
        random_order=config.random_order,
    )
    val_transforms = ImageTransforms(val_transforms_config)

    # Load full dataset first to get metadata and action stats
    print(f"Loading dataset: {config.dataset_name}")
    full_dataset = LeRobotDataset(config.dataset_name)

    # Get action normalization stats from full dataset
    q01 = full_dataset.meta.stats["action"]["q01"]
    q99 = full_dataset.meta.stats["action"]["q99"]
    action_normalization = ActionNormalization(
        torch.from_numpy(q01), torch.from_numpy(q99)
    )

    # Create train/val split at episode level (randomly sampled)
    total_episodes = full_dataset.meta.total_episodes
    num_train_episodes = int(config.train_split * total_episodes)

    # Randomly shuffle episodes for better task distribution
    all_episodes = list(range(total_episodes))
    random.shuffle(all_episodes)

    train_episodes = sorted(all_episodes[:num_train_episodes])
    val_episodes = sorted(all_episodes[num_train_episodes:])

    print(f"Splitting dataset (random): {num_train_episodes} train episodes, {len(val_episodes)} val episodes")

    # Create train and val datasets with separate transforms
    train_dataset = LeRobotDataset(
        full_dataset.repo_id,
        root=full_dataset.root,
        episodes=train_episodes,
        image_transforms=train_transforms,
    )

    val_dataset = LeRobotDataset(
        full_dataset.repo_id,
        root=full_dataset.root,
        episodes=val_episodes,
        image_transforms=val_transforms,
    )

    print(f"Training: augmentation {'enabled' if config.enable_transforms else 'disabled'}")
    print(f"Validation: augmentation {'enabled' if config.val_use_augmentation else 'disabled'}")

    # Create collators for train and val
    train_collator = DatasetCollator(
        dataset=train_dataset,
        action_chunk_len=config.action_chunk_len,
        action_normalization=action_normalization,
    )

    val_collator = DatasetCollator(
        dataset=val_dataset,
        action_chunk_len=config.action_chunk_len,
        action_normalization=action_normalization,
    )

    # Create dataloaders
    val_batch_size = config.val_batch_size if config.val_batch_size else config.batch_size

    train_dataloader = DataLoader(
        train_dataset.hf_dataset,
        batch_size=config.batch_size,
        sampler=None,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=config.num_workers,
    )

    # Create validation dataloader with limited random samples
    max_val_samples = min(
        len(val_dataset.hf_dataset), config.max_val_batches * val_batch_size
    )
    val_sampler = RandomSampler(
        val_dataset.hf_dataset, num_samples=max_val_samples, replacement=False
    )

    val_dataloader = DataLoader(
        val_dataset.hf_dataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=False,  # shuffle is controlled by sampler
        collate_fn=val_collator,
        num_workers=config.num_workers,
    )

    print(
        f"Validation will use {max_val_samples} samples "
        f"({max_val_samples // val_batch_size} batches) per evaluation"
    )

    # Load model
    print(f"Loading model: {config.model_name}")
    model = Qwen3VLA.from_pretrained(
        config.model_name,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(config.model_name)

    # Freeze VLM params, train only adapter
    model.model.requires_grad_(False)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    num_trainable_params = sum(p.numel() for p in trainable_params)
    num_action_head_params = sum(p.numel() for p in model.action_head.parameters())
    print(
        f"Total trainable parameters: {num_trainable_params:,} "
        f"(action head: {num_action_head_params:,})"
    )

    # Set up optimizer and scheduler
    optimizer = AdamW(
        trainable_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-6,
    )

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=config.max_steps,
    )

    objective = nn.L1Loss()

    # Load checkpoint if specified
    step = 0
    epoche = 0
    if config.checkpoint_path:
        checkpoint_path = Path(config.checkpoint_path)
        print(f"Loading checkpoint: {checkpoint_path}")
        step, epoche = load_checkpoint(
            checkpoint_path, model, optimizer, lr_scheduler, action_normalization
        )
        print(f"Resumed from step {step}, epoch {epoche}")

    optimizer.zero_grad()

    # Training loop
    print("Starting training...")
    while step < config.max_steps:

        for x in train_dataloader:

            model.train()

            # Prepare batch
            inputs, target = prepare_batch(processor, x)

            inputs.to(device)
            target = target.to(device)

            output = model.predict_action(**inputs)

            # Calculate loss
            loss = objective(output, target)

            # Print predictions periodically
            if step % config.print_actions_interval == 0:
                target_actions = action_normalization.unnormalize(
                    x["action_chunk"][0].cpu()
                )
                predicted_actions = action_normalization.unnormalize(output[0].cpu())
                print(f"Target actions:\n{format_actions(target_actions)}")
                print(f"Predicted actions:\n{format_actions(predicted_actions)}")
                print(f"Error:\n{format_actions(predicted_actions - target_actions)}\n")

            # Log to wandb
            if step % config.wandb_log_interval == 0 and config.enable_wandb:
                lr = lr_scheduler.get_last_lr()[0]
                print(f"[Step {step}, Epoch {epoche}] Loss: {loss.item():.6f}, LR: {lr:.2e}")
                log_dict = {
                    "train/loss": loss.item(),
                    "train/epoch": epoche,
                    "train/learning_rate": lr,
                    "train/step": step,
                }
                wandb.log(log_dict, step=step)

            loss = loss / config.grad_accumulation_steps

            loss.backward()

            if (step + 1) % config.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=config.max_grad_norm
                )
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()

            step += 1

            if step % config.save_checkpoint_interval == 0:
                save_checkpoint(
                    config.output_dir,
                    config.experiment_name,
                    step,
                    epoche,
                    model,
                    optimizer,
                    lr_scheduler,
                    action_normalization,
                )
                print(f"Checkpoint saved at step {step}")

            # Periodic validation
            if step % config.val_interval == 0:
                print(f"\nRunning validation at step {step}...")
                val_metrics = validate(
                    model,
                    val_dataloader,
                    processor,
                    objective,
                    device,
                )
                print(f"Validation loss: {val_metrics['val/loss']:.6f}")

                # Log to wandb
                if config.enable_wandb:
                    wandb.log(val_metrics, step=step)

            if step >= config.max_steps:
                break

        epoche += 1

    # Save final checkpoint if not already saved
    if step % config.save_checkpoint_interval != 0:
        save_checkpoint(
            config.output_dir,
            config.experiment_name,
            step,
            epoche,
            model,
            optimizer,
            lr_scheduler,
            action_normalization,
        )
        print(f"Final checkpoint saved at step {step}")

    # Final save
    print("\nTraining completed!")
    if config.enable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
