"""
Qwen3-VL model extended with FAST action tokens for VLA training.

Extends the Qwen3-VL vocabulary with 2048 FAST tokens and provides
training utilities for vision-language-action learning.
"""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)
from peft import LoraConfig, get_peft_model, TaskType
from typing import Optional, Dict


class Qwen3VLAModel(nn.Module):
    """
    Qwen3-VL model extended for Vision-Language-Action tasks.

    Key features:
    - Extended vocabulary: 151936 → 153984 tokens (+2048 FAST tokens)
    - New embeddings initialized with mean of existing embeddings
    - Supports LoRA fine-tuning
    - Custom loss computation (only on action tokens)
    """

    # Vocabulary configuration
    ORIGINAL_VOCAB_SIZE = 151936
    FAST_VOCAB_SIZE = 2048
    NEW_VOCAB_SIZE = ORIGINAL_VOCAB_SIZE + FAST_VOCAB_SIZE  # 153984

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_target_modules: list[str] = None,
        lora_dropout: float = 0.05,
    ):
        """
        Initialize Qwen3-VL model with extended vocabulary.

        Args:
            model_name: HuggingFace model name
            use_lora: Whether to use LoRA for efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_target_modules: Which modules to apply LoRA to
            lora_dropout: LoRA dropout rate
        """
        super().__init__()

        print(f"Loading {model_name}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        print(f"Original vocabulary size: {self.ORIGINAL_VOCAB_SIZE}")
        print(f"Extending vocabulary with {self.FAST_VOCAB_SIZE} FAST tokens...")

        # Extend vocabulary
        self._extend_vocabulary()

        print(f"New vocabulary size: {self.NEW_VOCAB_SIZE}")

        # Apply LoRA if requested
        if use_lora:
            print(f"Applying LoRA (r={lora_r}, alpha={lora_alpha})...")
            self._apply_lora(
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"],
                lora_dropout=lora_dropout,
            )

    def _extend_vocabulary(self):
        """Extend vocabulary with FAST tokens."""
        # Resize token embeddings
        # pad_to_multiple_of helps with hardware efficiency
        # mean_resizing initializes new tokens with mean of existing embeddings
        self.model.resize_token_embeddings(
            new_num_tokens=self.NEW_VOCAB_SIZE,
            pad_to_multiple_of=64,
            mean_resizing=True  # Initialize new embeddings with mean
        )

        print(f"  Input embeddings shape: {self.model.get_input_embeddings().weight.shape}")
        print(f"  Output embeddings shape: {self.model.get_output_embeddings().weight.shape}")

    def _apply_lora(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_target_modules: list[str],
        lora_dropout: float,
    ):
        """Apply LoRA to the model for efficient fine-tuning."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            pixel_values: Vision encoder inputs
            image_grid_thw: Image grid dimensions
            labels: Target token IDs for loss computation (batch, seq_len)
                   Use -100 for tokens where loss should not be computed

        Returns:
            Dictionary with loss and logits
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs
        )

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate action tokens autoregressively.

        Args:
            input_ids: Input token IDs (prompt)
            attention_mask: Attention mask
            pixel_values: Vision encoder inputs
            image_grid_thw: Image grid dimensions
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

    def save_pretrained(self, save_directory: str):
        """Save model and processor to directory."""
        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)
        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """Load a fine-tuned model from directory."""
        print(f"Loading model from {model_path}...")
        instance = cls.__new__(cls)
        super(Qwen3VLAModel, instance).__init__()

        instance.model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            **kwargs
        )

        instance.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        return instance


def test_model():
    """Test model initialization and vocabulary extension."""
    print("Testing Qwen3-VLA Model...")
    print("=" * 60)

    # Test without LoRA
    print("\n1. Testing without LoRA:")
    model = Qwen3VLAModel(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        use_lora=False,
    )

    print(f"\n Model parameters:")
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    # Test forward pass with dummy data
    print("\n2. Testing forward pass:")
    batch_size = 2
    seq_len = 100

    dummy_input = {
        "input_ids": torch.randint(0, model.NEW_VOCAB_SIZE, (batch_size, seq_len)),
        "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        "labels": torch.randint(0, model.NEW_VOCAB_SIZE, (batch_size, seq_len)),
    }

    # Move to same device as model
    device = next(model.model.parameters()).device
    dummy_input = {k: v.to(device) for k, v in dummy_input.items()}

    with torch.no_grad():
        outputs = model(**dummy_input)

    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Logits vocab size: {outputs['logits'].shape[-1]}")

    # Verify vocabulary size
    assert outputs['logits'].shape[-1] == model.NEW_VOCAB_SIZE, \
        f"Expected vocab size {model.NEW_VOCAB_SIZE}, got {outputs['logits'].shape[-1]}"

    print("\n" + "=" * 60)
    print("✓ Model test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
