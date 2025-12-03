"""
Qwen3-VL model with State History Encoder for VLA training.

Extends the base VLA model with a neural state encoder that processes
K timesteps of state history and injects the resulting embeddings
into the transformer input sequence.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from typing import Optional, Dict
from dataclasses import dataclass

from state_encoder import StateEncoderConfig, create_state_encoder


@dataclass
class Qwen3VLAWithStateHistoryConfig:
    """Configuration for VLA model with state history encoder."""

    # Base model settings
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    new_vocab_size: int = 152192  # 151936 + 255 action tokens, rounded to 64
    original_vocab_size: int = 151936

    # State encoder settings (None = disabled)
    state_encoder_config: Optional[StateEncoderConfig] = None

    # Whether to freeze the state encoder during training (for ablation studies)
    freeze_state_encoder: bool = False


class Qwen3VLAModelWithStateHistory(nn.Module):
    """
    Qwen3-VL model extended for VLA with state history encoding.

    Key features:
    - Extended vocabulary with action tokens
    - Neural state history encoder (Conv1D, MLP, Transformer, or RNN)
    - State embeddings prepended to transformer input sequence

    Architecture:
    1. State history (K, state_dim) → StateEncoder → (n_tokens, hidden_dim)
    2. Prepend state embeddings to input sequence
    3. Adjust attention mask for state tokens
    4. Standard forward pass through Qwen3-VL
    """

    ORIGINAL_VOCAB_SIZE = 151936

    def __init__(
        self,
        config: Qwen3VLAWithStateHistoryConfig,
        device_map: Optional[str] = None,
    ):
        """
        Initialize VLA model with state history encoder.

        Args:
            config: Configuration dataclass
            device_map: Device placement. Use None for FSDP/DDP.
        """
        super().__init__()
        self.config = config
        self.NEW_VOCAB_SIZE = config.new_vocab_size
        self.ACTION_VOCAB_SIZE = config.new_vocab_size - self.ORIGINAL_VOCAB_SIZE

        # Load base model
        print(f"Loading {config.model_name}...")
        self.model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )

        # Get hidden size from model config
        self.hidden_size = self.model.config.text_config.hidden_size
        print(f"Model hidden size: {self.hidden_size}")

        # Extend vocabulary
        print(f"Original vocabulary size: {self.ORIGINAL_VOCAB_SIZE}")
        print(f"Extending vocabulary with {self.ACTION_VOCAB_SIZE} action tokens...")
        self._extend_vocabulary()
        print(f"New vocabulary size: {self.NEW_VOCAB_SIZE}")

        # Initialize state encoder if configured
        self.state_encoder = None
        if config.state_encoder_config is not None:
            # Update output_dim to match model hidden size
            config.state_encoder_config.output_dim = self.hidden_size
            print(f"\nInitializing state encoder ({config.state_encoder_config.encoder_type})...")
            self.state_encoder = create_state_encoder(config.state_encoder_config)

            # Move to same dtype as model
            self.state_encoder = self.state_encoder.to(torch.bfloat16)

            # Count parameters
            encoder_params = sum(p.numel() for p in self.state_encoder.parameters())
            print(f"  State encoder parameters: {encoder_params:,}")
            print(f"  History length: {config.state_encoder_config.history_len}")
            print(f"  Output tokens: {config.state_encoder_config.n_output_tokens}")

            if config.freeze_state_encoder:
                print("  Freezing state encoder weights")
                for param in self.state_encoder.parameters():
                    param.requires_grad = False

    def _extend_vocabulary(self):
        """Extend vocabulary with action tokens."""
        self.model.resize_token_embeddings(
            new_num_tokens=self.NEW_VOCAB_SIZE,
            pad_to_multiple_of=64,
            mean_resizing=True
        )

        print(f"  Input embeddings shape: {self.model.get_input_embeddings().weight.shape}")
        print(f"  Output embeddings shape: {self.model.get_output_embeddings().weight.shape}")

    def _prepend_state_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        state_embeds: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        Prepend state embeddings to input sequence.

        Args:
            inputs_embeds: (batch, seq_len, hidden_dim) text/vision embeddings
            attention_mask: (batch, seq_len) attention mask
            state_embeds: (batch, n_state_tokens, hidden_dim) encoded state history
            labels: (batch, seq_len) labels for loss computation, or None

        Returns:
            Tuple of (new_inputs_embeds, new_attention_mask, new_labels)
        """
        batch_size, n_state_tokens, _ = state_embeds.shape
        device = inputs_embeds.device

        # Prepend state embeddings to inputs
        new_inputs_embeds = torch.cat([state_embeds, inputs_embeds], dim=1)

        # Extend attention mask for state tokens (all ones - attend to all state tokens)
        state_attention = torch.ones(
            batch_size, n_state_tokens,
            dtype=attention_mask.dtype,
            device=device
        )
        new_attention_mask = torch.cat([state_attention, attention_mask], dim=1)

        # Extend labels if provided (use -100 to ignore state tokens in loss)
        new_labels = None
        if labels is not None:
            state_labels = torch.full(
                (batch_size, n_state_tokens),
                -100,
                dtype=labels.dtype,
                device=device
            )
            new_labels = torch.cat([state_labels, labels], dim=1)

        return new_inputs_embeds, new_attention_mask, new_labels

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        state_history: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional state history encoding.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            pixel_values: Vision encoder inputs
            image_grid_thw: Image grid dimensions
            labels: Target token IDs for loss computation
            state_history: (batch, K, state_dim) normalized state history, or None

        Returns:
            Dictionary with loss and logits
        """
        # If we have state history and a state encoder, encode and prepend
        if state_history is not None and self.state_encoder is not None:
            # Get device from model
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype

            # Move state encoder to model device if needed
            self.state_encoder = self.state_encoder.to(device=model_device, dtype=model_dtype)

            # Encode state history: (batch, K, state_dim) -> (batch, n_tokens, hidden_dim)
            state_history = state_history.to(device=model_device, dtype=model_dtype)
            state_embeds = self.state_encoder(state_history)

            # Get input embeddings from token IDs
            inputs_embeds = self.model.get_input_embeddings()(input_ids.to(model_device))

            # Ensure attention_mask and labels are on the right device
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)
            if labels is not None:
                labels = labels.to(model_device)

            # Prepend state embeddings
            inputs_embeds, attention_mask, labels = self._prepend_state_embeddings(
                inputs_embeds, attention_mask, state_embeds, labels
            )

            # Forward with inputs_embeds instead of input_ids
            outputs = self.model(
                input_ids=None,  # Don't use input_ids when providing inputs_embeds
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                labels=labels,
                **kwargs
            )
        else:
            # Standard forward without state history
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
        state_history: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate action tokens autoregressively with optional state history.

        Args:
            input_ids: Input token IDs (prompt)
            attention_mask: Attention mask
            pixel_values: Vision encoder inputs
            image_grid_thw: Image grid dimensions
            state_history: (batch, K, state_dim) normalized state history, or None
            max_new_tokens: Maximum number of tokens to generate

        Returns:
            Generated token IDs
        """
        # If we have state history and a state encoder, encode and prepend
        if state_history is not None and self.state_encoder is not None:
            # Get device from model
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype

            # Move state encoder to model device if needed
            self.state_encoder = self.state_encoder.to(device=model_device, dtype=model_dtype)

            # Encode state history
            state_history = state_history.to(device=model_device, dtype=model_dtype)
            state_embeds = self.state_encoder(state_history)

            # Get input embeddings
            inputs_embeds = self.model.get_input_embeddings()(input_ids.to(model_device))

            # Ensure attention_mask is on the right device
            if attention_mask is not None:
                attention_mask = attention_mask.to(model_device)

            # Prepend state embeddings
            inputs_embeds, attention_mask, _ = self._prepend_state_embeddings(
                inputs_embeds, attention_mask, state_embeds, labels=None
            )

            # Generate with inputs_embeds
            return self.model.generate(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                **kwargs
            )
        else:
            # Standard generation
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                max_new_tokens=max_new_tokens,
                **kwargs
            )

    def save_pretrained(self, save_directory: str):
        """Save model, processor, and state encoder to directory."""
        import os

        self.model.save_pretrained(save_directory)
        self.processor.save_pretrained(save_directory)

        # Save state encoder separately if present
        if self.state_encoder is not None:
            encoder_path = os.path.join(save_directory, "state_encoder.pt")
            torch.save({
                "state_dict": self.state_encoder.state_dict(),
                "config": self.config.state_encoder_config,
            }, encoder_path)
            print(f"State encoder saved to {encoder_path}")

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device_map: Optional[str] = None,
        state_encoder_config: Optional[StateEncoderConfig] = None,
    ):
        """
        Load a fine-tuned model from directory.

        Args:
            model_path: Path to saved model
            device_map: Device placement
            state_encoder_config: State encoder config (if not saved with model)
        """
        import os

        print(f"Loading model from {model_path}...")

        # Check for saved state encoder
        encoder_path = os.path.join(model_path, "state_encoder.pt")
        if os.path.exists(encoder_path):
            print(f"Loading state encoder from {encoder_path}")
            encoder_data = torch.load(encoder_path, map_location="cpu")
            state_encoder_config = encoder_data["config"]

        # Create config
        config = Qwen3VLAWithStateHistoryConfig(
            model_name=model_path,  # Load from fine-tuned checkpoint
            state_encoder_config=state_encoder_config,
        )

        instance = cls(config, device_map=device_map)

        # Load state encoder weights if present
        if os.path.exists(encoder_path):
            encoder_data = torch.load(encoder_path, map_location="cpu")
            instance.state_encoder.load_state_dict(encoder_data["state_dict"])
            print("State encoder weights loaded")

        return instance


def test_model():
    """Test model initialization and forward pass."""
    print("Testing Qwen3-VLA Model with State History")
    print("=" * 60)

    # Test configuration with state encoder
    state_config = StateEncoderConfig(
        encoder_type="conv1d",
        history_len=10,
        state_dim=14,
        n_output_tokens=4,
    )

    config = Qwen3VLAWithStateHistoryConfig(
        model_name="Qwen/Qwen3-VL-2B-Instruct",
        state_encoder_config=state_config,
    )

    print("\n1. Testing model initialization...")
    model = Qwen3VLAModelWithStateHistory(config, device_map="cuda:0")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = sum(p.numel() for p in model.state_encoder.parameters()) if model.state_encoder else 0

    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  State encoder: {encoder_params:,}")

    # Test forward pass with dummy data
    print("\n2. Testing forward pass with state history...")
    batch_size = 2
    seq_len = 100
    history_len = 10
    state_dim = 14

    device = next(model.model.parameters()).device

    dummy_input = {
        "input_ids": torch.randint(0, model.NEW_VOCAB_SIZE, (batch_size, seq_len)).to(device),
        "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long).to(device),
        "labels": torch.randint(0, model.NEW_VOCAB_SIZE, (batch_size, seq_len)).to(device),
        "state_history": torch.randn(batch_size, history_len, state_dim).to(device),
    }

    with torch.no_grad():
        outputs = model(**dummy_input)

    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Logits shape: {outputs['logits'].shape}")

    # Expected shape: (batch, seq_len + n_state_tokens, vocab_size)
    expected_seq_len = seq_len + state_config.n_output_tokens
    print(f"  Expected seq_len: {expected_seq_len}")
    print(f"  Actual seq_len: {outputs['logits'].shape[1]}")

    print("\n3. Testing forward pass without state history...")
    dummy_input_no_state = {
        "input_ids": dummy_input["input_ids"],
        "attention_mask": dummy_input["attention_mask"],
        "labels": dummy_input["labels"],
    }

    with torch.no_grad():
        outputs_no_state = model(**dummy_input_no_state)

    print(f"  Loss: {outputs_no_state['loss'].item():.4f}")
    print(f"  Logits shape: {outputs_no_state['logits'].shape}")

    # Test with apply_chat_template and dummy image
    print("\n4. Testing forward pass with apply_chat_template and image...")

    # Create dummy image (3, 240, 320) - all zeros, float tensor in [0, 1]
    dummy_image = torch.zeros(3, 240, 320, dtype=torch.float32)

    # Build conversation with image
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": dummy_image},
                {"type": "text", "text": "Task: Pick up the red block\nRobot: test-robot\nState: [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]"},
            ],
        }
    ]

    # Apply chat template
    chat_inputs = model.processor.apply_chat_template(
        [conversation],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        do_rescale=False,  # Image is already in [0, 1]
    )

    # Move to device
    chat_inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in chat_inputs.items()}

    # Create dummy state history
    state_history_chat = torch.randn(1, history_len, state_dim).to(device)

    # Create dummy labels - mask most of prompt but keep last few tokens for loss
    prompt_len = chat_inputs["input_ids"].shape[1]
    labels_chat = chat_inputs["input_ids"].clone()
    # Mask all but the last 10 tokens (simulating action tokens at the end)
    labels_chat[:, :-10] = -100

    print(f"  Input IDs shape: {chat_inputs['input_ids'].shape}")
    print(f"  Pixel values shape: {chat_inputs['pixel_values'].shape}")
    print(f"  Image grid THW: {chat_inputs['image_grid_thw']}")

    with torch.no_grad():
        outputs_chat = model(
            input_ids=chat_inputs["input_ids"],
            attention_mask=chat_inputs["attention_mask"],
            pixel_values=chat_inputs["pixel_values"],
            image_grid_thw=chat_inputs["image_grid_thw"],
            labels=labels_chat,
            state_history=state_history_chat,
        )

    print(f"  Loss: {outputs_chat['loss'].item():.4f}")
    print(f"  Logits shape: {outputs_chat['logits'].shape}")
    expected_seq_len_chat = prompt_len + state_config.n_output_tokens
    print(f"  Expected seq_len: {expected_seq_len_chat}")
    print(f"  Actual seq_len: {outputs_chat['logits'].shape[1]}")

    print("\n" + "=" * 60)
    print("Model test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_model()
