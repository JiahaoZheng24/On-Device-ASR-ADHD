"""
WhisperWithDiarization model wrapper.

This is a wrapper for the USC SAIL joint ASR-diarization model.
Paper: "End-to-End Joint ASR and Speaker Role Diarization with Child-Adult Interactions"
Source: https://github.com/usc-sail/joint-asr-diarization-child-adult

To use this model:
1. Install dependencies: pip install transformers torch
2. The model will be automatically downloaded from HuggingFace:
   https://huggingface.co/AlexXu811/child-adult-joint-asr-diarization
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from transformers import WhisperForConditionalGeneration, WhisperConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
import logging

logger = logging.getLogger(__name__)


class DiarizationHead(nn.Module):
    """CNN-based diarization head for speaker classification.

    Architecture matches the USC SAIL model:
    - Conv layers: 3x Conv1d with kernel_size=1
    - Classifier: Conv1d with kernel_size=1
    """

    def __init__(self, input_dim: int, num_classes: int = 3, hidden_dim: int = 256):
        """
        Args:
            input_dim: Input dimension from Whisper encoder
            num_classes: Number of speaker classes (default: 3 for silence/child/adult)
            hidden_dim: Hidden dimension for CNN layers
        """
        super().__init__()

        # Conv layers (matches diarization_conv_layers.pt structure)
        # Keys: 0.weight, 0.bias, 3.weight, 3.bias, 6.weight, 6.bias
        # Structure: Conv1d -> ReLU -> ReLU -> Conv1d -> ReLU -> ReLU -> Conv1d
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),  # index 0
            nn.ReLU(),                                         # index 1
            nn.ReLU(),                                         # index 2
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1), # index 3
            nn.ReLU(),                                         # index 4
            nn.ReLU(),                                         # index 5
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1), # index 6
        )

        # Classifier is Conv1d, not Linear (matches diarization_classifier.pt)
        # weight shape: [3, 256, 1] -> Conv1d(256, 3, kernel_size=1)
        self.classifier = nn.Conv1d(hidden_dim, num_classes, kernel_size=1)

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_hidden_states: [batch, seq_len, hidden_dim]

        Returns:
            diar_logits: [batch, seq_len, num_classes]
        """
        # Transpose for Conv1d: [batch, hidden_dim, seq_len]
        x = encoder_hidden_states.transpose(1, 2)
        x = self.conv_layers(x)
        x = self.classifier(x)
        # Transpose back: [batch, seq_len, num_classes]
        return x.transpose(1, 2)


class WhisperWithDiarization(nn.Module):
    """
    Whisper model with joint ASR and speaker diarization.

    This model extends Whisper with a diarization head that classifies
    each time step as silence (0), child (1), or adult (2).
    """

    def __init__(
        self,
        whisper_model: WhisperForConditionalGeneration,
        num_diar_classes: int = 3,
        diar_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.model = whisper_model
        self.num_diar_classes = num_diar_classes
        self.diar_loss_weight = diar_loss_weight

        # Get encoder hidden size
        encoder_hidden_size = self.model.config.d_model

        # Initialize diarization head
        self.diar_head = DiarizationHead(
            input_dim=encoder_hidden_size,
            num_classes=num_diar_classes
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_diar_classes: int = 3,
        diar_loss_weight: float = 1.0,
        **kwargs
    ) -> "WhisperWithDiarization":
        """
        Load pretrained model from HuggingFace Hub.

        Args:
            pretrained_model_name_or_path: HuggingFace model ID or local path
            num_diar_classes: Number of diarization classes
            diar_loss_weight: Weight for diarization loss
        """
        logger.info(f"Loading WhisperWithDiarization from {pretrained_model_name_or_path}")

        # Try to load the full model with diarization head
        try:
            # Load base Whisper model
            whisper_model = WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs
            )

            model = cls(
                whisper_model=whisper_model,
                num_diar_classes=num_diar_classes,
                diar_loss_weight=diar_loss_weight
            )

            # Load diarization head weights from HuggingFace
            try:
                from huggingface_hub import hf_hub_download

                # Download conv layers weights
                conv_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="diarization_conv_layers.pt"
                )
                conv_state = torch.load(conv_path, map_location="cpu", weights_only=True)

                # Download classifier weights
                classifier_path = hf_hub_download(
                    repo_id=pretrained_model_name_or_path,
                    filename="diarization_classifier.pt"
                )
                classifier_state = torch.load(classifier_path, map_location="cpu", weights_only=True)

                # Detect hidden dimension from conv weights
                if isinstance(conv_state, dict) and '0.weight' in conv_state:
                    hidden_dim = conv_state['0.weight'].shape[0]
                    logger.info(f"Detected hidden dimension: {hidden_dim}")

                    # Rebuild diar_head with correct dimensions if needed
                    if hidden_dim != model.diar_head.classifier.in_channels:
                        logger.info(f"Rebuilding diar_head with hidden_dim={hidden_dim}")
                        model.diar_head = DiarizationHead(
                            input_dim=model.model.config.d_model,
                            num_classes=num_diar_classes,
                            hidden_dim=hidden_dim
                        )

                # Load conv layers weights
                if isinstance(conv_state, dict):
                    model.diar_head.conv_layers.load_state_dict(conv_state)
                    logger.info("Loaded diarization conv layers weights")

                # Load classifier weights (Conv1d format)
                if isinstance(classifier_state, dict):
                    model.diar_head.classifier.load_state_dict(classifier_state)
                    logger.info("Loaded diarization classifier weights")

                logger.info("Diarization head weights loaded successfully")

            except Exception as e:
                logger.warning(f"Could not load diarization head weights: {e}")
                logger.warning("Diarization results may be inaccurate without pretrained weights")
                import traceback
                logger.warning(traceback.format_exc())

            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        diar_labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with both ASR and diarization.

        Args:
            input_features: Mel spectrogram features
            labels: ASR labels for loss computation
            diar_labels: Diarization labels [batch, seq_len]

        Returns:
            Tuple of (total_loss, asr_loss, diar_loss, asr_logits, diar_logits)
        """
        # Get encoder outputs
        encoder_outputs = self.model.model.encoder(
            input_features,
            attention_mask=attention_mask,
        )

        # Compute diarization logits
        diar_logits = self.diar_head(encoder_outputs.last_hidden_state)

        # Forward through decoder for ASR
        outputs = self.model(
            input_features=input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            **kwargs
        )

        asr_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)

        # Compute diarization loss if labels provided
        if diar_labels is not None:
            diar_loss_fn = nn.CrossEntropyLoss()
            diar_loss = diar_loss_fn(
                diar_logits.view(-1, self.num_diar_classes),
                diar_labels.view(-1)
            )
        else:
            diar_loss = torch.tensor(0.0, device=input_features.device)

        # Combined loss
        total_loss = asr_loss + self.diar_loss_weight * diar_loss

        return {
            'loss': total_loss,
            'asr_loss': asr_loss,
            'diar_loss': diar_loss,
            'logits': outputs.logits,
            'diar_logits': diar_logits,
        }

    def generate(
        self,
        input_features: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Generate transcription tokens."""
        return self.model.generate(input_features, **kwargs)

    def to(self, device):
        """Move model to device."""
        super().to(device)
        self.model = self.model.to(device)
        self.diar_head = self.diar_head.to(device)
        return self

    def eval(self):
        """Set model to evaluation mode."""
        super().eval()
        self.model.eval()
        self.diar_head.eval()
        return self
