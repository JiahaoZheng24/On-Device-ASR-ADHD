"""
Pyannote.audio speaker diarization model wrapper.

This module provides a wrapper for pyannote.audio's speaker diarization pipeline,
which can identify and separate different speakers in an audio file.

Installation:
    pip install pyannote.audio

Note: Some models require accepting user conditions and providing a HuggingFace token.
To use speaker-diarization models:
1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
2. Accept the user conditions
3. Create a HuggingFace token at https://huggingface.co/settings/tokens
4. Set the token: export HF_TOKEN="your_token_here" or pass it to the constructor
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np

logger = logging.getLogger(__name__)


class PyAnnoteDiarization:
    """
    Wrapper for pyannote.audio speaker diarization pipeline.

    This class provides speaker diarization capabilities using pyannote.audio's
    pretrained models. It can identify different speakers and their speaking times.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "cpu",
        hf_token: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """
        Initialize pyannote.audio diarization pipeline.

        Args:
            model_name: HuggingFace model ID for diarization
            device: Device to run on ("cpu" or "cuda")
            hf_token: HuggingFace token (required for some models)
            min_speakers: Minimum number of speakers (None = auto-detect)
            max_speakers: Maximum number of speakers (None = auto-detect)
        """
        self.model_name = model_name
        self.device = device
        self.hf_token = hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.pipeline = None

        logger.info(f"Initializing PyAnnote diarization: {model_name}")

    def load_model(self):
        """Load the diarization pipeline."""
        try:
            from pyannote.audio import Pipeline

            # Load pipeline with authentication token if provided
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token
            )

            # Move to device
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self.pipeline.to(torch.device("cuda"))
                    logger.info("Using GPU for diarization")
                else:
                    logger.warning("CUDA not available, using CPU")
                    self.device = "cpu"

            logger.info(f"Loaded diarization pipeline on {self.device}")

        except ImportError:
            raise ImportError(
                "pyannote.audio not found. Install with: pip install pyannote.audio"
            )
        except Exception as e:
            logger.error(f"Failed to load pyannote pipeline: {e}")
            logger.error(
                "If you see authentication errors, you need to:\n"
                "1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Accept the user conditions\n"
                "3. Create a token at https://huggingface.co/settings/tokens\n"
                "4. Pass the token to this class"
            )
            raise

    def diarize_audio(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> List[Dict]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio

        Returns:
            List of diarization segments with format:
            [
                {
                    'start': float,  # Start time in seconds
                    'end': float,    # End time in seconds
                    'speaker': str   # Speaker label (e.g., "SPEAKER_00", "SPEAKER_01")
                },
                ...
            ]
        """
        if self.pipeline is None:
            self.load_model()

        try:
            # Prepare audio for pyannote (expects dict format)
            audio_dict = {
                "waveform": audio.reshape(1, -1),  # Add batch dimension
                "sample_rate": sample_rate
            }

            # Run diarization
            diarization_params = {}
            if self.min_speakers is not None:
                diarization_params['min_speakers'] = self.min_speakers
            if self.max_speakers is not None:
                diarization_params['max_speakers'] = self.max_speakers

            diarization = self.pipeline(
                audio_dict,
                **diarization_params
            )

            # Convert to list of segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                })

            logger.info(
                f"Diarization found {len(set(s['speaker'] for s in segments))} speakers "
                f"across {len(segments)} segments"
            )

            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise

    def diarize_file(self, audio_path: str) -> List[Dict]:
        """
        Perform speaker diarization on an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            List of diarization segments
        """
        if self.pipeline is None:
            self.load_model()

        try:
            # Run diarization on file directly
            diarization_params = {}
            if self.min_speakers is not None:
                diarization_params['min_speakers'] = self.min_speakers
            if self.max_speakers is not None:
                diarization_params['max_speakers'] = self.max_speakers

            diarization = self.pipeline(
                audio_path,
                **diarization_params
            )

            # Convert to list of segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                })

            logger.info(
                f"Diarization found {len(set(s['speaker'] for s in segments))} speakers "
                f"across {len(segments)} segments"
            )

            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise

    def merge_with_vad(
        self,
        diarization_segments: List[Dict],
        vad_segments: List[Tuple[float, float]]
    ) -> List[Dict]:
        """
        Merge diarization results with VAD segments.

        This ensures we only keep speech segments that were detected by VAD,
        and assign speaker labels to them.

        Args:
            diarization_segments: Output from diarize_audio()
            vad_segments: List of (start, end) tuples from VAD

        Returns:
            List of segments with speaker labels, filtered by VAD
        """
        merged = []

        for vad_start, vad_end in vad_segments:
            # Find overlapping diarization segments
            overlaps = []
            for diar_seg in diarization_segments:
                # Check for overlap
                overlap_start = max(vad_start, diar_seg['start'])
                overlap_end = min(vad_end, diar_seg['end'])

                if overlap_start < overlap_end:
                    duration = overlap_end - overlap_start
                    overlaps.append((duration, diar_seg['speaker']))

            # Assign speaker based on longest overlap
            if overlaps:
                speaker = max(overlaps, key=lambda x: x[0])[1]
            else:
                speaker = "UNKNOWN"

            merged.append({
                'start': vad_start,
                'end': vad_end,
                'speaker': speaker,
            })

        return merged


def create_diarization_model(config: dict) -> PyAnnoteDiarization:
    """
    Factory function to create diarization model from config.

    Args:
        config: Configuration dictionary

    Returns:
        PyAnnoteDiarization instance
    """
    diar_config = config.get('diarization', {})

    model = PyAnnoteDiarization(
        model_name=diar_config.get('model_name', 'pyannote/speaker-diarization-3.1'),
        device=diar_config.get('device', 'cpu'),
        hf_token=diar_config.get('hf_token'),
        min_speakers=diar_config.get('min_speakers'),
        max_speakers=diar_config.get('max_speakers'),
    )

    return model
