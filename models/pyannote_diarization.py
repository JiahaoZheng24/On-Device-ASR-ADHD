"""
Pyannote.audio speaker diarization model wrapper.

Requires:
    pip install pyannote.audio
    HuggingFace token with access to pyannote/speaker-diarization-3.1
"""

import logging
from typing import List, Optional, Dict
import numpy as np
import torch

logger = logging.getLogger(__name__)


class PyAnnoteDiarization:
    """Wrapper for pyannote.audio speaker diarization pipeline."""

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "cuda",
        hf_token: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
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

            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.hf_token
            )

            # Move to GPU if available
            if self.device == "cuda" and torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                logger.info("Using GPU for diarization")
            else:
                if self.device == "cuda":
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
                "If you see authentication errors:\n"
                "1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Accept the user conditions\n"
                "3. Set hf_token in config/settings.yaml"
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
            audio: Audio data as numpy array (1D)
            sample_rate: Sample rate of audio

        Returns:
            List of {'start': float, 'end': float, 'speaker': str}
        """
        if self.pipeline is None:
            self.load_model()

        try:
            # pyannote 4.0 expects a dict with torch tensor
            waveform = torch.from_numpy(audio).float()
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)  # (1, time)

            audio_dict = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            # Run diarization
            params = {}
            if self.min_speakers is not None:
                params['min_speakers'] = self.min_speakers
            if self.max_speakers is not None:
                params['max_speakers'] = self.max_speakers

            result = self.pipeline(audio_dict, **params)

            # pyannote 4.0 returns DiarizeOutput dataclass;
            # extract the Annotation object from it
            if hasattr(result, 'speaker_diarization'):
                diarization = result.speaker_diarization
            else:
                diarization = result

            # Convert to list of segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    'start': turn.start,
                    'end': turn.end,
                    'speaker': speaker,
                })

            n_speakers = len(set(s['speaker'] for s in segments))
            logger.info(
                f"Diarization found {n_speakers} speakers "
                f"across {len(segments)} segments"
            )

            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise
