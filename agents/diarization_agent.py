"""
Diarization Agent for speaker identification.

This agent performs speaker diarization to identify different speakers
in audio segments. It works with the pyannote.audio library.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

from models.base import BaseAgent

logger = logging.getLogger(__name__)


class DiarizationSegment:
    """Represents a speech segment with speaker information."""

    def __init__(
        self,
        audio_data: np.ndarray,
        sample_rate: int,
        start_time: float,
        end_time: float,
        speaker_id: str,
        timestamp: datetime = None
    ):
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.start_time = start_time
        self.end_time = end_time
        self.speaker_id = speaker_id
        self.timestamp = timestamp or datetime.now()

    @property
    def duration(self) -> float:
        """Calculate duration of the segment."""
        return self.end_time - self.start_time

    def __repr__(self):
        return (
            f"DiarizationSegment(start={self.start_time:.2f}s, "
            f"end={self.end_time:.2f}s, speaker={self.speaker_id})"
        )


class DiarizationAgent(BaseAgent):
    """
    Agent responsible for speaker diarization.

    This agent identifies different speakers in audio and assigns
    speaker labels to audio segments.
    """

    def __init__(self, config: dict):
        super().__init__("DiarizationAgent", config)
        self.diarization_config = config.get('diarization', {})
        self.model = None
        self.diar_method = self.diarization_config.get('method', 'simple')  # 'pyannote' or 'simple'

    def initialize(self):
        """Initialize diarization model."""
        try:
            logger.info(f"Initializing speaker diarization ({self.diar_method})...")

            if self.diar_method == 'pyannote':
                # Use pyannote.audio (requires HF token)
                from models.pyannote_diarization import PyAnnoteDiarization

                # Import HF token from environment if not in config
                import os
                hf_token = self.diarization_config.get('hf_token')
                if not hf_token:
                    hf_token = os.environ.get('HF_TOKEN')
                    if hf_token:
                        logger.info("Using HF_TOKEN from environment variable")

                self.model = PyAnnoteDiarization(
                    model_name=self.diarization_config.get(
                        'model_name',
                        'pyannote/speaker-diarization-3.1'
                    ),
                    device=self.diarization_config.get('device', 'cpu'),
                    hf_token=hf_token,
                    min_speakers=self.diarization_config.get('min_speakers'),
                    max_speakers=self.diarization_config.get('max_speakers'),
                )

                # Load model
                self.model.load_model()

            elif self.diar_method == 'simple':
                # Use simple clustering-based approach (no external dependencies)
                from models.simple_diarization import SimpleDiarization

                self.model = SimpleDiarization(
                    n_speakers=self.diarization_config.get('n_speakers', 2),
                    n_mfcc=self.diarization_config.get('n_mfcc', 20),
                    child_pitch_threshold=self.diarization_config.get('child_pitch_threshold', 250.0)
                )

            else:
                raise ValueError(f"Unknown diarization method: {self.diar_method}")

            self.is_initialized = True
            logger.info("Diarization agent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize diarization agent: {e}")
            raise

    def execute(self, audio: np.ndarray, sample_rate: int = 16000) -> List[Dict]:
        """
        Perform speaker diarization on audio.

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of audio

        Returns:
            List of diarization segments with speaker labels
        """
        if not self.is_initialized:
            raise RuntimeError("DiarizationAgent not initialized")

        try:
            logger.info("Performing speaker diarization...")

            # Run diarization
            diar_segments = self.model.diarize_audio(audio, sample_rate)

            logger.info(
                f"Diarization complete: found "
                f"{len(set(s['speaker'] for s in diar_segments))} speakers "
                f"in {len(diar_segments)} segments"
            )

            return diar_segments

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
        if not self.is_initialized:
            raise RuntimeError("DiarizationAgent not initialized")

        try:
            logger.info(f"Performing diarization on file: {audio_path}")

            # Run diarization on file
            diar_segments = self.model.diarize_file(audio_path)

            logger.info(
                f"Diarization complete: found "
                f"{len(set(s['speaker'] for s in diar_segments))} speakers "
                f"in {len(diar_segments)} segments"
            )

            return diar_segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise

    def diarize_vad_segments(self, vad_segments: List) -> List[DiarizationSegment]:
        """
        Diarize VAD-detected audio segments using simple clustering.

        This method is optimized for the VAD → Diarization → ASR flow.

        Args:
            vad_segments: List of AudioSegment objects from VAD

        Returns:
            List of DiarizationSegment objects with speaker labels
        """
        if not self.is_initialized:
            raise RuntimeError("DiarizationAgent not initialized")

        if self.diar_method != 'simple':
            raise ValueError("diarize_vad_segments only works with 'simple' method")

        try:
            logger.info(f"Diarizing {len(vad_segments)} VAD segments...")

            # Extract audio data from VAD segments
            audio_arrays = [seg.audio_data for seg in vad_segments]

            # Get speaker labels using simple diarization
            # Use sample_rate from config (AudioSegment doesn't store it)
            sample_rate = self.config.get('audio', {}).get('sample_rate', 16000)
            speaker_labels = self.model.diarize_segments(
                audio_arrays,
                sample_rate=sample_rate
            )

            # Create DiarizationSegment objects
            diar_segments = []
            for vad_seg, speaker_label in zip(vad_segments, speaker_labels):
                diar_seg = DiarizationSegment(
                    audio_data=vad_seg.audio_data,
                    sample_rate=sample_rate,
                    start_time=vad_seg.start_time,
                    end_time=vad_seg.end_time,
                    speaker_id=speaker_label,
                    timestamp=vad_seg.timestamp if hasattr(vad_seg, 'timestamp') else datetime.now()
                )
                diar_segments.append(diar_seg)

            logger.info(
                f"Diarization complete: "
                f"{sum(1 for s in speaker_labels if s == 'child')} child segments, "
                f"{sum(1 for s in speaker_labels if s == 'adult')} adult segments"
            )

            return diar_segments

        except Exception as e:
            logger.error(f"VAD segment diarization failed: {e}")
            raise

    def create_speaker_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        diar_segments: List[Dict]
    ) -> List[DiarizationSegment]:
        """
        Create audio segments with speaker labels.

        Args:
            audio: Full audio array
            sample_rate: Sample rate
            diar_segments: Diarization results from execute()

        Returns:
            List of DiarizationSegment objects with audio data
        """
        segments = []

        for seg in diar_segments:
            start_time = seg['start']
            end_time = seg['end']
            speaker_id = seg['speaker']

            # Extract audio segment
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            audio_segment = audio[start_sample:end_sample]

            # Create segment object
            diar_seg = DiarizationSegment(
                audio_data=audio_segment,
                sample_rate=sample_rate,
                start_time=start_time,
                end_time=end_time,
                speaker_id=speaker_id,
                timestamp=datetime.now()
            )

            segments.append(diar_seg)

        return segments

    def merge_adjacent_segments(
        self,
        segments: List[Dict],
        max_gap: float = 0.5
    ) -> List[Dict]:
        """
        Merge adjacent segments from the same speaker.

        Args:
            segments: List of diarization segments
            max_gap: Maximum gap in seconds to merge across

        Returns:
            Merged segments
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x['start'])

        merged = [sorted_segments[0].copy()]

        for seg in sorted_segments[1:]:
            last = merged[-1]

            # Check if same speaker and close enough
            if (seg['speaker'] == last['speaker'] and
                seg['start'] - last['end'] <= max_gap):
                # Merge
                last['end'] = seg['end']
            else:
                # Add as new segment
                merged.append(seg.copy())

        logger.info(
            f"Merged {len(segments)} segments into {len(merged)} segments"
        )

        return merged

    def cleanup(self):
        """Cleanup resources."""
        if self.model is not None:
            del self.model
            self.model = None

        self.is_initialized = False
        logger.info("Diarization agent cleaned up")
