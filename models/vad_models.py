"""
Voice Activity Detection (VAD) model implementations.
Currently supports Silero VAD and WebRTC VAD.
"""

import torch
import numpy as np
from typing import List, Dict, Any
import logging

from models.base import BaseVADModel, AudioSegment

logger = logging.getLogger(__name__)


class SileroVAD(BaseVADModel):
    """Silero VAD implementation - high quality, neural network based."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.utils = None
        
    def load_model(self) -> None:
        """Load Silero VAD model from torch hub."""
        try:
            logger.info("Loading Silero VAD model...")
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self._is_initialized = True
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        Detect speech segments using Silero VAD.
        
        Args:
            audio: Audio data as numpy array (float32, -1 to 1)
            sample_rate: Sample rate (Silero expects 16000 Hz)
            
        Returns:
            List of AudioSegment objects
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        # Ensure audio is the right format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Resample if necessary (Silero expects 16kHz)
        if sample_rate != 16000:
            logger.warning(f"Resampling from {sample_rate}Hz to 16000Hz for Silero VAD")
            audio = self._resample(audio, sample_rate, 16000)
            sample_rate = 16000
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio)
        
        # Get speech timestamps
        threshold = self.config.get('threshold', 0.5)
        min_speech_duration = int(self.config.get('min_speech_duration', 0.5) * sample_rate)
        min_silence_duration = int(self.config.get('min_silence_duration', 0.3) * sample_rate)
        
        speech_timestamps = self.utils[0](
            audio_tensor,
            self.model,
            threshold=threshold,
            min_speech_duration_ms=int(min_speech_duration / sample_rate * 1000),
            min_silence_duration_ms=int(min_silence_duration / sample_rate * 1000),
            return_seconds=False
        )
        
        # Convert to AudioSegment objects
        segments = []
        padding_samples = int(self.config.get('padding', 0.1) * sample_rate)
        
        for timestamp in speech_timestamps:
            start_sample = max(0, timestamp['start'] - padding_samples)
            end_sample = min(len(audio), timestamp['end'] + padding_samples)
            
            segment = AudioSegment(
                start_time=start_sample / sample_rate,
                end_time=end_sample / sample_rate,
                audio_data=audio[start_sample:end_sample]
            )
            segments.append(segment)
        
        logger.info(f"Detected {len(segments)} speech segments")
        return segments
    
    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling (in production, use librosa or torchaudio)."""
        # This is a placeholder - in production use proper resampling
        import scipy.signal
        number_of_samples = round(len(audio) * float(target_sr) / orig_sr)
        resampled = scipy.signal.resample(audio, number_of_samples)
        return resampled.astype(np.float32)


class WebRTCVAD(BaseVADModel):
    """WebRTC VAD implementation - lightweight, rule-based."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vad = None
        
    def load_model(self) -> None:
        """Load WebRTC VAD."""
        try:
            import webrtcvad
            logger.info("Loading WebRTC VAD...")
            self.vad = webrtcvad.Vad()
            
            # Set aggressiveness (0-3, higher = more aggressive filtering)
            aggressiveness = self.config.get('aggressiveness', 2)
            self.vad.set_mode(aggressiveness)
            
            self._is_initialized = True
            logger.info("WebRTC VAD loaded successfully")
        except ImportError:
            logger.error("webrtcvad not installed. Install with: pip install webrtcvad")
            raise
        except Exception as e:
            logger.error(f"Failed to load WebRTC VAD: {e}")
            raise
    
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        Detect speech segments using WebRTC VAD.
        
        Args:
            audio: Audio data as numpy array (int16 format expected)
            sample_rate: Sample rate (must be 8000, 16000, 32000, or 48000 Hz)
            
        Returns:
            List of AudioSegment objects
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        # WebRTC VAD requires specific sample rates
        valid_rates = [8000, 16000, 32000, 48000]
        if sample_rate not in valid_rates:
            raise ValueError(f"Sample rate must be one of {valid_rates} for WebRTC VAD")
        
        # Convert to int16 if needed
        if audio.dtype != np.int16:
            audio = (audio * 32767).astype(np.int16)
        
        # Process in frames (WebRTC VAD requires 10, 20, or 30 ms frames)
        frame_duration = 30  # ms
        frame_length = int(sample_rate * frame_duration / 1000)
        
        segments = []
        current_segment_start = None
        
        for i in range(0, len(audio) - frame_length, frame_length):
            frame = audio[i:i + frame_length].tobytes()
            
            is_speech = self.vad.is_speech(frame, sample_rate)
            
            if is_speech:
                if current_segment_start is None:
                    current_segment_start = i
            else:
                if current_segment_start is not None:
                    # End of speech segment
                    segment = AudioSegment(
                        start_time=current_segment_start / sample_rate,
                        end_time=i / sample_rate,
                        audio_data=audio[current_segment_start:i]
                    )
                    segments.append(segment)
                    current_segment_start = None
        
        # Handle last segment if still open
        if current_segment_start is not None:
            segment = AudioSegment(
                start_time=current_segment_start / sample_rate,
                end_time=len(audio) / sample_rate,
                audio_data=audio[current_segment_start:]
            )
            segments.append(segment)
        
        logger.info(f"Detected {len(segments)} speech segments")
        return segments


def create_vad_model(config: Dict[str, Any]) -> BaseVADModel:
    """
    Factory function to create VAD model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized VAD model
    """
    model_type = config.get('model', 'silero').lower()
    
    if model_type == 'silero':
        return SileroVAD(config)
    elif model_type == 'webrtc':
        return WebRTCVAD(config)
    else:
        raise ValueError(f"Unknown VAD model type: {model_type}")
