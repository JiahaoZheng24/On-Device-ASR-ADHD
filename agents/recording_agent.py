"""
Recording Agent - Handles continuous audio recording.
"""

import numpy as np
import sounddevice as sd
from typing import Dict, Any, Optional
from datetime import datetime
import logging
from pathlib import Path
import wave

from models.base import BaseAgent

logger = logging.getLogger(__name__)


class RecordingAgent(BaseAgent):
    """Agent responsible for continuous audio recording."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("RecordingAgent", config)
        self.sample_rate = config['audio']['sample_rate']
        self.channels = config['audio']['channels']
        self.chunk_duration = config['audio']['chunk_duration']
        self.output_dir = Path(config['audio']['output_dir'])
        
        self.is_recording = False
        self.audio_buffer = []
        self.stream = None
        
    def initialize(self) -> None:
        """Initialize the recording agent."""
        logger.info("Initializing RecordingAgent...")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Test audio device
        try:
            devices = sd.query_devices()
            logger.info(f"Available audio devices: {devices}")
            
            default_device = sd.default.device
            logger.info(f"Using default device: {default_device}")
            
        except Exception as e:
            logger.error(f"Failed to query audio devices: {e}")
            raise
        
        self.set_status("initialized")
        logger.info("RecordingAgent initialized successfully")
    
    def execute(self, duration: Optional[float] = None) -> np.ndarray:
        """
        Record audio for specified duration or continuously.
        
        Args:
            duration: Recording duration in seconds. If None, records one chunk.
            
        Returns:
            Recorded audio as numpy array
        """
        if duration is None:
            duration = self.chunk_duration
        
        logger.info(f"Starting recording for {duration} seconds...")
        self.set_status("recording")
        
        try:
            # Record audio
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            )
            sd.wait()  # Wait for recording to complete
            
            # Convert to mono if stereo
            if audio.ndim > 1 and audio.shape[1] > 1:
                audio = audio.mean(axis=1)
            
            audio = audio.flatten()
            
            logger.info(f"Recorded {len(audio) / self.sample_rate:.2f} seconds of audio")
            self.set_status("idle")
            
            return audio
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            self.set_status("error")
            raise
    
    def record_to_file(self, filepath: Path, duration: float) -> None:
        """
        Record audio directly to WAV file.
        
        Args:
            filepath: Output WAV file path
            duration: Recording duration in seconds
        """
        logger.info(f"Recording to file: {filepath}")
        
        audio = self.execute(duration)
        
        # Save as WAV
        self.save_audio(audio, filepath)
    
    def save_audio(self, audio: np.ndarray, filepath: Path) -> None:
        """
        Save audio array to WAV file.
        
        Args:
            audio: Audio data as numpy array
            filepath: Output file path
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to int16 for WAV format
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(str(filepath), 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        logger.info(f"Audio saved to {filepath}")
    
    def start_continuous_recording(self, callback=None):
        """
        Start continuous recording in background.
        
        Args:
            callback: Optional callback function for each chunk
        """
        self.is_recording = True
        self.set_status("recording_continuous")
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Recording status: {status}")
            
            if self.is_recording:
                # Convert to float32 and flatten
                audio_chunk = indata.copy().flatten()
                
                if callback:
                    callback(audio_chunk)
                else:
                    self.audio_buffer.append(audio_chunk)
        
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=audio_callback,
                dtype='float32'
            )
            self.stream.start()
            logger.info("Continuous recording started")
            
        except Exception as e:
            logger.error(f"Failed to start continuous recording: {e}")
            self.is_recording = False
            raise
    
    def stop_continuous_recording(self) -> np.ndarray:
        """
        Stop continuous recording and return buffered audio.
        
        Returns:
            Concatenated audio from buffer
        """
        self.is_recording = False
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        if self.audio_buffer:
            audio = np.concatenate(self.audio_buffer)
            self.audio_buffer = []
            logger.info(f"Continuous recording stopped. Total duration: {len(audio) / self.sample_rate:.2f}s")
            self.set_status("idle")
            return audio
        
        logger.warning("No audio in buffer")
        self.set_status("idle")
        return np.array([])
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.is_recording:
            self.stop_continuous_recording()
        
        self.set_status("cleanup")
        logger.info("RecordingAgent cleaned up")


class AudioFileAgent(BaseAgent):
    """Agent for loading audio from files (for testing/debugging)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("AudioFileAgent", config)
        self.sample_rate = config['audio']['sample_rate']
    
    def initialize(self) -> None:
        """Initialize the agent."""
        self.set_status("initialized")
        logger.info("AudioFileAgent initialized")
    
    def execute(self, filepath: Path) -> tuple[np.ndarray, int]:
        """
        Load audio from file.
        
        Args:
            filepath: Path to audio file
            
        Returns:
            Tuple of (audio array, sample rate)
        """
        logger.info(f"Loading audio from: {filepath}")
        
        try:
            import librosa
            audio, sr = librosa.load(filepath, sr=self.sample_rate, mono=True)
            logger.info(f"Loaded {len(audio) / sr:.2f}s of audio at {sr}Hz")
            return audio, sr
            
        except ImportError:
            # Fallback to wave for WAV files
            import wave
            with wave.open(str(filepath), 'rb') as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Resample if needed
                if sr != self.sample_rate:
                    logger.warning(f"Resampling from {sr}Hz to {self.sample_rate}Hz")
                    import scipy.signal
                    num_samples = int(len(audio) * self.sample_rate / sr)
                    audio = scipy.signal.resample(audio, num_samples)
                    sr = self.sample_rate
                
                return audio, sr
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.set_status("cleanup")
        logger.info("AudioFileAgent cleaned up")
