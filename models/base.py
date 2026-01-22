"""
Base abstract classes for models in the ADHD Audio Summarization System.
This module defines the interfaces that all model implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class AudioSegment:
    """Represents a detected speech segment."""
    start_time: float
    end_time: float
    audio_data: Optional[np.ndarray] = None
    timestamp: Optional[datetime] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class TranscriptSegment:
    """Represents a transcribed speech segment."""
    start_time: float
    end_time: float
    text: str
    confidence: float
    timestamp: datetime
    speaker_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "speaker_id": self.speaker_id
        }


@dataclass
class DailySummary:
    """Represents a daily summary report."""
    date: datetime
    total_speech_duration: float
    segment_count: int
    temporal_distribution: Dict[int, float]  # hour -> speech duration
    communication_patterns: Dict[str, Any]
    representative_excerpts: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "total_speech_duration": self.total_speech_duration,
            "segment_count": self.segment_count,
            "temporal_distribution": self.temporal_distribution,
            "communication_patterns": self.communication_patterns,
            "representative_excerpts": self.representative_excerpts,
            "metadata": self.metadata
        }


class BaseVADModel(ABC):
    """Abstract base class for Voice Activity Detection models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._is_initialized = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the VAD model."""
        pass
    
    @abstractmethod
    def detect_speech(self, audio: np.ndarray, sample_rate: int) -> List[AudioSegment]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            List of AudioSegment objects containing detected speech
        """
        pass
    
    def is_initialized(self) -> bool:
        return self._is_initialized


class BaseASRModel(ABC):
    """Abstract base class for Automatic Speech Recognition models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._is_initialized = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the ASR model."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_segment: AudioSegment, sample_rate: int) -> TranscriptSegment:
        """
        Transcribe a single audio segment.
        
        Args:
            audio_segment: AudioSegment object containing speech
            sample_rate: Sample rate of the audio
            
        Returns:
            TranscriptSegment object with transcription
        """
        pass
    
    @abstractmethod
    def transcribe_batch(self, audio_segments: List[AudioSegment], 
                        sample_rate: int) -> List[TranscriptSegment]:
        """
        Transcribe multiple audio segments in batch.
        
        Args:
            audio_segments: List of AudioSegment objects
            sample_rate: Sample rate of the audio
            
        Returns:
            List of TranscriptSegment objects
        """
        pass
    
    def is_initialized(self) -> bool:
        return self._is_initialized


class BaseLLMModel(ABC):
    """Abstract base class for Large Language Models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._is_initialized = False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the LLM model."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on prompt.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        pass
    
    @abstractmethod
    def generate_summary(self, transcripts: List[TranscriptSegment]) -> DailySummary:
        """
        Generate daily summary from transcripts.
        
        Args:
            transcripts: List of TranscriptSegment objects
            
        Returns:
            DailySummary object
        """
        pass
    
    def is_initialized(self) -> bool:
        return self._is_initialized


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self._status = "idle"
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the agent and its resources."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the agent's main task."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass
    
    @property
    def status(self) -> str:
        return self._status
    
    def set_status(self, status: str) -> None:
        self._status = status
