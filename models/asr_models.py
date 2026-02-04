"""
Automatic Speech Recognition (ASR) model implementations.
Currently supports OpenAI Whisper and faster-whisper.
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from models.base import BaseASRModel, AudioSegment, TranscriptSegment

logger = logging.getLogger(__name__)


class WhisperASR(BaseASRModel):
    """OpenAI Whisper ASR implementation. Supports both openai-whisper and faster-whisper."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.using_faster_whisper = False
        
    def load_model(self) -> None:
        """Load Whisper model. Tries openai-whisper first, then faster-whisper."""
        model_name = self.config.get('model_name', 'base')
        device = self.config.get('device', 'cpu')
        
        # Try openai-whisper first (most compatible)
        try:
            import whisper
            
            logger.info(f"Loading Whisper model (openai-whisper): {model_name} on {device}")
            
            # Check if custom model path is provided
            model_path = self.config.get('model_path')
            if model_path:
                logger.info(f"Using custom model path: {model_path}")
                self.model = whisper.load_model(model_path, device=device)
            else:
                self.model = whisper.load_model(model_name, device=device)
            
            self.using_faster_whisper = False
            self._is_initialized = True
            logger.info("Whisper model (openai-whisper) loaded successfully")
            return
            
        except ImportError:
            logger.warning("openai-whisper not found, trying faster-whisper...")
        except Exception as e:
            logger.warning(f"Failed to load openai-whisper: {e}, trying faster-whisper...")
        
        # Fall back to faster-whisper
        try:
            from faster_whisper import WhisperModel
            
            compute_type = self.config.get('compute_type', 'int8')
            logger.info(f"Loading Whisper model (faster-whisper): {model_name} on {device} with {compute_type}")
            
            # Check if custom model path is provided
            model_path = self.config.get('model_path')
            if model_path:
                logger.info(f"Using custom model path: {model_path}")
                model_name = model_path
            
            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type
            )
            
            self.using_faster_whisper = True
            self._is_initialized = True
            logger.info("Whisper model (faster-whisper) loaded successfully")
            return
            
        except ImportError:
            error_msg = (
                "Neither openai-whisper nor faster-whisper is installed.\n"
                "Install one of them:\n"
                "  pip install openai-whisper  (recommended, most compatible)\n"
                "  pip install faster-whisper  (faster, may have Windows issues)"
            )
            logger.error(error_msg)
            raise ImportError(error_msg)
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio_segment: AudioSegment, sample_rate: int) -> TranscriptSegment:
        """
        Transcribe a single audio segment.
        
        Args:
            audio_segment: AudioSegment containing speech
            sample_rate: Sample rate of audio
            
        Returns:
            TranscriptSegment with transcription
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        # Ensure audio is float32
        audio = audio_segment.audio_data
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        if audio.max() > 1.0:
            audio = audio / 32768.0
        
        try:
            language = self.config.get('language', 'en')
            
            if self.using_faster_whisper:
                # Use faster-whisper transcribe
                segments, info = self.model.transcribe(
                    audio,
                    language=language if language != 'auto' else None,
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    vad_filter=True,
                    vad_parameters={
                        "threshold": 0.5,
                        "min_speech_duration_ms": 250,
                    }
                )
                
                # Combine all segment texts
                full_text = ""
                confidence_scores = []
                
                for segment in segments:
                    full_text += segment.text + " "
                    confidence_scores.append(segment.avg_logprob)
                
                # Calculate average confidence
                avg_confidence = np.exp(np.mean(confidence_scores)) if confidence_scores else 0.0
                
            else:
                # Use openai-whisper transcribe
                result = self.model.transcribe(
                    audio,
                    language=language if language != 'auto' else None,
                    fp16=False,  # Use fp32 for CPU
                    verbose=False
                )
                
                full_text = result['text']
                
                # Extract confidence from segments if available
                if 'segments' in result and result['segments']:
                    confidence_scores = []
                    for seg in result['segments']:
                        if 'avg_logprob' in seg:
                            confidence_scores.append(seg['avg_logprob'])
                        elif 'no_speech_prob' in seg:
                            # Convert no_speech_prob to confidence
                            confidence_scores.append(np.log(1 - seg['no_speech_prob']))
                    
                    avg_confidence = np.exp(np.mean(confidence_scores)) if confidence_scores else 0.8
                else:
                    # Default confidence
                    avg_confidence = 0.8
            
            transcript = TranscriptSegment(
                start_time=audio_segment.start_time,
                end_time=audio_segment.end_time,
                text=full_text.strip(),
                confidence=float(avg_confidence),
                timestamp=datetime.now()
            )
            
            logger.debug(f"Transcribed segment ({audio_segment.duration:.2f}s): {transcript.text[:50]}...")
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed for segment: {e}")
            # Return empty transcript on failure
            return TranscriptSegment(
                start_time=audio_segment.start_time,
                end_time=audio_segment.end_time,
                text="[Transcription failed]",
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    def transcribe_batch(self, audio_segments: List[AudioSegment], 
                        sample_rate: int) -> List[TranscriptSegment]:
        """
        Transcribe multiple audio segments.
        
        Note: faster-whisper doesn't support true batch processing,
        so we process sequentially but with optimizations.
        
        Args:
            audio_segments: List of AudioSegment objects
            sample_rate: Sample rate of audio
            
        Returns:
            List of TranscriptSegment objects
        """
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        transcripts = []
        total = len(audio_segments)
        
        logger.info(f"Transcribing {total} audio segments...")
        
        for i, segment in enumerate(audio_segments):
            if (i + 1) % 10 == 0 or i == total - 1:
                logger.info(f"Progress: {i+1}/{total} segments transcribed")
            
            transcript = self.transcribe(segment, sample_rate)
            transcripts.append(transcript)
        
        logger.info(f"Completed transcription of {total} segments")
        return transcripts


class WhisperOriginalASR(BaseASRModel):
    """Original OpenAI Whisper implementation (slower but more compatible)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        
    def load_model(self) -> None:
        """Load original Whisper model."""
        try:
            import whisper
            
            model_name = self.config.get('model_name', 'base')
            device = self.config.get('device', 'cpu')
            
            logger.info(f"Loading Whisper (original) model: {model_name} on {device}")
            
            self.model = whisper.load_model(model_name, device=device)
            self._is_initialized = True
            
            logger.info("Whisper model loaded successfully")
            
        except ImportError:
            logger.error("whisper not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio_segment: AudioSegment, sample_rate: int) -> TranscriptSegment:
        """Transcribe using original Whisper."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        audio = audio_segment.audio_data
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        if audio.max() > 1.0:
            audio = audio / 32768.0
        
        try:
            language = self.config.get('language', 'en')
            
            result = self.model.transcribe(
                audio,
                language=language if language != 'auto' else None,
                fp16=False
            )
            
            transcript = TranscriptSegment(
                start_time=audio_segment.start_time,
                end_time=audio_segment.end_time,
                text=result['text'].strip(),
                confidence=1.0,  # Original whisper doesn't provide confidence
                timestamp=datetime.now()
            )
            
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptSegment(
                start_time=audio_segment.start_time,
                end_time=audio_segment.end_time,
                text="[Transcription failed]",
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    def transcribe_batch(self, audio_segments: List[AudioSegment], 
                        sample_rate: int) -> List[TranscriptSegment]:
        """Transcribe batch sequentially."""
        transcripts = []
        for segment in audio_segments:
            transcripts.append(self.transcribe(segment, sample_rate))
        return transcripts


def create_asr_model(config: Dict[str, Any]) -> BaseASRModel:
    """
    Factory function to create ASR model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized ASR model
    """
    model_type = config.get('model_type', 'whisper').lower()
    
    if model_type == 'whisper':
        # Default to faster-whisper for efficiency
        return WhisperASR(config)
    elif model_type == 'whisper-original':
        return WhisperOriginalASR(config)
    else:
        raise ValueError(f"Unknown ASR model type: {model_type}")
