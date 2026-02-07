"""
VAD Agent - Handles voice activity detection.
Transcription Agent - Handles speech-to-text conversion.
"""

import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import logging
from pathlib import Path
import json

from models.base import BaseAgent, AudioSegment, TranscriptSegment
from models.vad_models import create_vad_model
from models.asr_models import create_asr_model

logger = logging.getLogger(__name__)


class VADAgent(BaseAgent):
    """Agent responsible for detecting speech segments in audio."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("VADAgent", config)
        self.vad_config = config['vad']
        self.sample_rate = config['audio']['sample_rate']
        self.output_dir = Path(config['audio']['output_dir'])
        self.vad_model = None
        
    def initialize(self) -> None:
        """Initialize the VAD model."""
        logger.info("Initializing VADAgent...")
        
        self.vad_model = create_vad_model(self.vad_config)
        self.vad_model.load_model()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.set_status("initialized")
        logger.info("VADAgent initialized successfully")
    
    def execute(self, audio: np.ndarray, timestamp: datetime = None) -> List[AudioSegment]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Audio data as numpy array
            timestamp: Timestamp for this audio chunk
            
        Returns:
            List of AudioSegment objects
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        logger.info(f"Running VAD on {len(audio) / self.sample_rate:.2f}s of audio...")
        self.set_status("processing")
        
        try:
            segments = self.vad_model.detect_speech(audio, self.sample_rate)
            
            # Add timestamps to segments
            for segment in segments:
                segment.timestamp = timestamp
            
            logger.info(f"Detected {len(segments)} speech segments")
            
            # Log segment statistics
            if segments:
                total_duration = sum(s.duration for s in segments)
                avg_duration = total_duration / len(segments)
                logger.info(f"Total speech: {total_duration:.2f}s, Average segment: {avg_duration:.2f}s")
            
            self.set_status("idle")
            return segments
            
        except Exception as e:
            logger.error(f"VAD processing failed: {e}")
            self.set_status("error")
            raise
    
    def save_segments(self, segments: List[AudioSegment], output_dir: Path = None) -> List[Path]:
        """
        Save audio segments to files.
        
        Args:
            segments: List of AudioSegment objects
            output_dir: Output directory (uses default if None)
            
        Returns:
            List of saved file paths
        """
        if output_dir is None:
            output_dir = self.output_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        for i, segment in enumerate(segments):
            timestamp_str = segment.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"segment_{timestamp_str}_{i:04d}.npy"
            filepath = output_dir / filename
            
            # Save as numpy array (more efficient than WAV for temporary storage)
            np.save(filepath, segment.audio_data)
            saved_files.append(filepath)
        
        logger.info(f"Saved {len(saved_files)} segments to {output_dir}")
        return saved_files
    
    def load_segments(self, segment_files: List[Path]) -> List[AudioSegment]:
        """
        Load audio segments from files.
        
        Args:
            segment_files: List of file paths
            
        Returns:
            List of AudioSegment objects
        """
        segments = []
        
        for filepath in segment_files:
            audio_data = np.load(filepath)
            
            # Parse timestamp from filename
            filename = filepath.stem
            parts = filename.split('_')
            if len(parts) >= 3:
                timestamp_str = f"{parts[1]}_{parts[2]}"
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            else:
                timestamp = datetime.now()
            
            segment = AudioSegment(
                start_time=0.0,  # Relative times not preserved in this format
                end_time=len(audio_data) / self.sample_rate,
                audio_data=audio_data,
                timestamp=timestamp
            )
            segments.append(segment)
        
        logger.info(f"Loaded {len(segments)} segments from files")
        return segments
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.vad_model = None
        self.set_status("cleanup")
        logger.info("VADAgent cleaned up")


class TranscriptionAgent(BaseAgent):
    """Agent responsible for transcribing speech segments."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("TranscriptionAgent", config)
        self.asr_config = config['asr']
        self.sample_rate = config['audio']['sample_rate']
        self.output_dir = Path(config.get('system', {}).get('output_dir', 'data/transcripts'))
        self.asr_model = None
        
    def initialize(self) -> None:
        """Initialize the ASR model."""
        logger.info("Initializing TranscriptionAgent...")
        
        self.asr_model = create_asr_model(self.asr_config)
        self.asr_model.load_model()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.set_status("initialized")
        logger.info("TranscriptionAgent initialized successfully")
    
    def execute(self, segments: List[AudioSegment]) -> List[TranscriptSegment]:
        """
        Transcribe audio segments.
        
        Args:
            segments: List of AudioSegment objects
            
        Returns:
            List of TranscriptSegment objects
        """
        if not segments:
            logger.warning("No segments to transcribe")
            return []
        
        logger.info(f"Transcribing {len(segments)} segments...")
        self.set_status("processing")

        try:
            # Use batch processing if available
            transcripts = self.asr_model.transcribe_batch(segments, self.sample_rate)

            # Copy speaker_id from input segments to transcripts if available
            # (for diarization-first pipeline where segments have speaker labels)
            for i, (segment, transcript) in enumerate(zip(segments, transcripts)):
                if hasattr(segment, 'speaker_id') and segment.speaker_id:
                    transcript.speaker_id = segment.speaker_id

            # Filter out failed transcriptions
            valid_transcripts = [
                t for t in transcripts
                if t.text and t.text != "[Transcription failed]"
            ]

            logger.info(f"Successfully transcribed {len(valid_transcripts)}/{len(segments)} segments")

            self.set_status("idle")
            return valid_transcripts
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            self.set_status("error")
            raise
    
    def transcribe_single(self, segment: AudioSegment) -> TranscriptSegment:
        """
        Transcribe a single audio segment.
        
        Args:
            segment: AudioSegment object
            
        Returns:
            TranscriptSegment object
        """
        return self.asr_model.transcribe(segment, self.sample_rate)
    
    def save_transcripts(self, transcripts: List[TranscriptSegment],
                        date: datetime = None,
                        input_filename: str = None,
                        output_subdir: Path = None) -> Path:
        """
        Save transcripts to JSON file.

        Args:
            transcripts: List of TranscriptSegment objects
            date: Date for the transcripts (uses current date if None)
            input_filename: Original input audio filename (optional, deprecated)
            output_subdir: Subdirectory for this input file's outputs (optional)

        Returns:
            Path to saved file
        """
        if date is None:
            date = datetime.now()

        # Determine output directory
        if output_subdir:
            save_dir = output_subdir
        else:
            save_dir = self.output_dir

        save_dir.mkdir(parents=True, exist_ok=True)

        # Build filename
        date_str = date.strftime('%Y%m%d')
        filename = f"transcripts_{date_str}.json"
        filepath = save_dir / filename
        
        # Convert to JSON-serializable format
        data = {
            "date": date.isoformat(),
            "total_segments": len(transcripts),
            "transcripts": [t.to_dict() for t in transcripts]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(transcripts)} transcripts to {filepath}")
        return filepath
    
    def load_transcripts(self, filepath: Path) -> List[TranscriptSegment]:
        """
        Load transcripts from JSON file.
        
        Args:
            filepath: Path to transcript file
            
        Returns:
            List of TranscriptSegment objects
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        transcripts = []
        for t_dict in data['transcripts']:
            transcript = TranscriptSegment(
                start_time=t_dict['start_time'],
                end_time=t_dict['end_time'],
                text=t_dict['text'],
                confidence=t_dict['confidence'],
                timestamp=datetime.fromisoformat(t_dict['timestamp']),
                speaker_id=t_dict.get('speaker_id')
            )
            transcripts.append(transcript)
        
        logger.info(f"Loaded {len(transcripts)} transcripts from {filepath}")
        return transcripts
    
    def filter_transcripts(self, transcripts: List[TranscriptSegment], 
                          min_confidence: float = 0.5,
                          min_words: int = 3) -> List[TranscriptSegment]:
        """
        Filter transcripts by confidence and length.
        
        Args:
            transcripts: List of TranscriptSegment objects
            min_confidence: Minimum confidence score
            min_words: Minimum number of words
            
        Returns:
            Filtered list of transcripts
        """
        filtered = []
        
        for t in transcripts:
            word_count = len(t.text.split())
            
            if t.confidence >= min_confidence and word_count >= min_words:
                filtered.append(t)
        
        logger.info(f"Filtered {len(filtered)}/{len(transcripts)} transcripts")
        return filtered
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.asr_model = None
        self.set_status("cleanup")
        logger.info("TranscriptionAgent cleaned up")
