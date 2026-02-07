"""
Pipeline Orchestrator - Coordinates all agents in the system.
"""

from typing import Dict, Any, Optional
from datetime import datetime, time as datetime_time
from pathlib import Path
import logging
import yaml

from agents.recording_agent import RecordingAgent, AudioFileAgent
from agents.vad_transcription_agents import VADAgent, TranscriptionAgent
from agents.summary_agent import SummaryAgent
from agents.diarization_agent import DiarizationAgent

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Main orchestrator that coordinates all agents in the system.
    Implements the complete pipeline from audio recording to daily summary.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize agents (lazy loading)
        self.recording_agent: Optional[RecordingAgent] = None
        self.audio_file_agent: Optional[AudioFileAgent] = None
        self.vad_agent: Optional[VADAgent] = None
        self.diarization_agent: Optional[DiarizationAgent] = None
        self.transcription_agent: Optional[TranscriptionAgent] = None
        self.summary_agent: Optional[SummaryAgent] = None

        self.is_initialized = False
        
    @staticmethod
    def _load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        logger.info(f"Loading configuration from {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def initialize_agents(self, agents: list = None) -> None:
        """
        Initialize specified agents or all agents.
        
        Args:
            agents: List of agent names to initialize. If None, initializes all.
        """
        if agents is None:
            agents = ['recording', 'vad', 'transcription', 'summary']
        
        logger.info(f"Initializing agents: {agents}")
        
        if 'recording' in agents:
            self.recording_agent = RecordingAgent(self.config)
            self.recording_agent.initialize()
        
        if 'audio_file' in agents:
            self.audio_file_agent = AudioFileAgent(self.config)
            self.audio_file_agent.initialize()
        
        if 'vad' in agents:
            self.vad_agent = VADAgent(self.config)
            self.vad_agent.initialize()

        if 'diarization' in agents:
            self.diarization_agent = DiarizationAgent(self.config)
            self.diarization_agent.initialize()

        if 'transcription' in agents:
            self.transcription_agent = TranscriptionAgent(self.config)
            self.transcription_agent.initialize()

        if 'summary' in agents:
            self.summary_agent = SummaryAgent(self.config)
            self.summary_agent.initialize()

        self.is_initialized = True
        logger.info("All requested agents initialized successfully")
    
    def run_full_pipeline(self, audio_source: str = "record", 
                         duration: float = 300) -> Path:
        """
        Run the complete pipeline from audio to summary.
        
        Args:
            audio_source: "record" for live recording or path to audio file
            duration: Recording duration in seconds (ignored if using file)
            
        Returns:
            Path to generated summary report
        """
        logger.info("=" * 60)
        logger.info("Starting full pipeline execution")
        logger.info("=" * 60)
        
        # Determine which agents are needed based on audio source and diarization config
        diarization_enabled = self.config.get('diarization', {}).get('enabled', False)
        required_agents = ['vad', 'transcription']  # Summary is optional

        if diarization_enabled:
            required_agents.insert(1, 'diarization')  # Add after VAD
            logger.info("Diarization-first mode enabled")

        if audio_source == "record":
            required_agents.insert(0, 'recording')
        else:
            required_agents.insert(0, 'audio_file')

        # Ensure required agents are initialized
        if not self.is_initialized:
            self.initialize_agents(required_agents)
        else:
            # Initialize missing agents if needed
            if audio_source == "record" and self.recording_agent is None:
                self.recording_agent = RecordingAgent(self.config)
                self.recording_agent.initialize()
            elif audio_source != "record" and self.audio_file_agent is None:
                self.audio_file_agent = AudioFileAgent(self.config)
                self.audio_file_agent.initialize()
        
        # Step 1: Get audio
        output_subdir = None
        if audio_source == "record":
            logger.info(f"Step 1/4: Recording audio ({duration}s)...")
            audio = self.recording_agent.execute(duration)
            sample_rate = self.config['audio']['sample_rate']
        else:
            logger.info(f"Step 1/4: Loading audio from {audio_source}...")
            audio, sample_rate = self.audio_file_agent.execute(Path(audio_source))
            # Create subdirectory for this input file's outputs
            input_basename = Path(audio_source).stem
            # Sanitize the folder name
            input_basename = "".join(c for c in input_basename if c.isalnum() or c in ('_', '-'))
            output_subdir = Path(self.config['system']['output_dir']) / input_basename
            output_subdir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_subdir}")
        
        # Step 2: Speaker diarization or VAD
        if diarization_enabled:
            diar_method = self.config.get('diarization', {}).get('method', 'simple')

            if diar_method == 'pyannote':
                # Pyannote: Run diarization on full audio
                logger.info("Step 2/5: Performing speaker diarization (pyannote)...")
                diar_segments = self.diarization_agent.execute(audio, sample_rate)

                if not diar_segments:
                    logger.warning("No speech detected. Pipeline terminated.")
                    return None

                logger.info(
                    f"Found {len(set(s['speaker'] for s in diar_segments))} speakers "
                    f"in {len(diar_segments)} segments"
                )

                # Step 3: Create audio segments with speaker labels
                logger.info("Step 3/5: Creating speaker-labeled audio segments...")
                speaker_segments = self.diarization_agent.create_speaker_segments(
                    audio, sample_rate, diar_segments
                )

                # Step 4: Transcribe each speaker segment
                logger.info("Step 4/5: Transcribing speaker segments...")
                transcripts = self.transcription_agent.execute(speaker_segments)

            elif diar_method == 'simple':
                # Simple: VAD first, then diarize VAD segments
                logger.info("Step 2/5: Detecting speech segments (VAD)...")
                speech_segments = self.vad_agent.execute(audio)

                if not speech_segments:
                    logger.warning("No speech detected. Pipeline terminated.")
                    return None

                logger.info(f"Found {len(speech_segments)} speech segments")

                # Step 3: Diarize VAD segments
                logger.info("Step 3/5: Diarizing speech segments (simple clustering)...")
                speaker_segments = self.diarization_agent.diarize_vad_segments(speech_segments)

                # Step 4: Transcribe speaker-labeled segments
                logger.info("Step 4/5: Transcribing speaker-labeled segments...")
                transcripts = self.transcription_agent.execute(speaker_segments)

            else:
                raise ValueError(f"Unknown diarization method: {diar_method}")

        else:
            # Standard VAD-based flow (no diarization)
            logger.info("Step 2/4: Detecting speech segments...")
            speech_segments = self.vad_agent.execute(audio)

            if not speech_segments:
                logger.warning("No speech detected. Pipeline terminated.")
                return None

            logger.info(f"Found {len(speech_segments)} speech segments")

            # Step 3: Transcribe speech
            logger.info("Step 3/4: Transcribing speech...")
            transcripts = self.transcription_agent.execute(speech_segments)
        
        if not transcripts:
            logger.warning("No transcripts generated. Pipeline terminated.")
            return None

        # Save transcripts
        transcript_path = self.transcription_agent.save_transcripts(
            transcripts, output_subdir=output_subdir
        )
        logger.info(f"Transcripts saved to {transcript_path}")

        # Final step: Generate summary (optional)
        output_dir = output_subdir or Path(self.config['system']['output_dir'])
        try:
            if self.summary_agent is None:
                logger.info("Initializing summary agent...")
                self.summary_agent = SummaryAgent(self.config)
                self.summary_agent.initialize()

            final_step = "5/5" if diarization_enabled else "4/4"
            logger.info(f"Step {final_step}: Generating daily summary...")
            summary = self.summary_agent.execute(transcripts)

            # Save summary
            output_dir = self.summary_agent.save_summary(summary, output_subdir=output_subdir)
        except Exception as e:
            logger.warning(f"Summary generation skipped due to error: {e}")
            logger.info("Transcripts are still available for review")

        logger.info("=" * 60)
        logger.info("Pipeline execution completed")
        logger.info(f"Output saved to: {output_dir}")
        logger.info("=" * 60)

        return output_dir
    
    def run_vad_only(self, audio_source: str, save_segments: bool = True) -> list:
        """
        Run only VAD on audio.
        
        Args:
            audio_source: "record" for live recording or path to audio file
            save_segments: Whether to save detected segments
            
        Returns:
            List of AudioSegment objects
        """
        logger.info("Running VAD-only mode")
        
        if not self.vad_agent:
            self.initialize_agents(['vad'])
        
        # Load or record audio
        if audio_source == "record":
            if not self.recording_agent:
                self.initialize_agents(['recording'])
            audio = self.recording_agent.execute()
        else:
            if not self.audio_file_agent:
                self.initialize_agents(['audio_file'])
            audio, _ = self.audio_file_agent.execute(Path(audio_source))
        
        # Detect speech
        segments = self.vad_agent.execute(audio)
        
        if save_segments:
            self.vad_agent.save_segments(segments)
        
        return segments
    
    def run_transcription_only(self, segments_dir: str = None) -> list:
        """
        Run only transcription on pre-detected segments.
        
        Args:
            segments_dir: Directory containing saved segments
            
        Returns:
            List of TranscriptSegment objects
        """
        logger.info("Running transcription-only mode")
        
        if not self.transcription_agent:
            self.initialize_agents(['transcription'])
        
        # Load segments
        if segments_dir:
            segment_files = sorted(Path(segments_dir).glob("*.npy"))
            
            if not self.vad_agent:
                self.initialize_agents(['vad'])
            
            segments = self.vad_agent.load_segments(segment_files)
        else:
            logger.error("No segments directory provided")
            return []
        
        # Transcribe
        transcripts = self.transcription_agent.execute(segments)
        
        # Save transcripts
        self.transcription_agent.save_transcripts(transcripts)
        
        return transcripts
    
    def run_summary_only(self, transcript_file: str) -> Path:
        """
        Run only summary generation from existing transcripts.
        
        Args:
            transcript_file: Path to transcript JSON file
            
        Returns:
            Path to generated summary
        """
        logger.info("Running summary-only mode")
        
        if not self.summary_agent:
            self.initialize_agents(['summary'])
        
        if not self.transcription_agent:
            self.initialize_agents(['transcription'])
        
        # Load transcripts
        transcripts = self.transcription_agent.load_transcripts(Path(transcript_file))
        
        # Generate summary
        summary = self.summary_agent.execute(transcripts)
        
        # Save summary
        output_dir = self.summary_agent.save_summary(summary)
        
        return output_dir
    
    def run_daily_job(self) -> None:
        """
        Run as a daily scheduled job.
        This would typically be called by a scheduler (e.g., cron, Windows Task Scheduler).
        """
        logger.info("Running daily scheduled job")
        
        # Check if it's time to generate summary
        current_time = datetime.now().time()
        scheduled_time_str = self.config['summary']['daily_summary_time']
        scheduled_time = datetime.strptime(scheduled_time_str, "%H:%M").time()
        
        if current_time.hour != scheduled_time.hour:
            logger.info(f"Not time for summary yet. Scheduled for {scheduled_time_str}")
            return
        
        # Find today's transcript file
        today = datetime.now().date()
        transcript_dir = Path(self.config.get('system', {}).get('output_dir', 'data/transcripts'))
        transcript_file = transcript_dir / f"transcripts_{today.strftime('%Y%m%d')}.json"
        
        if not transcript_file.exists():
            logger.warning(f"No transcript file found for today: {transcript_file}")
            return
        
        # Generate summary
        self.run_summary_only(str(transcript_file))
        
        logger.info("Daily job completed")
    
    def cleanup(self) -> None:
        """Clean up all agents."""
        logger.info("Cleaning up pipeline orchestrator...")
        
        if self.recording_agent:
            self.recording_agent.cleanup()
        
        if self.audio_file_agent:
            self.audio_file_agent.cleanup()
        
        if self.vad_agent:
            self.vad_agent.cleanup()

        if self.diarization_agent:
            self.diarization_agent.cleanup()

        if self.transcription_agent:
            self.transcription_agent.cleanup()

        if self.summary_agent:
            self.summary_agent.cleanup()

        logger.info("Cleanup completed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
