"""
Example usage scripts for the ADHD Audio Summarization System.
"""

from pathlib import Path
import logging

from pipeline.orchestrator import PipelineOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_1_full_pipeline_live_recording():
    """
    Example 1: Record live audio and run full pipeline.
    """
    logger.info("Example 1: Full pipeline with live recording")
    
    with PipelineOrchestrator("config/settings.yaml") as orchestrator:
        # Record 5 minutes of audio and process
        output_dir = orchestrator.run_full_pipeline(
            audio_source="record",
            duration=300  # 5 minutes
        )
        
        logger.info(f"Report saved to: {output_dir}")


def example_2_process_audio_file():
    """
    Example 2: Process an existing audio file.
    """
    logger.info("Example 2: Process audio file")
    
    audio_file = "path/to/your/audio.wav"
    
    with PipelineOrchestrator("config/settings.yaml") as orchestrator:
        output_dir = orchestrator.run_full_pipeline(
            audio_source=audio_file
        )
        
        logger.info(f"Report saved to: {output_dir}")


def example_3_vad_only():
    """
    Example 3: Run only Voice Activity Detection.
    """
    logger.info("Example 3: VAD only")
    
    with PipelineOrchestrator("config/settings.yaml") as orchestrator:
        segments = orchestrator.run_vad_only(
            audio_source="path/to/audio.wav",
            save_segments=True
        )
        
        logger.info(f"Detected {len(segments)} speech segments")


def example_4_transcribe_segments():
    """
    Example 4: Transcribe pre-detected segments.
    """
    logger.info("Example 4: Transcribe segments")
    
    with PipelineOrchestrator("config/settings.yaml") as orchestrator:
        transcripts = orchestrator.run_transcription_only(
            segments_dir="data/audio_segments"
        )
        
        logger.info(f"Transcribed {len(transcripts)} segments")


def example_5_generate_summary():
    """
    Example 5: Generate summary from existing transcripts.
    """
    logger.info("Example 5: Generate summary")
    
    with PipelineOrchestrator("config/settings.yaml") as orchestrator:
        output_dir = orchestrator.run_summary_only(
            transcript_file="data/transcripts/transcripts_20240101.json"
        )
        
        logger.info(f"Summary saved to: {output_dir}")


def example_6_custom_configuration():
    """
    Example 6: Use custom configuration.
    """
    logger.info("Example 6: Custom configuration")
    
    # You can modify config before creating orchestrator
    from pipeline.orchestrator import PipelineOrchestrator
    
    orchestrator = PipelineOrchestrator("config/settings.yaml")
    
    # Override specific settings
    orchestrator.config['asr']['model_name'] = 'small'  # Use larger Whisper model
    orchestrator.config['llm']['temperature'] = 0.5     # Lower temperature for more focused summaries
    
    orchestrator.initialize_agents()
    
    output_dir = orchestrator.run_full_pipeline(
        audio_source="path/to/audio.wav"
    )
    
    logger.info(f"Report saved to: {output_dir}")
    orchestrator.cleanup()


def example_7_switching_models():
    """
    Example 7: Demonstrate switching between models.
    """
    logger.info("Example 7: Switching models")
    
    # Use Qwen
    orchestrator = PipelineOrchestrator("config/settings.yaml")
    orchestrator.config['llm']['model_type'] = 'qwen'
    orchestrator.config['llm']['model_name'] = 'Qwen/Qwen2.5-7B-Instruct'
    
    # Or use Llama
    # orchestrator.config['llm']['model_type'] = 'llama'
    # orchestrator.config['llm']['model_name'] = 'meta-llama/Llama-3.1-8B-Instruct'
    
    # Use different Whisper model size
    orchestrator.config['asr']['model_name'] = 'medium'  # tiny, base, small, medium, large
    
    orchestrator.initialize_agents()
    output_dir = orchestrator.run_full_pipeline(audio_source="path/to/audio.wav")
    
    logger.info(f"Report saved to: {output_dir}")
    orchestrator.cleanup()


def example_8_batch_processing():
    """
    Example 8: Process multiple audio files in batch.
    """
    logger.info("Example 8: Batch processing")
    
    audio_files = [
        "data/raw_audio/day1.wav",
        "data/raw_audio/day2.wav",
        "data/raw_audio/day3.wav"
    ]
    
    with PipelineOrchestrator("config/settings.yaml") as orchestrator:
        for audio_file in audio_files:
            logger.info(f"Processing: {audio_file}")
            
            try:
                output_dir = orchestrator.run_full_pipeline(
                    audio_source=audio_file
                )
                logger.info(f"✓ Completed: {output_dir}")
            
            except Exception as e:
                logger.error(f"✗ Failed to process {audio_file}: {e}")


if __name__ == "__main__":
    # Run examples
    print("ADHD Audio Summarization System - Examples")
    print("=" * 60)
    print("\nAvailable examples:")
    print("1. Full pipeline with live recording")
    print("2. Process audio file")
    print("3. VAD only")
    print("4. Transcribe segments")
    print("5. Generate summary")
    print("6. Custom configuration")
    print("7. Switching models")
    print("8. Batch processing")
    print("\nEdit this file to run specific examples.")
    
    # Uncomment to run an example:
    # example_1_full_pipeline_live_recording()
    # example_2_process_audio_file()
    # example_3_vad_only()
