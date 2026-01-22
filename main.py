"""
Main entry point for the ADHD Audio Summarization System.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from pipeline.orchestrator import PipelineOrchestrator


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/app_{datetime.now().strftime("%Y%m%d")}.log')
        ]
    )
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="On-Device Audio Summarization System for Children with ADHD"
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['full', 'vad', 'transcribe', 'summarize', 'daily'],
        default='full',
        help='Operation mode: full pipeline, VAD only, transcription only, summary only, or daily job'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/settings.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--audio',
        type=str,
        default='record',
        help='Audio source: "record" for live recording or path to audio file'
    )
    
    parser.add_argument(
        '--duration',
        type=float,
        default=300,
        help='Recording duration in seconds (only for live recording)'
    )
    
    parser.add_argument(
        '--segments-dir',
        type=str,
        help='Directory containing audio segments (for transcription-only mode)'
    )
    
    parser.add_argument(
        '--transcript-file',
        type=str,
        help='Path to transcript file (for summary-only mode)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("ADHD Audio Summarization System")
    logger.info("On-Device Processing - Privacy Preserved")
    logger.info("=" * 70)
    
    try:
        # Initialize pipeline orchestrator
        with PipelineOrchestrator(args.config) as orchestrator:
            
            if args.mode == 'full':
                logger.info("Running full pipeline...")
                output_dir = orchestrator.run_full_pipeline(
                    audio_source=args.audio,
                    duration=args.duration
                )
                
                if output_dir:
                    logger.info(f"\n✓ Success! Reports available at: {output_dir}")
                else:
                    logger.warning("\n✗ No output generated (no speech detected)")
            
            elif args.mode == 'vad':
                logger.info("Running VAD only...")
                segments = orchestrator.run_vad_only(args.audio, save_segments=True)
                logger.info(f"\n✓ Detected {len(segments)} speech segments")
            
            elif args.mode == 'transcribe':
                if not args.segments_dir:
                    logger.error("--segments-dir is required for transcription-only mode")
                    return 1
                
                logger.info("Running transcription only...")
                transcripts = orchestrator.run_transcription_only(args.segments_dir)
                logger.info(f"\n✓ Transcribed {len(transcripts)} segments")
            
            elif args.mode == 'summarize':
                if not args.transcript_file:
                    logger.error("--transcript-file is required for summary-only mode")
                    return 1
                
                logger.info("Running summary generation only...")
                output_dir = orchestrator.run_summary_only(args.transcript_file)
                logger.info(f"\n✓ Summary generated at: {output_dir}")
            
            elif args.mode == 'daily':
                logger.info("Running daily scheduled job...")
                orchestrator.run_daily_job()
        
        logger.info("\n" + "=" * 70)
        logger.info("Processing completed successfully")
        logger.info("=" * 70)
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("\n\nOperation cancelled by user")
        return 130
    
    except Exception as e:
        logger.error(f"\n\n✗ Error occurred: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
