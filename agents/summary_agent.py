"""
Summary Agent - Generates daily summaries from transcripts.
"""

from typing import Dict, Any, List
from datetime import datetime
import logging
from pathlib import Path
import json

from models.base import BaseAgent, TranscriptSegment, DailySummary
from models.llm_models import create_llm_model

logger = logging.getLogger(__name__)


class SummaryAgent(BaseAgent):
    """Agent responsible for generating daily summaries."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("SummaryAgent", config)
        self.llm_config = config['llm']
        self.summary_config = config['summary']
        self.output_dir = Path(config['system']['output_dir'])
        self.llm_model = None
        
    def initialize(self) -> None:
        """Initialize the LLM model."""
        logger.info("Initializing SummaryAgent...")
        
        self.llm_model = create_llm_model(self.llm_config)
        self.llm_model.load_model()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.set_status("initialized")
        logger.info("SummaryAgent initialized successfully")
    
    def execute(self, transcripts: List[TranscriptSegment]) -> DailySummary:
        """
        Generate daily summary from transcripts.
        
        Args:
            transcripts: List of TranscriptSegment objects
            
        Returns:
            DailySummary object
        """
        if not transcripts:
            logger.warning("No transcripts to summarize")
            return self._create_empty_summary()
        
        logger.info(f"Generating summary for {len(transcripts)} transcripts...")
        self.set_status("processing")
        
        try:
            # Filter transcripts if configured
            min_words = self.summary_config.get('min_segment_length', 3)
            min_confidence = self.summary_config.get('min_confidence', 0.2)
            filtered_transcripts = [
                t for t in transcripts
                if len(t.text.split()) >= min_words and t.confidence >= min_confidence
            ]

            logger.info(f"Using {len(filtered_transcripts)} transcripts after filtering "
                       f"(min_words={min_words}, min_confidence={min_confidence})")
            
            # Generate summary using LLM
            summary = self.llm_model.generate_summary(filtered_transcripts)

            # Add filtering info to metadata
            summary.metadata['original_transcript_count'] = len(transcripts)
            summary.metadata['filtered_transcript_count'] = len(filtered_transcripts)
            summary.metadata['filter_settings'] = {
                'min_words': min_words,
                'min_confidence': min_confidence
            }

            logger.info("Summary generated successfully")
            self.set_status("idle")

            return summary
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            self.set_status("error")
            raise
    
    def save_summary(self, summary: DailySummary, input_filename: str = None) -> Path:
        """
        Save summary to files (JSON and markdown).

        Args:
            summary: DailySummary object
            input_filename: Original input audio filename (optional)

        Returns:
            Path to saved files directory
        """
        date_str = summary.date.strftime("%Y%m%d")

        # Build filename with optional input filename
        if input_filename:
            # Remove extension and sanitize filename
            from pathlib import Path as PathLib
            base_name = PathLib(input_filename).stem
            # Remove any characters that might cause issues
            base_name = "".join(c for c in base_name if c.isalnum() or c in ('_', '-'))
            file_suffix = f"{date_str}_{base_name}"
        else:
            file_suffix = date_str

        # Save JSON version
        json_path = self.output_dir / f"summary_{file_suffix}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved summary JSON to {json_path}")

        # Save human-readable markdown version
        md_path = self.output_dir / f"report_{file_suffix}.md"
        markdown_content = self._generate_markdown_report(summary)
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Saved summary report to {md_path}")
        
        return self.output_dir
    
    def _generate_markdown_report(self, summary: DailySummary) -> str:
        """
        Generate human-readable markdown report.
        
        Args:
            summary: DailySummary object
            
        Returns:
            Markdown formatted string
        """
        date_str = summary.date.strftime("%A, %B %d, %Y")
        
        # Header
        md = f"# Daily Communication Report\n\n"
        md += f"**Date:** {date_str}\n\n"
        md += "---\n\n"
        
        # Overview
        md += "## Overview\n\n"
        md += f"- **Total Speech Duration:** {summary.total_speech_duration / 60:.1f} minutes\n"

        # Show both original and filtered counts if available
        original_count = summary.metadata.get('original_transcript_count')
        filtered_count = summary.metadata.get('filtered_transcript_count')
        if original_count and filtered_count:
            md += f"- **Segments Analyzed:** {filtered_count} (filtered from {original_count} total)\n"
        else:
            md += f"- **Number of Segments:** {summary.segment_count}\n"

        md += f"- **Average Confidence:** {summary.metadata.get('average_confidence', 0):.2%}\n\n"
        
        # Temporal Distribution (by audio position in minutes)
        md += "## Speech Distribution in Audio\n\n"

        if summary.temporal_distribution:
            md += "| Audio Position | Speech Duration (seconds) |\n"
            md += "|----------------|---------------------------|\n"

            for minute in sorted(summary.temporal_distribution.keys()):
                duration_sec = summary.temporal_distribution[minute]
                time_str = f"{minute:02d}:00 - {minute+1:02d}:00"
                md += f"| {time_str} | {duration_sec:.1f} |\n"

            md += "\n"

            # Visual bar chart (ASCII)
            md += self._create_temporal_chart(summary.temporal_distribution)
            md += "\n"
        else:
            md += "*No temporal data available*\n\n"
        
        # Communication Patterns
        md += "## Communication Patterns\n\n"
        patterns = summary.communication_patterns
        
        if isinstance(patterns, dict):
            if 'primary_characteristics' in patterns:
                md += "### Primary Characteristics\n\n"
                for char in patterns['primary_characteristics']:
                    md += f"- {char}\n"
                md += "\n"
            
            if 'temporal_notes' in patterns:
                md += f"### Temporal Notes\n\n{patterns['temporal_notes']}\n\n"
            
            if 'interaction_style' in patterns:
                md += f"### Interaction Style\n\n{patterns['interaction_style']}\n\n"
        
        # Representative Excerpts
        md += "## Representative Excerpts\n\n"
        md += "*These excerpts illustrate typical moments throughout the day.*\n\n"
        
        for i, excerpt in enumerate(summary.representative_excerpts, 1):
            if isinstance(excerpt, dict):
                time = excerpt.get('time', 'Unknown')
                text = excerpt.get('text', '')
                context = excerpt.get('context', '')
                
                md += f"### Excerpt {i} - [{time}]\n\n"
                md += f"> {text}\n\n"
                if context:
                    md += f"*Context: {context}*\n\n"
        
        # Notes
        md += "---\n\n"
        md += "## Important Notes\n\n"
        md += "- This report provides **descriptive observations** only, not diagnoses or interpretations.\n"
        md += "- All observations are **time-anchored** and based on recorded speech patterns.\n"
        md += "- This information is intended to **support clinical conversations** between families and healthcare providers.\n"
        md += "- All data processing occurred **on-device** with no cloud uploads.\n\n"
        
        # Metadata
        md += "---\n\n"
        md += "*Generated with privacy-preserving on-device processing*\n"
        md += f"*Model: {summary.metadata.get('model', 'Unknown')}*\n"
        
        return md
    
    def _create_temporal_chart(self, temporal_dist: Dict[int, float]) -> str:
        """
        Create ASCII bar chart of temporal distribution (by audio minute).

        Args:
            temporal_dist: Dictionary of minute -> duration (seconds)

        Returns:
            ASCII chart string
        """
        if not temporal_dist:
            return ""

        max_duration = max(temporal_dist.values())
        max_bar_length = 40

        # Determine range of minutes to show
        min_minute = min(temporal_dist.keys())
        max_minute = max(temporal_dist.keys())

        chart = "### Speech Activity Chart (by audio minute)\n\n"
        chart += "```\n"

        for minute in range(min_minute, max_minute + 1):
            duration = temporal_dist.get(minute, 0)
            bar_length = int((duration / max_duration) * max_bar_length) if max_duration > 0 else 0
            bar = "█" * bar_length

            chart += f"{minute:02d}:00 | {bar} {duration:.1f}s\n"

        chart += "```\n"

        return chart
    
    def generate_weekly_summary(self, daily_summaries: List[DailySummary]) -> Dict[str, Any]:
        """
        Generate a weekly summary from multiple daily summaries.
        
        Args:
            daily_summaries: List of DailySummary objects
            
        Returns:
            Dictionary containing weekly summary data
        """
        if not daily_summaries:
            return {"error": "No daily summaries provided"}
        
        logger.info(f"Generating weekly summary from {len(daily_summaries)} daily summaries...")
        
        # Aggregate statistics
        total_duration = sum(s.total_speech_duration for s in daily_summaries)
        total_segments = sum(s.segment_count for s in daily_summaries)
        
        # Aggregate temporal distribution
        weekly_temporal = {}
        for summary in daily_summaries:
            for hour, duration in summary.temporal_distribution.items():
                weekly_temporal[hour] = weekly_temporal.get(hour, 0) + duration
        
        # Average temporal distribution
        for hour in weekly_temporal:
            weekly_temporal[hour] /= len(daily_summaries)
        
        weekly_summary = {
            "period": f"{daily_summaries[0].date.date()} to {daily_summaries[-1].date.date()}",
            "total_days": len(daily_summaries),
            "total_speech_duration": total_duration,
            "average_daily_duration": total_duration / len(daily_summaries),
            "total_segments": total_segments,
            "average_daily_segments": total_segments / len(daily_summaries),
            "weekly_temporal_distribution": weekly_temporal,
        }
        
        logger.info("Weekly summary generated")
        return weekly_summary
    
    @staticmethod
    def _create_empty_summary() -> DailySummary:
        """Create an empty summary for days with no data."""
        return DailySummary(
            date=datetime.now(),
            total_speech_duration=0.0,
            segment_count=0,
            temporal_distribution={},
            communication_patterns={
                "primary_characteristics": ["No speech detected"],
                "temporal_notes": "No activity recorded",
                "interaction_style": "N/A"
            },
            representative_excerpts=[],
            metadata={}
        )
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.llm_model = None
        self.set_status("cleanup")
        logger.info("SummaryAgent cleaned up")
