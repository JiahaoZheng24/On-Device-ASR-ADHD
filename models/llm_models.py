"""
Large Language Model (LLM) implementations for summary generation.
Supports Qwen2.5 and Llama3.1 models.
"""

import json
from typing import List, Dict, Any
from datetime import datetime
from collections import defaultdict
import logging

from models.base import BaseLLMModel, TranscriptSegment, DailySummary

logger = logging.getLogger(__name__)


class QwenLLM(BaseLLMModel):
    """Qwen2.5 LLM implementation for summary generation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load Qwen2.5 model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_name = self.config.get('model_name', 'Qwen/Qwen2.5-7B-Instruct')
            model_path = self.config.get('model_path', model_name)
            device = self.config.get('device', 'cpu')
            
            logger.info(f"Loading Qwen model: {model_path} on {device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            
            # Load model with quantization if specified
            load_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }
            
            if self.config.get('load_in_8bit', False):
                load_kwargs['load_in_8bit'] = True
            elif self.config.get('load_in_4bit', False):
                load_kwargs['load_in_4bit'] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            if device == "cuda":
                self.model = self.model.cuda()
            
            self._is_initialized = True
            logger.info("Qwen model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Qwen model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Qwen model."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        max_length = kwargs.get('max_length', self.config.get('max_length', 4096))
        temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
        top_p = kwargs.get('top_p', self.config.get('top_p', 0.9))
        
        # Qwen uses ChatML format
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer([text], return_tensors="pt")
        if self.config.get('device') == "cuda":
            inputs = inputs.to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1]
        response = response.replace("<|im_end|>", "").strip()
        
        return response
    
    def generate_summary(self, transcripts: List[TranscriptSegment]) -> DailySummary:
        """Generate daily summary from transcripts."""
        return self._generate_summary_common(transcripts)
    
    def _generate_summary_common(self, transcripts: List[TranscriptSegment]) -> DailySummary:
        """Common summary generation logic."""
        if not transcripts:
            logger.warning("No transcripts to summarize")
            return self._create_empty_summary()
        
        # Analyze temporal distribution
        temporal_dist = self._analyze_temporal_distribution(transcripts)
        
        # Calculate basic statistics
        total_duration = sum(t.end_time - t.start_time for t in transcripts)
        
        # Prepare transcript data for LLM
        transcript_data = self._prepare_transcript_data(transcripts)
        
        # Generate summary using LLM
        prompt = self._create_summary_prompt(transcript_data, temporal_dist, total_duration)
        
        try:
            llm_response = self.generate(prompt, max_length=2048)
            summary_content = self._parse_llm_response(llm_response, transcripts)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            summary_content = self._create_fallback_summary(transcripts, temporal_dist)
        
        # Create DailySummary object
        date = transcripts[0].timestamp.date() if transcripts else datetime.now().date()
        
        return DailySummary(
            date=datetime.combine(date, datetime.min.time()),
            total_speech_duration=total_duration,
            segment_count=len(transcripts),
            temporal_distribution=temporal_dist,
            communication_patterns=summary_content['communication_patterns'],
            representative_excerpts=summary_content['representative_excerpts'],
            metadata={
                'model': self.config.get('model_name'),
                'total_transcripts': len(transcripts),
                'average_confidence': sum(t.confidence for t in transcripts) / len(transcripts)
            }
        )
    
    @staticmethod
    def _analyze_temporal_distribution(transcripts: List[TranscriptSegment]) -> Dict[int, float]:
        """Analyze speech distribution across hours."""
        hourly_duration = defaultdict(float)
        
        for t in transcripts:
            hour = t.timestamp.hour
            duration = t.end_time - t.start_time
            hourly_duration[hour] += duration
        
        return dict(hourly_duration)
    
    @staticmethod
    def _prepare_transcript_data(transcripts: List[TranscriptSegment]) -> str:
        """Prepare transcript data for LLM prompt."""
        # Group transcripts by hour for better structure
        hourly_transcripts = defaultdict(list)
        
        for t in transcripts:
            hour = t.timestamp.hour
            hourly_transcripts[hour].append(t)
        
        data_parts = []
        for hour in sorted(hourly_transcripts.keys()):
            segments = hourly_transcripts[hour]
            hour_text = f"\n=== {hour:02d}:00-{hour+1:02d}:00 ===\n"
            
            for seg in segments[:10]:  # Limit to 10 segments per hour
                time_str = f"[{seg.timestamp.strftime('%H:%M:%S')}]"
                hour_text += f"{time_str} {seg.text}\n"
            
            if len(segments) > 10:
                hour_text += f"... ({len(segments) - 10} more segments)\n"
            
            data_parts.append(hour_text)
        
        return "".join(data_parts)
    
    def _create_summary_prompt(self, transcript_data: str, 
                               temporal_dist: Dict[int, float], 
                               total_duration: float) -> str:
        """Create prompt for LLM summary generation."""
        
        prompt = f"""You are analyzing audio transcripts from a child's home environment to create a structured daily report for parents and clinicians.

IMPORTANT INSTRUCTIONS:
- Focus on DESCRIPTIVE observations, not interpretations or diagnoses
- Emphasize TEMPORAL PATTERNS (when speech occurred)
- Identify COMMUNICATION PATTERNS (brief exchanges vs. extended conversations)
- Select representative excerpts that show typical daily moments
- Always link observations to specific times
- Avoid labeling behaviors or making clinical judgments

TRANSCRIPT DATA:
{transcript_data}

TEMPORAL SUMMARY:
- Total speech duration: {total_duration/60:.1f} minutes
- Speech distributed across {len(temporal_dist)} hours

Please generate a JSON response with the following structure:
{{
  "communication_patterns": {{
    "primary_characteristics": ["description1", "description2"],
    "temporal_notes": "When was speech most concentrated?",
    "interaction_style": "Brief exchanges / Extended conversations / Mixed"
  }},
  "representative_excerpts": [
    {{
      "time": "HH:MM:SS",
      "text": "excerpt text",
      "context": "why this excerpt is representative"
    }}
  ]
}}

Generate 5-8 representative excerpts showing typical moments throughout the day.
"""
        return prompt
    
    @staticmethod
    def _parse_llm_response(response: str, transcripts: List[TranscriptSegment]) -> Dict[str, Any]:
        """Parse LLM JSON response."""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response
            
            data = json.loads(json_str.strip())
            return data
        except Exception as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return QwenLLM._create_fallback_summary(transcripts, {})
    
    @staticmethod
    def _create_fallback_summary(transcripts: List[TranscriptSegment], 
                                 temporal_dist: Dict[int, float]) -> Dict[str, Any]:
        """Create a basic summary if LLM fails."""
        # Select representative excerpts (evenly distributed)
        num_excerpts = min(8, len(transcripts))
        step = max(1, len(transcripts) // num_excerpts)
        
        excerpts = []
        for i in range(0, len(transcripts), step):
            if len(excerpts) >= num_excerpts:
                break
            t = transcripts[i]
            excerpts.append({
                "time": t.timestamp.strftime("%H:%M:%S"),
                "text": t.text[:200],  # Limit length
                "context": "Representative sample from this time period"
            })
        
        return {
            "communication_patterns": {
                "primary_characteristics": [
                    f"Total of {len(transcripts)} speech segments recorded",
                    f"Speech occurred across {len(temporal_dist)} different hours"
                ],
                "temporal_notes": "See temporal distribution for details",
                "interaction_style": "Mixed patterns observed"
            },
            "representative_excerpts": excerpts
        }
    
    @staticmethod
    def _create_empty_summary() -> DailySummary:
        """Create empty summary when no data available."""
        return DailySummary(
            date=datetime.now(),
            total_speech_duration=0.0,
            segment_count=0,
            temporal_distribution={},
            communication_patterns={
                "primary_characteristics": ["No speech detected today"],
                "temporal_notes": "No activity recorded",
                "interaction_style": "N/A"
            },
            representative_excerpts=[],
            metadata={}
        )


class LlamaLLM(BaseLLMModel):
    """Llama3.1 LLM implementation for summary generation."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        
    def load_model(self) -> None:
        """Load Llama3.1 model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_name = self.config.get('model_name', 'meta-llama/Llama-3.1-8B-Instruct')
            model_path = self.config.get('model_path', model_name)
            device = self.config.get('device', 'cpu')
            
            logger.info(f"Loading Llama model: {model_path} on {device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            load_kwargs = {
                "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
            }
            
            if self.config.get('load_in_8bit', False):
                load_kwargs['load_in_8bit'] = True
            elif self.config.get('load_in_4bit', False):
                load_kwargs['load_in_4bit'] = True
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **load_kwargs
            )
            
            if device == "cuda":
                self.model = self.model.cuda()
            
            self._is_initialized = True
            logger.info("Llama model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Llama model: {e}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Llama model."""
        if not self._is_initialized:
            raise RuntimeError("Model not initialized. Call load_model() first.")
        
        max_length = kwargs.get('max_length', self.config.get('max_length', 4096))
        temperature = kwargs.get('temperature', self.config.get('temperature', 0.7))
        top_p = kwargs.get('top_p', self.config.get('top_p', 0.9))
        
        # Llama uses specific chat format
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        if self.config.get('device') == "cuda":
            inputs = inputs.to("cuda")
        
        outputs = self.model.generate(
            inputs,
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "assistant" in response:
            parts = response.split("assistant")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        return response
    
    def generate_summary(self, transcripts: List[TranscriptSegment]) -> DailySummary:
        """Generate daily summary - uses same logic as Qwen."""
        # Reuse the common summary generation from QwenLLM
        qwen_instance = QwenLLM(self.config)
        qwen_instance._is_initialized = True
        qwen_instance.generate = self.generate
        return qwen_instance._generate_summary_common(transcripts)


def create_llm_model(config: Dict[str, Any]) -> BaseLLMModel:
    """
    Factory function to create LLM model based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized LLM model
    """
    model_type = config.get('model_type', 'qwen').lower()
    
    if model_type == 'qwen':
        return QwenLLM(config)
    elif model_type == 'llama':
        return LlamaLLM(config)
    else:
        raise ValueError(f"Unknown LLM model type: {model_type}")
