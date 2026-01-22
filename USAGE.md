# Quick Start Guide

## Installation

### 1. Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install python3-dev portaudio19-dev ffmpeg
```

**macOS:**
```bash
brew install portaudio ffmpeg
```

**Windows:**
- Install Python 3.8+
- Install ffmpeg from https://ffmpeg.org/

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Models (Optional)

If you want to use local models instead of downloading at runtime:

**Whisper Models:**
```python
# They will be downloaded automatically on first use
# Or pre-download:
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

**LLM Models:**
```bash
# For Qwen (requires ~14GB disk space for 7B model)
huggingface-cli login  # Optional, for gated models
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# For Llama (requires authentication)
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

## Basic Usage

### Command Line Interface

**Full Pipeline (Record → VAD → Transcribe → Summarize):**
```bash
python main.py --mode full --audio record --duration 300
```

**Process Audio File:**
```bash
python main.py --mode full --audio /path/to/audio.wav
```

**VAD Only:**
```bash
python main.py --mode vad --audio /path/to/audio.wav
```

**Transcription Only:**
```bash
python main.py --mode transcribe --segments-dir data/audio_segments
```

**Summary Only:**
```bash
python main.py --mode summarize --transcript-file data/transcripts/transcripts_20240101.json
```

### Python API

```python
from pipeline.orchestrator import PipelineOrchestrator

# Initialize orchestrator
with PipelineOrchestrator("config/settings.yaml") as orchestrator:
    # Run full pipeline
    output_dir = orchestrator.run_full_pipeline(
        audio_source="path/to/audio.wav"
    )
    print(f"Report saved to: {output_dir}")
```

## Configuration

Edit `config/settings.yaml` to customize:

### Switch ASR Model

```yaml
asr:
  model_type: "whisper"
  model_name: "base"  # Options: tiny, base, small, medium, large
  device: "cpu"       # or "cuda" for GPU
```

### Switch LLM Model

**For Qwen:**
```yaml
llm:
  model_type: "qwen"
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  device: "cpu"
  load_in_4bit: true  # Reduces memory usage
```

**For Llama:**
```yaml
llm:
  model_type: "llama"
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  device: "cpu"
  load_in_4bit: true
```

### Adjust VAD Sensitivity

```yaml
vad:
  model: "silero"
  threshold: 0.5      # Lower = more sensitive (0.0-1.0)
  min_speech_duration: 0.5  # Minimum speech segment length
```

## Output Format

### Daily Report Structure

```
outputs/daily_reports/
├── report_20240101.md       # Human-readable markdown report
└── summary_20240101.json    # Machine-readable JSON data
```

### Report Contents

1. **Overview**: Total speech duration, segment count
2. **Temporal Distribution**: When speech occurred throughout the day
3. **Communication Patterns**: Descriptive observations about interaction style
4. **Representative Excerpts**: 5-8 time-stamped examples

## Common Workflows

### Workflow 1: Daily Recording and Analysis

```bash
# Morning: Start recording
python main.py --mode full --audio record --duration 28800  # 8 hours

# Evening: Generate summary (if not auto-generated)
python main.py --mode summarize --transcript-file data/transcripts/transcripts_20240101.json
```

### Workflow 2: Batch Processing Multiple Days

```python
from pipeline.orchestrator import PipelineOrchestrator
from pathlib import Path

audio_files = list(Path("recordings").glob("*.wav"))

with PipelineOrchestrator() as orchestrator:
    for audio_file in audio_files:
        print(f"Processing {audio_file}...")
        orchestrator.run_full_pipeline(audio_source=str(audio_file))
```

### Workflow 3: Testing Different Models

```python
# Test different Whisper models
for model_size in ['tiny', 'base', 'small']:
    orchestrator = PipelineOrchestrator()
    orchestrator.config['asr']['model_name'] = model_size
    orchestrator.run_full_pipeline(audio_source="test_audio.wav")
    orchestrator.cleanup()
```

## Troubleshooting

### Issue: "Model download failed"

**Solution:** Check internet connection, or pre-download models:
```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"
```

### Issue: "CUDA out of memory"

**Solution:** Enable quantization in config:
```yaml
llm:
  load_in_4bit: true
  device: "cpu"  # Or use CPU
```

### Issue: "No audio input device found"

**Solution:** 
```bash
# List available devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Set default device in your script
import sounddevice as sd
sd.default.device = 0  # Use device ID from list
```

### Issue: "Transcription quality is poor"

**Solutions:**
1. Use larger Whisper model: `model_name: "medium"` or `"large"`
2. Ensure audio quality is good (16kHz, mono, clear speech)
3. Adjust VAD settings to capture more context

## Performance Optimization

### CPU Optimization

```yaml
asr:
  compute_type: "int8"  # Faster on CPU
  
llm:
  load_in_4bit: true    # Reduces memory usage
```

### GPU Optimization

```yaml
asr:
  device: "cuda"
  compute_type: "float16"  # Faster on GPU
  
llm:
  device: "cuda"
  load_in_4bit: true
```

### Memory Management

For systems with limited RAM:
1. Use smaller models: `whisper tiny` + `Qwen2.5-1.5B`
2. Enable 4-bit quantization
3. Process in smaller chunks
4. Delete audio segments after transcription

```yaml
retention:
  keep_audio_segments: false
```

## Privacy & Security

- **All processing is on-device**: No data sent to cloud
- **Audio retention**: Configure in `retention` section
- **Data location**: All data in `data/` and `outputs/` directories
- **Cleanup**: Use `--cleanup` flag or delete directories manually

```bash
# Remove all processed data
rm -rf data/audio_segments/*
rm -rf data/transcripts/*

# Keep only recent summaries
find outputs/daily_reports -name "*.md" -mtime +30 -delete
```

## Scheduling Daily Jobs

### Linux/macOS (cron)

```bash
# Edit crontab
crontab -e

# Add daily job at 11 PM
0 23 * * * cd /path/to/adhd_audio_system && python main.py --mode daily
```

### Windows (Task Scheduler)

Create a scheduled task to run:
```
python C:\path\to\adhd_audio_system\main.py --mode daily
```

## Advanced Usage

See `examples.py` for more advanced usage patterns including:
- Custom configuration
- Batch processing
- Model switching
- Error handling
- Performance tuning
