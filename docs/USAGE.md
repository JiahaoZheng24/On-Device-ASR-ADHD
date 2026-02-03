# Usage Guide

## Audio Input Overview

The system supports **two modes of audio input**:

### 1. Live Recording Mode
Record audio directly from your microphone in real-time.

### 2. Audio File Mode  
Process pre-recorded audio files in various formats.

## Supported Audio Formats

✅ **All Major Audio Formats Supported:**

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | .wav | ⭐ **Recommended** - No conversion needed, fastest |
| MP3 | .mp3 | ✅ Popular compressed format |
| FLAC | .flac | ✅ Lossless, high quality |
| OGG | .ogg | ✅ Open-source format |
| M4A | .m4a | ✅ Apple/iTunes format |
| AAC | .aac | ✅ Advanced Audio Coding |
| WMA | .wma | ✅ Windows Media Audio |
| OPUS | .opus | ✅ Modern codec |

**How it works:**
- The system uses `librosa` which automatically converts all formats to the required format
- No manual conversion needed - just provide the file path
- All formats are automatically resampled to 16kHz mono (optimal for speech)

## Quick Start Guide

### Installation

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

**Option 1: Live Recording (Record → VAD → Transcribe → Summarize)**
```bash
# Record 5 minutes of audio
python main.py --mode full --audio record --duration 300

# Record 1 hour
python main.py --mode full --audio record --duration 3600
```

**Option 2: Process Audio Files**

Works with **any audio format** - the system handles conversion automatically:

```bash
# Process WAV file (fastest, no conversion)
python main.py --mode full --audio /path/to/recording.wav

# Process MP3 file
python main.py --mode full --audio /path/to/recording.mp3

# Process FLAC file (high quality)
python main.py --mode full --audio /path/to/recording.flac

# Process M4A file (iPhone recordings)
python main.py --mode full --audio /path/to/recording.m4a

# Process OGG file
python main.py --mode full --audio /path/to/recording.ogg

# Works with relative paths
python main.py --mode full --audio recordings/day1.mp3

# Works with absolute paths
python main.py --mode full --audio /home/user/audio/interview.wav
```

**Run Specific Stages:**
```bash
# VAD only - detect speech segments
python main.py --mode vad --audio recording.mp3

# Transcription only - convert speech to text
python main.py --mode transcribe --segments-dir data/audio_segments

# Summary only - generate report from transcripts
python main.py --mode summarize --transcript-file data/transcripts/transcripts_20240101.json
```

### Python API

**Process Single Audio File:**
```python
from pipeline.orchestrator import PipelineOrchestrator

# Process WAV file
with PipelineOrchestrator("config/settings.yaml") as orchestrator:
    output_dir = orchestrator.run_full_pipeline(
        audio_source="recordings/day1.wav"
    )
    print(f"Report saved to: {output_dir}")

# Process MP3 file
with PipelineOrchestrator("config/settings.yaml") as orchestrator:
    output_dir = orchestrator.run_full_pipeline(
        audio_source="recordings/day1.mp3"
    )
    print(f"Report saved to: {output_dir}")
```

**Process Multiple Audio Files:**
```python
from pipeline.orchestrator import PipelineOrchestrator
from pathlib import Path

# Get all audio files from a directory
audio_dir = Path("recordings")
audio_files = (
    list(audio_dir.glob("*.wav")) +
    list(audio_dir.glob("*.mp3")) +
    list(audio_dir.glob("*.flac")) +
    list(audio_dir.glob("*.m4a"))
)

print(f"Found {len(audio_files)} audio files to process")

# Process each file
with PipelineOrchestrator() as orchestrator:
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n[{i}/{len(audio_files)}] Processing: {audio_file.name}")
        
        try:
            output_dir = orchestrator.run_full_pipeline(
                audio_source=str(audio_file)
            )
            print(f"✓ Success! Report: {output_dir}")
        except Exception as e:
            print(f"✗ Failed: {e}")

print("\n✓ Batch processing complete!")
```

**Process with Custom Settings:**
```python
from pipeline.orchestrator import PipelineOrchestrator

orchestrator = PipelineOrchestrator("config/settings.yaml")

# Use larger Whisper model for better accuracy
orchestrator.config['asr']['model_name'] = 'small'

# Use Llama instead of Qwen
orchestrator.config['llm']['model_type'] = 'llama'
orchestrator.config['llm']['model_name'] = 'meta-llama/Llama-3.1-8B-Instruct'

orchestrator.initialize_agents()

# Process audio file
output_dir = orchestrator.run_full_pipeline(
    audio_source="important_recording.mp3"
)

orchestrator.cleanup()
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
