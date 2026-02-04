# On-Device Daily Audio Summarization System for Children with ADHD

## 📚 Documentation

- **[START_HERE.md](docs/START_HERE.md)** - Complete getting started guide (READ THIS FIRST!)
- **[AUDIO_FORMATS.md](docs/AUDIO_FORMATS.md)** - Supported audio formats (WAV, MP3, FLAC, M4A, etc.)
- **[QUICKREF.md](docs/QUICKREF.md)** - Quick reference card
- **[INSTALL.md](docs/INSTALL.md)** - Detailed installation guide
- **[USAGE.md](docs/USAGE.md)** - Usage examples
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture
- **[DELIVERY.md](docs/DELIVERY.md)** - Project summary

## Quick Start

```bash
# 1. Create environment
conda create -n adhd_audio python=3.10 -y && conda activate adhd_audio

# 2. Install system dependencies
# Ubuntu/Debian: sudo apt-get install -y ffmpeg portaudio19-dev
# macOS: brew install ffmpeg portaudio
# Windows: conda install -c conda-forge ffmpeg -y

# 3. Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 4. Install project dependencies
cd On-Device-ASR-ADHD && pip install -r requirements.txt

# 5. Test installation
python test_setup.py

# 6. Run (process an audio file)
python main.py --mode full --audio your_audio.wav
```

## Installation

### Step 1: Create Conda Environment

```bash
# Create environment with Python 3.10
conda create -n adhd_audio python=3.10 -y

# Activate environment
conda activate adhd_audio
```

### Step 2: Install System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg portaudio19-dev
```

**macOS:**
```bash
brew install ffmpeg portaudio
```

**Windows:**
```bash
conda install -c conda-forge ffmpeg -y
```

### Step 3: Install PyTorch

**CPU Only (Recommended for most users):**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

**GPU with CUDA 11.8:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

**GPU with CUDA 12.1:**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Step 4: Install Python Dependencies

```bash
cd On-Device-ASR-ADHD
pip install -r requirements.txt
```

**What gets installed:**
- `numpy`, `scipy` - Numerical computing
- `openai-whisper` - Speech recognition (most compatible)
- `transformers`, `accelerate` - LLM support
- `sounddevice`, `librosa` - Audio processing
- `PyYAML` - Configuration

### Step 5: Verify Installation

```bash
python test_setup.py
```

✅ If all tests pass, you're ready to use the system!

## Key Features

- **Privacy-First**: All processing happens locally on your device
- **Multi-Format Support**: WAV, MP3, FLAC, M4A, OGG, AAC, and more
- **Modular Design**: Easy to swap ASR/LLM models
- **Agent Architecture**: Independent components for flexibility
- **Cross-Platform**: Works on Linux, macOS, and Windows
- **No Cloud APIs**: Complete on-device processing

## Supported Audio Formats

✅ **All Major Formats:**
- **WAV** (.wav) - Recommended, fastest processing
- **MP3** (.mp3) - Most common format
- **FLAC** (.flac) - Lossless quality
- **M4A** (.m4a) - iPhone/iTunes recordings
- **OGG**, **AAC**, **WMA**, **OPUS** - All supported

The system automatically handles format conversion. See [AUDIO_FORMATS.md](docs/AUDIO_FORMATS.md) for details.

## Usage

### Process Audio Files

```bash
# Activate environment first
conda activate adhd_audio

# Process any audio format
python main.py --mode full --audio recording.wav
python main.py --mode full --audio recording.mp3
python main.py --mode full --audio voice_memo.m4a

# Record live audio (5 minutes)
python main.py --mode full --audio record --duration 300
```

### Python API

```python
from pipeline.orchestrator import PipelineOrchestrator

# Process audio file
with PipelineOrchestrator() as orch:
    output_dir = orch.run_full_pipeline(
        audio_source="recording.mp3"
    )
    print(f"Report saved to: {output_dir}")
```

### Batch Processing

```python
from pipeline.orchestrator import PipelineOrchestrator
from pathlib import Path

# Process all MP3 files in a directory
audio_files = list(Path("recordings").glob("*.mp3"))

with PipelineOrchestrator() as orch:
    for audio_file in audio_files:
        print(f"Processing {audio_file.name}...")
        output_dir = orch.run_full_pipeline(
            audio_source=str(audio_file)
        )
        print(f"✓ Report: {output_dir}")
```

See [USAGE.md](docs/USAGE.md) for more examples.

## Configuration

Edit `config/settings.yaml` to customize models and parameters:

```yaml
# Speech Recognition
asr:
  model_name: "base"  # Options: tiny, base, small, medium, large
  device: "cpu"       # Options: cpu, cuda

# Language Model
llm:
  model_type: "qwen"  # Options: qwen, llama
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  load_in_4bit: true  # Reduce memory usage

# Voice Activity Detection
vad:
  model: "silero"     # Options: silero, webrtc
  threshold: 0.5      # Lower = more sensitive (0.0-1.0)
```

## Model Storage Locations

**Linux/macOS:**
```
~/.cache/
├── torch/hub/
│   └── snakers4_silero-vad_master/     # VAD (2 MB)
└── huggingface/hub/
    ├── models--openai--whisper-base/    # Whisper (145 MB)
    └── models--Qwen--Qwen2.5-7B-Instruct/ # Qwen (14 GB)
```

**Windows:**
```
C:\Users\<YourUsername>\.cache\
├── torch\hub\
│   └── snakers4_silero-vad_master\     # VAD (2 MB)
└── huggingface\hub\
    ├── models--openai--whisper-base\    # Whisper (145 MB)
    └── models--Qwen--Qwen2.5-7B-Instruct\ # Qwen (14 GB)
```

**Check downloaded models:**

Linux/macOS:
```bash
ls ~/.cache/torch/hub/
ls ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/hub/
```

Windows (PowerShell):
```powershell
dir $env:USERPROFILE\.cache\torch\hub\
dir $env:USERPROFILE\.cache\huggingface\hub\
```

## Project Structure

```
On-Device-ASR-ADHD/
├── README.md                    # This file
├── config/
│   └── settings.yaml           # Configuration
├── models/                     # Model implementations
│   ├── base.py
│   ├── vad_models.py
│   ├── asr_models.py
│   └── llm_models.py
├── agents/                     # Processing agents
│   ├── recording_agent.py
│   ├── vad_transcription_agents.py
│   └── summary_agent.py
├── pipeline/
│   └── orchestrator.py         # Main coordinator
├── docs/                       # Documentation
├── data/                       # Audio segments & transcripts
├── outputs/                    # Generated reports
├── main.py                     # Entry point
├── examples.py                 # Usage examples
├── test_setup.py              # Installation verification
└── requirements.txt           # Dependencies
```

## Generated Reports

Reports are saved to `outputs/daily_reports/`:

```
outputs/daily_reports/
├── report_20240101.md      # Human-readable markdown
└── summary_20240101.json   # Machine-readable JSON
```

### Report Contents

1. **Overview**: Total speech duration, segment count
2. **Temporal Distribution**: When speech occurred throughout the day
3. **Communication Patterns**: Descriptive observations
4. **Representative Excerpts**: 5-8 time-stamped examples with transcripts

## Disk Space Requirements

| Configuration | Total Size |
|--------------|-----------|
| **Minimal** (tiny Whisper + 1.5B LLM) | ~1.1 GB |
| **Recommended** (base Whisper + 7B LLM quantized) | ~4.7 GB |
| **High Quality** (small Whisper + 7B LLM full) | ~14.5 GB |

## System Requirements

**Minimum:**
- Python 3.10+
- 4 GB RAM
- 2 GB disk space

**Recommended:**
- Python 3.10+
- 8 GB RAM
- 5 GB disk space
- Multi-core CPU

**For GPU:**
- NVIDIA GPU with 4+ GB VRAM
- CUDA 11.8 or 12.1

## Privacy & Ethics

- ✅ All data stays on your device
- ✅ No cloud API calls
- ✅ No internet required after installation
- ✅ Audio can be automatically deleted after processing
- ✅ System provides observations, not medical diagnoses

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError` after installation
```bash
# Ensure you're in the correct environment
conda activate adhd_audio

# Reinstall dependencies
pip install -r requirements.txt
```

**Issue:** Audio file loading fails
```bash
# Ensure ffmpeg is installed
ffmpeg -version

# If not installed:
# Windows: conda install -c conda-forge ffmpeg
# Ubuntu: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
```

**Issue:** `No module named 'sounddevice'` (only needed for live recording)
```bash
conda install -c conda-forge sounddevice
```

**Issue:** Out of memory during LLM processing
```yaml
# Edit config/settings.yaml
llm:
  load_in_4bit: true  # Enable 4-bit quantization
```

### For More Help

- **Installation**: See [INSTALL.md](docs/INSTALL.md)
- **Audio formats**: See [AUDIO_FORMATS.md](docs/AUDIO_FORMATS.md)
- **Usage examples**: See [USAGE.md](docs/USAGE.md)
- **Architecture**: See [ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Examples

### Example 1: Process a Single File

```bash
python main.py --mode full --audio interview.mp3
```

### Example 2: Process Multiple Files

```bash
# Process all audio files in a directory
for file in recordings/*.wav; do
    python main.py --mode full --audio "$file"
done
```

### Example 3: Custom Configuration

```python
from pipeline.orchestrator import PipelineOrchestrator

# Initialize with custom settings
orch = PipelineOrchestrator()
orch.config['asr']['model_name'] = 'small'  # Better accuracy
orch.config['vad']['threshold'] = 0.3       # More sensitive
orch.initialize_agents()

# Process file
orch.run_full_pipeline(audio_source="recording.mp3")
orch.cleanup()
```

## What Happens on First Run?

Models will auto-download on first use:

1. **Silero VAD** (~2 MB) - Downloads in seconds
2. **Whisper base** (~145 MB) - Takes 1-2 minutes
3. **Qwen LLM** (~14 GB) - Takes 10-30 minutes depending on internet speed

Total first-run time: ~15-35 minutes (one-time only)

## Performance Tips

### For Faster Processing:
- Use `tiny` or `base` Whisper models
- Enable 4-bit quantization for LLM
- Process shorter audio segments
- Use CPU for small files, GPU for large batches

### For Better Accuracy:
- Use `small` or `medium` Whisper models
- Disable quantization (requires more RAM)
- Lower VAD threshold for more speech detection

## License

See project license file for details.

## Citation

If you use this system in your research or clinical practice, please cite appropriately.

## Support

For issues, questions, or contributions, please refer to the documentation in the `docs/` folder.
