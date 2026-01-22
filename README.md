# On-Device Daily Audio Summarization System for Children with ADHD

## 📚 Documentation

- **[START_HERE.md](docs/START_HERE.md)** - Complete getting started guide (READ THIS FIRST!)
- **[QUICKREF.md](docs/QUICKREF.md)** - Quick reference card with common commands
- **[INSTALL.md](docs/INSTALL.md)** - Detailed installation and model management guide
- **[USAGE.md](docs/USAGE.md)** - Usage examples and tips
- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture documentation
- **[DELIVERY.md](docs/DELIVERY.md)** - Project delivery summary

## Quick Navigation
- [Installation Guide](#installation-guide)
- [Model Locations](#where-models-are-stored)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)

## Installation Guide

### Step 1: Create Conda Environment

```bash
# Create a new conda environment with Python 3.10
conda create -n adhd_audio python=3.10 -y

# Activate the environment
conda activate adhd_audio
```

### Step 2: Install System Dependencies

#### On Ubuntu/Debian:
```bash
# Install ffmpeg and audio libraries
sudo apt-get update
sudo apt-get install -y ffmpeg portaudio19-dev

# Optional: Install build tools if needed
sudo apt-get install -y build-essential
```

#### On macOS:
```bash
# Install ffmpeg using Homebrew
brew install ffmpeg portaudio
```

#### On Windows:
```bash
# Install ffmpeg via conda (recommended on Windows)
conda install -c conda-forge ffmpeg

# Or download manually from: https://ffmpeg.org/download.html
# Add ffmpeg to your system PATH
```

### Step 3: Install PyTorch

Choose the appropriate command based on your system:

#### CPU Only (Recommended for most users):
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### CUDA 11.8 (For NVIDIA GPU):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### CUDA 12.1 (For newer NVIDIA GPU):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### Step 4: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
# Run the setup test script
python test_setup.py
```

If all tests pass, you're ready to go! ✅

## Where Models Are Stored

### 1. VAD Models (Voice Activity Detection)

**Silero VAD** (Default, Recommended):
- **Auto-downloaded** from PyTorch Hub on first use
- **Location**: `~/.cache/torch/hub/snakers4_silero-vad_master/`
- **Size**: ~2 MB
- **No manual download needed** - happens automatically

**WebRTC VAD** (Alternative):
- Built into the `webrtcvad` package
- No separate model files
- Very lightweight

### 2. ASR Models (Whisper)

**faster-whisper** (Default, Recommended):
- **Auto-downloaded** on first use
- **Location**: `~/.cache/huggingface/hub/`
- **Model sizes**:
  - `tiny`: ~75 MB
  - `base`: ~145 MB
  - `small`: ~466 MB
  - `medium`: ~1.5 GB
  - `large`: ~3 GB

**Pre-download Whisper models** (Optional):
```bash
# Pre-download to avoid waiting during first run
python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')"
```

**Custom model location** (Optional):
```yaml
# In config/settings.yaml
asr:
  model_path: "/path/to/your/whisper/model"
```

### 3. LLM Models (Qwen / Llama)

**Qwen2.5-7B-Instruct**:
- **Location**: `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/`
- **Size**: 
  - Full precision: ~14 GB
  - 4-bit quantized: ~4.5 GB (automatically done by system)

**Llama-3.1-8B-Instruct**:
- **Location**: `~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/`
- **Size**: 
  - Full precision: ~16 GB
  - 4-bit quantized: ~5 GB (automatically done by system)

**Pre-download LLM models** (Optional but recommended):

```bash
# For Qwen (no authentication needed)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# For Llama (requires HuggingFace account and access approval)
huggingface-cli login  # Enter your token
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

**Note**: Llama models require:
1. HuggingFace account: https://huggingface.co/join
2. Request access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. Generate token: https://huggingface.co/settings/tokens

**Use local model path** (Optional):
```yaml
# In config/settings.yaml
llm:
  model_path: "/path/to/your/downloaded/model"
```

### Model Storage Summary

```
~/.cache/
├── torch/hub/
│   └── snakers4_silero-vad_master/     # Silero VAD (~2 MB)
│
└── huggingface/hub/
    ├── models--Qwen--Qwen2.5-7B-Instruct/        # Qwen LLM (~14 GB)
    ├── models--meta-llama--Llama-3.1-8B-Instruct/  # Llama LLM (~16 GB)
    └── models--guillaumekln--faster-whisper-*/    # Whisper ASR (75MB-3GB)
```

**Total disk space needed**: ~20-30 GB (depending on model choices)

## Project Structure

```
adhd_audio_system/
├── config/
│   ├── __init__.py
│   └── settings.yaml          # Configuration file for models and parameters
├── models/
│   ├── __init__.py
│   ├── base.py                # Abstract base classes
│   ├── vad_models.py          # Voice Activity Detection implementations
│   ├── asr_models.py          # ASR model implementations (Whisper)
│   └── llm_models.py          # LLM implementations (Qwen, Llama)
├── agents/
│   ├── __init__.py
│   ├── recording_agent.py     # Continuous audio recording
│   ├── vad_transcription_agents.py  # VAD and transcription agents
│   └── summary_agent.py       # Daily summary generation
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py        # Main pipeline orchestrator
├── outputs/
│   └── daily_reports/         # Generated daily reports
├── data/
│   ├── audio_segments/        # Detected speech segments
│   └── transcripts/           # Transcribed segments
├── main.py                     # Entry point
├── examples.py                 # Usage examples
├── test_setup.py              # Installation verification
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── USAGE.md                   # Detailed usage guide
├── ARCHITECTURE.md            # System architecture documentation
└── DELIVERY.md                # Project delivery summary
```

## Key Features

1. **Modular Design**: Easy to swap ASR/LLM models by changing configuration
2. **Agent-Based Architecture**: Each component is an independent agent
3. **Privacy-First**: All processing happens locally
4. **Efficient Processing**: Only transcribes detected speech segments
5. **Clinical Focus**: Generates structured, time-anchored reports

## Configuration

Edit `config/settings.yaml` to customize:

### Switch ASR Model:
```yaml
asr:
  model_name: "base"  # Options: tiny, base, small, medium, large
  device: "cpu"       # Options: cpu, cuda
```

### Switch LLM Model:
```yaml
# Use Qwen (default, no auth needed)
llm:
  model_type: "qwen"
  model_name: "Qwen/Qwen2.5-7B-Instruct"

# Or use Llama (requires HuggingFace auth)
llm:
  model_type: "llama"
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

### Adjust VAD Sensitivity:
```yaml
vad:
  model: "silero"      # Options: silero, webrtc
  threshold: 0.5       # Lower = more sensitive (0.0-1.0)
```

## Usage

### Basic Usage

```bash
# Activate conda environment first
conda activate adhd_audio

# Run the full pipeline with live recording (5 minutes)
python main.py --mode full --audio record --duration 300

# Process an existing audio file
python main.py --mode full --audio /path/to/audio.wav

# Run specific stages
python main.py --mode vad --audio audio.wav
python main.py --mode transcribe --segments-dir data/audio_segments
python main.py --mode summarize --transcript-file data/transcripts/transcripts_20240101.json
```

### Python API Usage

```python
from pipeline.orchestrator import PipelineOrchestrator

# Initialize and run
with PipelineOrchestrator("config/settings.yaml") as orchestrator:
    output_dir = orchestrator.run_full_pipeline(
        audio_source="path/to/audio.wav"
    )
    print(f"Report saved to: {output_dir}")
```

## Quick Start Example

```bash
# 1. Setup environment
conda create -n adhd_audio python=3.10 -y
conda activate adhd_audio

# 2. Install dependencies
conda install pytorch cpuonly -c pytorch
pip install -r requirements.txt

# 3. Test installation
python test_setup.py

# 4. Run a quick test (1 minute recording)
python main.py --mode full --audio record --duration 60

# 5. View the generated report
cat outputs/daily_reports/report_*.md
```

## Privacy & Ethics

- All data stays on device
- No cloud API calls
- Audio segments can be automatically deleted after processing
- System provides observations, not diagnoses
