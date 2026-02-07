# On-Device Daily Audio Summarization System for Children with ADHD

A privacy-preserving, fully on-device system that processes child-adult audio interactions for ADHD behavioral analysis. The system performs Voice Activity Detection (VAD), speaker diarization (child vs adult), speech recognition, and generates daily summary reports — all locally without cloud APIs.

## Pipeline Overview

```
Audio File (.wav/.mp3/.flac/...)
    |
    v
[1. VAD] Silero VAD - detect speech segments
    |
    v
[2. Diarization] MFCC + K-means + Pitch (pyin) - classify child vs adult
    |
    v
[3. ASR] OpenAI Whisper - transcribe each segment
    |
    v
[4. Summary] Qwen 7B LLM - generate daily report
    |
    v
Output: report.md + transcripts.json + summary.json
```

## Quick Start

```bash
# 1. Create environment
conda create -n adhd_audio python=3.10 -y
conda activate adhd_audio

# 2. Install ffmpeg
# Windows: conda install -c conda-forge ffmpeg -y
# Ubuntu:  sudo apt-get install -y ffmpeg
# macOS:   brew install ffmpeg

# 3. Install PyTorch with GPU support (recommended)
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# 3b. Or CPU-only (slower)
# pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run
python main.py --mode full --audio your_audio.wav
```

## Installation

### Step 1: Create Conda Environment

```bash
conda create -n adhd_audio python=3.10 -y
conda activate adhd_audio
```

### Step 2: Install System Dependencies

**Windows:**
```bash
conda install -c conda-forge ffmpeg -y
```

**Ubuntu/Debian:**
```bash
sudo apt-get update && sudo apt-get install -y ffmpeg portaudio19-dev
```

**macOS:**
```bash
brew install ffmpeg portaudio
```

### Step 3: Install PyTorch

**GPU with CUDA 12.8 (Recommended):**
```bash
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

**GPU with CUDA 12.6:**
```bash
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
```

**CPU Only:**
```bash
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0
```

> **Note:** Version pinning (`==2.8.0`) is important to avoid compatibility issues between torch, torchaudio, and torchvision.

### Step 4: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import whisper; print('Whisper:', whisper.__version__)"
```

## Usage

### Process Audio Files

```bash
conda activate adhd_audio

# Process a single audio file (full pipeline)
python main.py --mode full --audio recording.wav

# VAD only (detect speech segments)
python main.py --mode vad --audio recording.wav

# Record live audio (5 minutes)
python main.py --mode full --audio record --duration 300
```

### Supported Audio Formats

WAV, MP3, FLAC, M4A, OGG, AAC, WMA, OPUS — the system automatically handles format conversion.

## Configuration

Edit `config/settings.yaml`:

```yaml
# Speaker Diarization
diarization:
  enabled: true
  method: "simple"       # Local MFCC + K-means + pitch analysis
  n_speakers: 2          # Expected number of speakers
  child_pitch_threshold: 250.0  # Hz (fallback threshold)

# Speech Recognition
asr:
  model_type: "whisper"
  model_name: "base"     # Options: tiny, base, small, medium, large
  device: "cuda"         # Options: cpu, cuda

# Language Model (for summary generation)
llm:
  model_type: "qwen"
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  device: "cuda"         # Options: cpu, cuda
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `diarization.method` | `"simple"` (local) or `"pyannote"` (needs HF token) | `"simple"` |
| `diarization.n_speakers` | Expected number of speakers | `2` |
| `asr.model_name` | Whisper model size | `"base"` |
| `asr.device` | `"cuda"` for GPU, `"cpu"` for CPU | `"cuda"` |
| `llm.device` | `"cuda"` for GPU, `"cpu"` for CPU | `"cuda"` |

## Project Structure

```
On-Device-ASR-ADHD/
├── main.py                          # Entry point
├── config/
│   └── settings.yaml                # Configuration
├── models/
│   ├── base.py                      # Base classes
│   ├── vad_models.py                # Silero VAD
│   ├── asr_models.py                # Whisper ASR
│   ├── simple_diarization.py        # MFCC + K-means + pyin diarization
│   └── llm_models.py               # Qwen LLM for summaries
├── agents/
│   ├── recording_agent.py           # Audio file loading / recording
│   ├── vad_transcription_agents.py  # VAD & transcription agents
│   ├── diarization_agent.py         # Speaker diarization agent
│   └── summary_agent.py             # Summary generation agent
├── pipeline/
│   └── orchestrator.py              # Pipeline coordinator
├── outputs/daily_reports/           # Generated reports
├── requirements.txt                 # Dependencies
└── docs/                            # Additional documentation
```

## Speaker Diarization

The system uses a lightweight local approach for child vs adult classification:

1. **Feature Extraction**: MFCC (mean + std), pitch (pyin), spectral centroid, ZCR, RMS energy
2. **Clustering**: K-means groups segments into speaker clusters
3. **Classification**: Relative pitch comparison — the cluster with higher median pitch is labeled "child", lower is "adult"

This approach is completely local, requires no external API tokens, and works well for two-speaker child-adult interactions.

## Generated Reports

Reports are saved to `outputs/daily_reports/<filename>/`:

```
outputs/daily_reports/ENNIA413/
├── transcripts_20260207.json   # All transcripts with speaker labels
├── summary_20260207.json       # Machine-readable summary
└── report_20260207.md          # Human-readable report
```

### Report Contents

1. **Overview**: Total speech duration, speaker distribution (child vs adult)
2. **Speech Distribution**: When speech occurred in the audio
3. **Communication Patterns**: Descriptive observations
4. **Representative Excerpts**: Time-stamped examples with transcripts

## System Requirements

**Minimum (CPU):**
- Python 3.10+
- 8 GB RAM
- 5 GB disk space

**Recommended (GPU):**
- Python 3.10+
- NVIDIA GPU with 8+ GB VRAM
- CUDA 12.6+ driver
- 16 GB disk space

## Model Storage

Models auto-download on first run:

| Model | Size | Purpose |
|-------|------|---------|
| Silero VAD | ~2 MB | Voice activity detection |
| Whisper base | ~145 MB | Speech recognition |
| Qwen 7B | ~14 GB | Summary generation |

Storage location: `~/.cache/huggingface/hub/` and `~/.cache/torch/hub/`

## Privacy & Ethics

- All processing happens locally on-device
- No cloud API calls required
- No internet needed after model download
- Audio can be automatically deleted after processing
- Reports provide observations only, not medical diagnoses

## Troubleshooting

**CUDA not available:**
```bash
# Check your GPU
nvidia-smi

# Reinstall PyTorch with CUDA
pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
```

**Module not found errors:**
```bash
# Make sure you're in the correct environment
conda activate adhd_audio
pip install -r requirements.txt
```

**Out of memory:**
```yaml
# Use smaller models in config/settings.yaml
asr:
  model_name: "tiny"    # Smaller Whisper model
llm:
  load_in_4bit: true    # Enable quantization
```

## License

See project license file for details.
