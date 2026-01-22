# Quick Reference Card

## 🚀 Quick Start (Copy & Paste)

```bash
# 1. Create environment
conda create -n adhd_audio python=3.10 -y && conda activate adhd_audio

# 2. Install system deps (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y ffmpeg portaudio19-dev

# 3. Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# 4. Install Python packages
cd adhd_audio_system && pip install -r requirements.txt

# 5. Test
python test_setup.py

# 6. Run
python main.py --mode full --audio record --duration 60
```

## 📦 Model Storage Locations

```
~/.cache/
├── torch/hub/
│   └── snakers4_silero-vad_master/     # VAD model (2 MB)
│
└── huggingface/hub/
    ├── models--*faster-whisper*/        # Whisper ASR
    │   ├── tiny:   75 MB
    │   ├── base:   145 MB
    │   ├── small:  466 MB
    │   ├── medium: 1.5 GB
    │   └── large:  3 GB
    │
    ├── models--Qwen--Qwen2.5-7B-Instruct/      # Qwen LLM (14 GB)
    └── models--meta-llama--Llama-3.1-8B-Instruct/  # Llama LLM (16 GB)
```

## 🔧 Model Configuration

### Switch ASR Model
```yaml
# config/settings.yaml
asr:
  model_name: "base"  # tiny, base, small, medium, large
```

### Switch LLM Model
```yaml
# Qwen (no auth)
llm:
  model_type: "qwen"
  model_name: "Qwen/Qwen2.5-7B-Instruct"

# Llama (needs auth)
llm:
  model_type: "llama"
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

## 📥 Pre-download Models

```bash
# Whisper
python -c "from faster_whisper import WhisperModel; WhisperModel('base')"

# Qwen (no auth needed)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# Llama (auth needed)
huggingface-cli login  # Enter token
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
```

## 💻 Commands

```bash
# Full pipeline
python main.py --mode full --audio record --duration 300
python main.py --mode full --audio audio.wav

# Individual stages
python main.py --mode vad --audio audio.wav
python main.py --mode transcribe --segments-dir data/audio_segments
python main.py --mode summarize --transcript-file data/transcripts/transcripts_20240101.json
```

## 🐍 Python API

```python
from pipeline.orchestrator import PipelineOrchestrator

# Basic usage
with PipelineOrchestrator() as orch:
    orch.run_full_pipeline(audio_source="audio.wav")

# Custom config
orch = PipelineOrchestrator()
orch.config['asr']['model_name'] = 'small'
orch.config['llm']['model_type'] = 'llama'
orch.initialize_agents()
orch.run_full_pipeline(audio_source="audio.wav")
orch.cleanup()
```

## 🔍 Check Models

```bash
# VAD
ls ~/.cache/torch/hub/snakers4_silero-vad_master/files/

# Whisper
ls ~/.cache/huggingface/hub/ | grep whisper

# LLM
ls ~/.cache/huggingface/hub/ | grep -E "(Qwen|llama)"

# Disk usage
du -sh ~/.cache/torch/hub/
du -sh ~/.cache/huggingface/hub/
```

## 🎯 Recommended Configs

### Low Memory (~2 GB RAM)
```yaml
asr:
  model_name: "tiny"
llm:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  load_in_4bit: true
```

### Balanced (~8 GB RAM)
```yaml
asr:
  model_name: "base"
llm:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  load_in_4bit: true
```

### High Quality (~16 GB RAM)
```yaml
asr:
  model_name: "small"
llm:
  model_name: "Qwen/Qwen2.5-7B-Instruct"
  load_in_4bit: false
```

### GPU Accelerated
```yaml
asr:
  model_name: "medium"
  device: "cuda"
  compute_type: "float16"
llm:
  device: "cuda"
  load_in_4bit: true
```

## 🐛 Common Issues

### ffmpeg not found
```bash
# Ubuntu: sudo apt-get install ffmpeg
# macOS: brew install ffmpeg
# Windows: conda install -c conda-forge ffmpeg
```

### CUDA out of memory
```yaml
llm:
  load_in_4bit: true  # Enable in config
  device: "cpu"       # Or use CPU
```

### Model download slow
```bash
export HF_ENDPOINT=https://hf-mirror.com  # Use mirror
```

### Wrong conda environment
```bash
conda activate adhd_audio  # Always activate first
```

## 📊 Disk Space Guide

| Config | VAD | ASR | LLM | Total |
|--------|-----|-----|-----|-------|
| Minimal | 2 MB | 75 MB | 1 GB | ~1.1 GB |
| Recommended | 2 MB | 145 MB | 4.5 GB | ~4.7 GB |
| High Quality | 2 MB | 466 MB | 14 GB | ~14.5 GB |
| Maximum | 2 MB | 3 GB | 16 GB | ~19 GB |

## 🔗 Useful Links

- **Whisper Models**: https://huggingface.co/guillaumekln
- **Qwen Models**: https://huggingface.co/Qwen
- **Llama Access**: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- **HF Token**: https://huggingface.co/settings/tokens

## 📝 Output Files

```
outputs/daily_reports/
├── report_20240101.md      # Human-readable report
└── summary_20240101.json   # Machine-readable data

data/
├── audio_segments/         # Temporary speech segments
└── transcripts/
    └── transcripts_20240101.json  # Saved transcripts
```

## 🧪 Testing

```bash
# Test installation
python test_setup.py

# Run examples
python examples.py

# Quick test (1 min)
python main.py --mode full --audio record --duration 60
```

## 🎓 Documentation

- **README.md** - Project overview
- **INSTALL.md** - Detailed installation guide
- **USAGE.md** - Usage examples and tips
- **ARCHITECTURE.md** - System design details
- **DELIVERY.md** - Project summary
