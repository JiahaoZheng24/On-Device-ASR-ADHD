# Installation and Model Management Guide

## Complete Installation Guide

### Step-by-Step Installation

#### 1. Create and Activate Conda Environment

```bash
# Create a new conda environment
conda create -n adhd_audio python=3.10 -y

# Activate the environment
conda activate adhd_audio

# Verify Python version
python --version  # Should show Python 3.10.x
```

#### 2. Install System Dependencies

**Ubuntu/Debian:**
```bash
# Update package list
sudo apt-get update

# Install ffmpeg (required for audio processing)
sudo apt-get install -y ffmpeg

# Install PortAudio (required for sounddevice)
sudo apt-get install -y portaudio19-dev

# Install build tools (may be needed for some packages)
sudo apt-get install -y build-essential

# Verify ffmpeg installation
ffmpeg -version
```

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install ffmpeg
brew install ffmpeg

# Install PortAudio
brew install portaudio

# Verify installation
ffmpeg -version
```

**Windows:**
```bash
# Option 1: Install via conda (recommended)
conda install -c conda-forge ffmpeg

# Option 2: Manual installation
# 1. Download ffmpeg from https://ffmpeg.org/download.html
# 2. Extract to C:\ffmpeg
# 3. Add C:\ffmpeg\bin to System PATH
# 4. Restart terminal and verify:
ffmpeg -version
```

#### 3. Install PyTorch

Choose based on your hardware:

**CPU Only (Works on all systems, recommended for most users):**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

**GPU with CUDA 11.8 (For NVIDIA GPUs):**
```bash
# Check CUDA version first
nvidia-smi

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify GPU is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**GPU with CUDA 12.1 (For newer NVIDIA GPUs):**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Verify
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

#### 4. Install Python Packages

```bash
# Navigate to project directory
cd adhd_audio_system

# Install all required packages
pip install -r requirements.txt

# This will install:
# - numpy, scipy (numerical computing)
# - faster-whisper (ASR)
# - transformers, accelerate (LLMs)
# - sounddevice, librosa (audio processing)
# - PyYAML (configuration)
# - and more...
```

#### 5. Verify Installation

```bash
# Run the test script
python test_setup.py

# Expected output:
# ✓ All imports successful
# ✓ Configuration loaded
# ✓ Dependencies installed
# ✓ Directories exist
```

## Model Download and Location Guide

### Understanding Model Storage

All models are automatically downloaded to your cache directory:
- **Linux/macOS**: `~/.cache/`
- **Windows**: `C:\Users\<YourUsername>\.cache\`

### 1. VAD Models

#### Silero VAD (Default, Recommended)

**Auto-Download:**
- First use downloads automatically via PyTorch Hub
- No manual action required

**Storage Location:**
```
~/.cache/torch/hub/
└── snakers4_silero-vad_master/
    ├── files/
    │   └── silero_vad.jit      # Main model file (~2 MB)
    └── hubconf.py
```

**Pre-download (Optional):**
```python
# Run this to download in advance
python -c "
import torch
model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad',
    force_reload=False
)
print('Silero VAD downloaded successfully!')
"
```

**Check if downloaded:**
```bash
ls -lh ~/.cache/torch/hub/snakers4_silero-vad_master/files/
```

#### WebRTC VAD (Alternative)

**Installation:**
```bash
pip install webrtcvad
```

**Storage:**
- Built into the package, no separate model files
- No download needed

### 2. ASR Models (Whisper)

#### faster-whisper (Default, Recommended)

**Auto-Download:**
- Downloads on first use of each model size
- Cached for subsequent uses

**Storage Location:**
```
~/.cache/huggingface/hub/
└── models--guillaumekln--faster-whisper-*/
    ├── snapshots/
    │   └── <hash>/
    │       ├── config.json
    │       ├── model.bin       # Main model
    │       ├── vocabulary.txt
    │       └── tokenizer.json
    └── refs/
```

**Model Sizes and Disk Space:**

| Model  | Disk Space | Memory (RAM) | Speed (CPU) |
|--------|-----------|--------------|-------------|
| tiny   | ~75 MB    | ~400 MB      | Very Fast   |
| base   | ~145 MB   | ~500 MB      | Fast        |
| small  | ~466 MB   | ~1 GB        | Medium      |
| medium | ~1.5 GB   | ~2.5 GB      | Slow        |
| large  | ~3 GB     | ~5 GB        | Very Slow   |

**Pre-download specific model:**
```python
from faster_whisper import WhisperModel

# Download base model (recommended)
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper base model downloaded!")

# Or download small model
model = WhisperModel("small", device="cpu", compute_type="int8")
print("Whisper small model downloaded!")
```

**Download all models at once:**
```bash
python -c "
from faster_whisper import WhisperModel
for size in ['tiny', 'base', 'small']:
    print(f'Downloading {size}...')
    WhisperModel(size, device='cpu', compute_type='int8')
    print(f'{size} downloaded!')
"
```

**Check downloaded models:**
```bash
ls -lh ~/.cache/huggingface/hub/ | grep faster-whisper
```

**Use custom model location:**
```yaml
# In config/settings.yaml
asr:
  model_path: "/path/to/your/whisper/model"
```

### 3. LLM Models (Qwen / Llama)

#### HuggingFace CLI Setup

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Login (needed for Llama, optional for Qwen)
huggingface-cli login
# Enter your token from: https://huggingface.co/settings/tokens
```

#### Qwen Models (Recommended, No Auth Required)

**Model: Qwen/Qwen2.5-7B-Instruct**

**Auto-Download:**
```python
# Will download on first use
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_4bit=True  # Uses ~4.5 GB instead of ~14 GB
)
```

**Pre-download (Recommended):**
```bash
# Download entire model
huggingface-cli download Qwen/Qwen2.5-7B-Instruct

# Or download with Python
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Qwen2.5-7B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
print('Download complete!')
"
```

**Storage Location:**
```
~/.cache/huggingface/hub/
└── models--Qwen--Qwen2.5-7B-Instruct/
    ├── snapshots/
    │   └── <hash>/
    │       ├── config.json
    │       ├── generation_config.json
    │       ├── model-*.safetensors      # Model weights (~14 GB total)
    │       ├── tokenizer.json
    │       └── tokenizer_config.json
    └── refs/
```

**Check if downloaded:**
```bash
ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/
```

#### Llama Models (Requires Authentication)

**Model: meta-llama/Llama-3.1-8B-Instruct**

**Prerequisites:**
1. Create HuggingFace account: https://huggingface.co/join
2. Request access: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
3. Wait for approval (usually instant to a few hours)
4. Generate access token: https://huggingface.co/settings/tokens

**Login:**
```bash
huggingface-cli login
# Paste your token when prompted
```

**Pre-download:**
```bash
# Download via CLI
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct

# Or via Python
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Llama-3.1-8B-Instruct...')
AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
print('Download complete!')
"
```

**Storage Location:**
```
~/.cache/huggingface/hub/
└── models--meta-llama--Llama-3.1-8B-Instruct/
    ├── snapshots/
    │   └── <hash>/
    │       ├── config.json
    │       ├── model-*.safetensors      # Model weights (~16 GB total)
    │       ├── tokenizer.json
    │       └── tokenizer_config.json
    └── refs/
```

#### Smaller LLM Options (For Limited Resources)

**Qwen2.5-1.5B-Instruct** (~3 GB):
```bash
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct
```

**Qwen2.5-3B-Instruct** (~6 GB):
```bash
huggingface-cli download Qwen/Qwen2.5-3B-Instruct
```

**Update config to use smaller model:**
```yaml
llm:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
```

### Using Custom Model Locations

If you want to use a different directory for models:

```yaml
# In config/settings.yaml
asr:
  model_path: "/mnt/models/whisper-base"

llm:
  model_path: "/mnt/models/Qwen2.5-7B-Instruct"
```

Or set environment variable:
```bash
export HF_HOME=/mnt/models/huggingface
export TORCH_HOME=/mnt/models/torch
```

## Disk Space Requirements

### Minimum Configuration
- Silero VAD: 2 MB
- Whisper tiny: 75 MB
- Qwen2.5-1.5B (4-bit): 1 GB
- **Total**: ~1.1 GB

### Recommended Configuration
- Silero VAD: 2 MB
- Whisper base: 145 MB
- Qwen2.5-7B (4-bit): 4.5 GB
- **Total**: ~4.7 GB

### High-Performance Configuration
- Silero VAD: 2 MB
- Whisper small: 466 MB
- Qwen2.5-7B (full): 14 GB
- **Total**: ~14.5 GB

### Maximum Configuration
- Silero VAD: 2 MB
- Whisper large: 3 GB
- Llama-3.1-8B (full): 16 GB
- **Total**: ~19 GB

## Model Download Verification

### Check All Models

```bash
# Check VAD model
ls ~/.cache/torch/hub/snakers4_silero-vad_master/files/

# Check Whisper models
ls ~/.cache/huggingface/hub/ | grep whisper

# Check LLM models
ls ~/.cache/huggingface/hub/ | grep -E "(Qwen|llama)"

# Check total disk usage
du -sh ~/.cache/torch/hub/
du -sh ~/.cache/huggingface/hub/
```

### Python Verification Script

```python
import os
from pathlib import Path

def check_model_locations():
    home = Path.home()
    
    # Check VAD
    vad_path = home / ".cache/torch/hub/snakers4_silero-vad_master"
    print(f"Silero VAD: {'✓ Found' if vad_path.exists() else '✗ Not found'}")
    
    # Check Whisper
    whisper_path = home / ".cache/huggingface/hub"
    whisper_models = list(whisper_path.glob("models--*faster-whisper*"))
    print(f"Whisper models: {len(whisper_models)} found")
    
    # Check Qwen
    qwen_models = list(whisper_path.glob("models--Qwen*"))
    print(f"Qwen models: {len(qwen_models)} found")
    
    # Check Llama
    llama_models = list(whisper_path.glob("models--*llama*"))
    print(f"Llama models: {len(llama_models)} found")

check_model_locations()
```

## Troubleshooting

### Issue: "Command 'ffmpeg' not found"

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (via conda)
conda install -c conda-forge ffmpeg
```

### Issue: "ModuleNotFoundError: No module named 'sounddevice'"

**Solution:**
```bash
# Install PortAudio first
# Ubuntu: sudo apt-get install portaudio19-dev
# macOS: brew install portaudio

# Then install sounddevice
pip install sounddevice
```

### Issue: Model download is slow

**Solution:**
```bash
# Use a mirror (for users in China)
export HF_ENDPOINT=https://hf-mirror.com

# Or download manually and use model_path in config
```

### Issue: "CUDA out of memory"

**Solution:**
```yaml
# Use quantization in config/settings.yaml
llm:
  load_in_4bit: true
  device: "cpu"  # Or use CPU instead
```

### Issue: Models are taking too much disk space

**Solution:**
```bash
# Remove unused model versions
huggingface-cli delete-cache

# Or manually delete specific models
rm -rf ~/.cache/huggingface/hub/models--<model-name>
```

## Complete Installation Script

```bash
#!/bin/bash
# complete_install.sh

echo "Creating conda environment..."
conda create -n adhd_audio python=3.10 -y
conda activate adhd_audio

echo "Installing system dependencies..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg portaudio19-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg portaudio
fi

echo "Installing PyTorch (CPU)..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

echo "Installing Python packages..."
pip install -r requirements.txt

echo "Downloading models (this may take a while)..."
python -c "
from faster_whisper import WhisperModel
print('Downloading Whisper base...')
WhisperModel('base', device='cpu', compute_type='int8')
print('Whisper downloaded!')
"

python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
print('Downloading Qwen2.5-7B...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
print('Qwen downloaded!')
"

echo "Verifying installation..."
python test_setup.py

echo "Installation complete! Activate with: conda activate adhd_audio"
```

Run with:
```bash
chmod +x complete_install.sh
./complete_install.sh
```
