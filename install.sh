#!/bin/bash
# Automated installation script for ADHD Audio Summarization System
# This script automates the entire installation process

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=================================================="
echo "ADHD Audio Summarization System - Installation"
echo "=================================================="
echo ""

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_error "Conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

print_status "Conda found"

# Create conda environment
ENV_NAME="adhd_audio"
echo ""
echo "Step 1: Creating conda environment '$ENV_NAME'..."

if conda env list | grep -q "^$ENV_NAME "; then
    print_warning "Environment '$ENV_NAME' already exists"
    read -p "Do you want to remove it and create a new one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
        print_status "Old environment removed"
    else
        print_warning "Using existing environment"
    fi
fi

if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -n $ENV_NAME python=3.10 -y
    print_status "Conda environment created"
fi

# Activate environment
echo ""
echo "Step 2: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME
print_status "Environment activated"

# Install system dependencies
echo ""
echo "Step 3: Installing system dependencies..."

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    print_status "Detected Linux system"
    
    if command -v apt-get &> /dev/null; then
        echo "Installing ffmpeg and portaudio..."
        sudo apt-get update -qq
        sudo apt-get install -y ffmpeg portaudio19-dev build-essential
        print_status "System dependencies installed"
    else
        print_warning "apt-get not found. Please install ffmpeg and portaudio manually"
    fi
    
elif [[ "$OSTYPE" == "darwin"* ]]; then
    print_status "Detected macOS system"
    
    if command -v brew &> /dev/null; then
        echo "Installing ffmpeg and portaudio..."
        brew install ffmpeg portaudio
        print_status "System dependencies installed"
    else
        print_warning "Homebrew not found. Please install ffmpeg and portaudio manually"
    fi
    
else
    print_warning "Unsupported OS. Please install ffmpeg manually"
fi

# Verify ffmpeg
if command -v ffmpeg &> /dev/null; then
    print_status "ffmpeg is available"
else
    print_error "ffmpeg not found. Please install it manually"
fi

# Install PyTorch
echo ""
echo "Step 4: Installing PyTorch..."
echo "Select hardware:"
echo "1) CPU only (recommended for most users)"
echo "2) GPU with CUDA 11.8"
echo "3) GPU with CUDA 12.1"
read -p "Enter choice (1-3) [default: 1]: " pytorch_choice
pytorch_choice=${pytorch_choice:-1}

case $pytorch_choice in
    1)
        print_status "Installing PyTorch for CPU"
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        ;;
    2)
        print_status "Installing PyTorch with CUDA 11.8"
        conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
        ;;
    3)
        print_status "Installing PyTorch with CUDA 12.1"
        conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
        ;;
    *)
        print_warning "Invalid choice. Installing CPU version"
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        ;;
esac

print_status "PyTorch installed"

# Install Python packages
echo ""
echo "Step 5: Installing Python packages..."
pip install -r requirements.txt
print_status "Python packages installed"

# Test installation
echo ""
echo "Step 6: Testing installation..."
python test_setup.py

if [ $? -eq 0 ]; then
    print_status "Installation test passed!"
else
    print_error "Installation test failed"
    exit 1
fi

# Pre-download models (optional)
echo ""
echo "Step 7: Pre-downloading models..."
echo "This will download ~5-15 GB of models. Do you want to continue?"
echo "You can also skip this and models will download automatically on first use."
read -p "Pre-download models now? (y/n) [default: n]: " download_models
download_models=${download_models:-n}

if [[ $download_models =~ ^[Yy]$ ]]; then
    echo ""
    echo "Downloading Whisper base model..."
    python -c "
import whisper
print('Downloading Whisper base model...')
model = whisper.load_model('base')
print('Whisper base model downloaded!')
"
    print_status "Whisper model downloaded"
    
    echo ""
    echo "Downloading Qwen2.5-7B-Instruct model..."
    echo "This may take 10-30 minutes depending on your internet speed..."
    python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print('Downloading Qwen2.5-7B-Instruct...')
print('This will download ~14 GB of data...')

tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    trust_remote_code=True
)
print('Tokenizer downloaded')

# Download model with 4-bit quantization to save disk space
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    trust_remote_code=True,
    torch_dtype=torch.float16
)
print('Model downloaded!')
"
    print_status "Qwen model downloaded"
else
    print_warning "Skipping model pre-download"
    echo "Models will be automatically downloaded on first use"
fi

# Print summary
echo ""
echo "=================================================="
echo "Installation Complete!"
echo "=================================================="
echo ""
echo "Summary of installed models:"
echo "  - VAD: Silero VAD (will download on first use)"
echo "  - ASR: Whisper base ($([ "$download_models" = "y" ] && echo "✓ downloaded" || echo "will download on first use"))"
echo "  - LLM: Qwen2.5-7B ($([ "$download_models" = "y" ] && echo "✓ downloaded" || echo "will download on first use"))"
echo ""
echo "To use the system:"
echo "  1. Activate environment: conda activate $ENV_NAME"
echo "  2. Run test: python main.py --mode full --audio record --duration 60"
echo "  3. Check output: cat outputs/daily_reports/report_*.md"
echo ""
echo "Documentation:"
echo "  - Quick start: cat QUICKREF.md"
echo "  - Full guide: cat INSTALL.md"
echo ""
echo "To change models, edit: config/settings.yaml"
echo ""

print_status "All done! Happy analyzing!"
