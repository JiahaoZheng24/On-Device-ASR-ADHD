# 📁 Project Structure

## Changes Made

✅ **All documentation moved to `docs/` folder** (except README.md)
✅ **All content is now in English**

## Complete Project Structure

```
adhd_audio_system/
│
├── README.md                    # Main project overview (English)
├── PROJECT_STRUCTURE.md         # This file - project structure overview
│
├── docs/                        # 📚 All documentation here
│   ├── START_HERE.md           # Complete getting started guide ⭐
│   ├── QUICKREF.md             # Quick reference card
│   ├── INSTALL.md              # Detailed installation guide
│   ├── USAGE.md                # Usage examples and tips
│   ├── ARCHITECTURE.md         # System architecture documentation
│   └── DELIVERY.md             # Project delivery summary
│
├── config/
│   ├── __init__.py
│   └── settings.yaml           # Configuration file for all models
│
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── base.py                 # Abstract base classes
│   ├── vad_models.py           # VAD models (Silero, WebRTC)
│   ├── asr_models.py           # ASR models (Whisper)
│   └── llm_models.py           # LLM models (Qwen, Llama)
│
├── agents/                     # Agent implementations
│   ├── __init__.py
│   ├── recording_agent.py      # Audio recording agent
│   ├── vad_transcription_agents.py  # VAD and transcription agents
│   └── summary_agent.py        # Daily summary generation agent
│
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py         # Main pipeline orchestrator
│
├── utils/                      # Utility functions (currently empty)
│
├── data/                       # Data storage (created at runtime)
│   ├── audio_segments/         # Temporary speech segments
│   └── transcripts/            # Transcribed text files
│
├── outputs/                    # Output files
│   └── daily_reports/          # Generated daily reports
│
├── logs/                       # Log files (created at runtime)
│
├── install.sh                  # Automated installation script
├── test_setup.py              # Installation verification script
├── examples.py                # 8 usage examples
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
└── __init__.py                # Package initialization
```

## File Counts

- **Python files**: 18 files (~2400 lines of code)
- **Documentation**: 7 Markdown files
- **Configuration**: 1 YAML file
- **Scripts**: 2 shell/Python scripts
- **Total**: ~30 files

## Documentation Access

All documentation is now in the `docs/` folder:

```bash
# View documentation
cat docs/START_HERE.md      # Start here first ⭐
cat docs/QUICKREF.md        # Quick commands reference
cat docs/INSTALL.md         # Detailed installation guide
cat docs/USAGE.md           # Usage examples
cat docs/ARCHITECTURE.md    # System design details
cat docs/DELIVERY.md        # Project summary
```

## Quick Start

```bash
# 1. Extract
tar -xzf adhd_audio_system_final.tar.gz
cd adhd_audio_system

# 2. Read documentation
cat docs/START_HERE.md

# 3. Install
chmod +x install.sh
./install.sh

# 4. Run
conda activate adhd_audio
python main.py --mode full --audio record --duration 60
```

## Documentation Reading Order

1. **README.md** - Project overview and quick installation
2. **docs/START_HERE.md** - Complete getting started guide ⭐
3. **docs/QUICKREF.md** - Quick reference for common tasks
4. **docs/INSTALL.md** - Detailed installation with model locations
5. **docs/USAGE.md** - Advanced usage examples
6. **docs/ARCHITECTURE.md** - Deep dive into system design

## Key Files Explained

### Core Modules
- **models/base.py**: Abstract base classes for all models
- **models/*_models.py**: Concrete implementations (VAD, ASR, LLM)
- **agents/*.py**: Independent processing agents
- **pipeline/orchestrator.py**: Coordinates all agents

### Configuration
- **config/settings.yaml**: Single file to configure all models

### Entry Points
- **main.py**: Command-line interface
- **examples.py**: Python API examples
- **install.sh**: Automated setup

### Documentation
- **docs/START_HERE.md**: Best starting point for new users
- **docs/INSTALL.md**: Comprehensive installation guide
- **docs/QUICKREF.md**: Quick command reference

## All Content in English

✅ All `.md` files are now in English
✅ All code comments are in English  
✅ All configuration examples are in English
✅ All error messages are in English
✅ All documentation is in English
