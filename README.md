# On-Device Daily Audio Summarization System for Children with ADHD

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
│   ├── vad_agent.py           # Voice activity detection
│   ├── transcription_agent.py # Speech-to-text
│   └── summary_agent.py       # Daily summary generation
├── utils/
│   ├── __init__.py
│   ├── audio_utils.py         # Audio processing utilities
│   ├── time_utils.py          # Time-related utilities
│   └── storage.py             # Data persistence
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py        # Main pipeline orchestrator
├── outputs/
│   └── daily_reports/         # Generated daily reports
├── data/
│   ├── audio_segments/        # Detected speech segments
│   └── transcripts/           # Transcribed segments
├── main.py                     # Entry point
└── requirements.txt
```

## Key Features

1. **Modular Design**: Easy to swap ASR/LLM models by changing configuration
2. **Agent-Based Architecture**: Each component is an independent agent
3. **Privacy-First**: All processing happens locally
4. **Efficient Processing**: Only transcribes detected speech segments
5. **Clinical Focus**: Generates structured, time-anchored reports

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/settings.yaml` to customize:
- ASR model (Whisper variants)
- LLM model (Qwen2.5-7B, Llama3.1-8B, etc.)
- VAD sensitivity
- Processing parameters

## Usage

```bash
# Run the full pipeline
python main.py --mode full

# Run specific agents
python main.py --mode vad
python main.py --mode transcribe
python main.py --mode summarize
```

## Privacy & Ethics

- All data stays on device
- No cloud API calls
- Audio segments can be automatically deleted after processing
- System provides observations, not diagnoses
