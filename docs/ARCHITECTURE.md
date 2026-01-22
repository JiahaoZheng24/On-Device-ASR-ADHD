# System Architecture Documentation

## Overview

The ADHD Audio Summarization System is a modular, privacy-preserving pipeline for processing audio recordings and generating structured daily reports. The system is designed with three core principles:

1. **Privacy-First**: All processing happens on-device
2. **Modular Design**: Easy to swap models and components
3. **Clinical Focus**: Provides objective, time-structured observations

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                            │
│                    (CLI / Python API)                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Pipeline Orchestrator                           │
│  - Coordinates all agents                                       │
│  - Manages data flow                                            │
│  - Handles error recovery                                       │
└───────┬─────────┬─────────┬─────────┬───────────────────────────┘
        │         │         │         │
        ▼         ▼         ▼         ▼
    ┌──────┐ ┌──────┐ ┌────────┐ ┌────────┐
    │Record│ │ VAD  │ │Transcr.│ │Summary │
    │Agent │ │Agent │ │ Agent  │ │ Agent  │
    └───┬──┘ └───┬──┘ └────┬───┘ └────┬───┘
        │        │         │          │
        ▼        ▼         ▼          ▼
    ┌──────┐ ┌──────┐ ┌────────┐ ┌────────┐
    │Audio │ │Silero│ │Whisper │ │ Qwen/  │
    │Device│ │ VAD  │ │  ASR   │ │ Llama  │
    └──────┘ └──────┘ └────────┘ └────────┘
```

## Component Details

### 1. Models Layer (`models/`)

#### Base Classes (`base.py`)
- `BaseVADModel`: Abstract interface for VAD models
- `BaseASRModel`: Abstract interface for ASR models  
- `BaseLLMModel`: Abstract interface for LLMs
- `BaseAgent`: Abstract interface for agents

#### Data Structures
- `AudioSegment`: Represents detected speech with timing
- `TranscriptSegment`: Represents transcribed text with metadata
- `DailySummary`: Structured daily report

#### VAD Models (`vad_models.py`)
- **SileroVAD**: Neural network-based, high accuracy
  - Uses PyTorch Hub
  - Supports 16kHz audio
  - Configurable sensitivity
  
- **WebRTCVAD**: Rule-based, lightweight
  - Lower memory footprint
  - Faster processing
  - Limited to specific sample rates

**Factory Pattern:**
```python
vad_model = create_vad_model(config)
```

#### ASR Models (`asr_models.py`)
- **WhisperASR**: Uses faster-whisper implementation
  - Supports all Whisper model sizes (tiny → large)
  - CPU/GPU support
  - Int8/Float16 quantization
  - Batch processing
  
- **WhisperOriginalASR**: Uses original OpenAI implementation
  - Fallback option
  - Better compatibility

**Factory Pattern:**
```python
asr_model = create_asr_model(config)
```

#### LLM Models (`llm_models.py`)
- **QwenLLM**: Qwen2.5 model series
  - 7B parameter model recommended
  - 4-bit/8-bit quantization support
  - Structured JSON output
  
- **LlamaLLM**: Llama 3.1 model series
  - 8B parameter model recommended
  - Similar capabilities to Qwen

**Factory Pattern:**
```python
llm_model = create_llm_model(config)
```

### 2. Agents Layer (`agents/`)

Agents are autonomous components that perform specific tasks in the pipeline.

#### RecordingAgent (`recording_agent.py`)
**Purpose**: Capture audio from microphone or load from file

**Key Methods:**
- `execute(duration)`: Record audio for specified duration
- `start_continuous_recording()`: Background recording
- `save_audio()`: Save audio to WAV file

**Features:**
- Supports live recording via sounddevice
- Can load audio files via librosa
- Automatic format conversion

#### VADAgent (`vad_transcription_agents.py`)
**Purpose**: Detect speech segments in audio

**Key Methods:**
- `execute(audio)`: Detect speech, return segments
- `save_segments()`: Persist segments for later processing
- `load_segments()`: Load previously saved segments

**Features:**
- Filters silence and background noise
- Configurable sensitivity
- Reduces data by 80-95%

#### TranscriptionAgent (`vad_transcription_agents.py`)
**Purpose**: Convert speech to text

**Key Methods:**
- `execute(segments)`: Transcribe all segments
- `transcribe_single()`: Transcribe one segment
- `filter_transcripts()`: Filter by confidence/length
- `save_transcripts()`: Save to JSON

**Features:**
- Batch processing
- Confidence scoring
- JSON serialization

#### SummaryAgent (`summary_agent.py`)
**Purpose**: Generate daily reports from transcripts

**Key Methods:**
- `execute(transcripts)`: Generate DailySummary
- `save_summary()`: Save JSON + Markdown reports
- `generate_weekly_summary()`: Aggregate multiple days

**Features:**
- Temporal analysis
- Communication pattern detection
- Representative excerpt selection
- Human-readable reports

### 3. Pipeline Layer (`pipeline/`)

#### PipelineOrchestrator (`orchestrator.py`)
**Purpose**: Coordinate all agents and manage the complete workflow

**Key Methods:**
- `run_full_pipeline()`: End-to-end processing
- `run_vad_only()`: VAD stage only
- `run_transcription_only()`: Transcription stage only
- `run_summary_only()`: Summary stage only
- `run_daily_job()`: Scheduled daily execution

**Features:**
- Lazy agent initialization
- Error handling and recovery
- Context manager support
- Flexible execution modes

## Data Flow

### Full Pipeline Flow

```
1. Audio Input
   │
   ├─ Live Recording: sounddevice → numpy array
   └─ File Input: librosa → numpy array
   │
   ▼
2. Voice Activity Detection
   │
   ├─ Input: Full audio (e.g., 8 hours)
   ├─ Process: Silero/WebRTC VAD
   └─ Output: Speech segments (e.g., 45 minutes total)
   │
   ▼
3. Speech Transcription  
   │
   ├─ Input: Speech segments
   ├─ Process: Whisper ASR (batch)
   └─ Output: Text transcripts with timestamps
   │
   ▼
4. Daily Summary Generation
   │
   ├─ Input: Transcripts + temporal data
   ├─ Process: LLM analysis (Qwen/Llama)
   └─ Output: Structured report (JSON + Markdown)
```

### Data Persistence

```
data/
├── audio_segments/          # Temporary speech segments
│   └── segment_*.npy        # NumPy arrays
│
├── transcripts/             # Transcribed text
│   └── transcripts_YYYYMMDD.json
│
outputs/
└── daily_reports/           # Final reports
    ├── summary_YYYYMMDD.json
    └── report_YYYYMMDD.md
```

## Configuration System

### Configuration File Structure (`config/settings.yaml`)

```yaml
audio:           # Audio recording settings
vad:             # Voice activity detection
asr:             # Speech recognition
llm:             # Language model
summary:         # Report generation
retention:       # Data retention policies
system:          # System-wide settings
```

### Configuration Loading

1. Load YAML file
2. Validate required fields
3. Apply defaults for optional fields
4. Pass to agents during initialization

### Runtime Configuration Override

```python
orchestrator = PipelineOrchestrator("config/settings.yaml")
orchestrator.config['asr']['model_name'] = 'small'
orchestrator.initialize_agents()
```

## Model Swapping

### Switching ASR Models

```python
# In config/settings.yaml
asr:
  model_name: "base"  # Change to: tiny, small, medium, large

# Or at runtime
orchestrator.config['asr']['model_name'] = 'medium'
```

### Switching LLM Models

```python
# Qwen
llm:
  model_type: "qwen"
  model_name: "Qwen/Qwen2.5-7B-Instruct"

# Llama
llm:
  model_type: "llama"
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

### Adding New Models

1. **Create model class** inheriting from base
2. **Implement required methods** (load_model, process, etc.)
3. **Add to factory function**
4. **Update configuration options**

Example:
```python
# In models/asr_models.py
class NewASRModel(BaseASRModel):
    def load_model(self): ...
    def transcribe(self, segment): ...
    
# In factory function
def create_asr_model(config):
    if model_type == 'new_asr':
        return NewASRModel(config)
```

## Extensibility

### Adding New Agents

```python
from models.base import BaseAgent

class CustomAgent(BaseAgent):
    def initialize(self):
        # Setup
        pass
    
    def execute(self, *args):
        # Main logic
        return result
    
    def cleanup(self):
        # Cleanup
        pass
```

### Custom Pipeline Stages

```python
class CustomOrchestrator(PipelineOrchestrator):
    def run_custom_pipeline(self):
        # Your custom workflow
        audio = self.recording_agent.execute()
        # ... custom processing ...
        return result
```

### Plugin System (Future)

The modular design supports future plugin systems:
- Dynamic model loading
- Custom report templates
- Additional analysis modules
- Third-party integrations

## Performance Considerations

### Memory Usage

| Component | Memory (GB) | Notes |
|-----------|-------------|-------|
| Silero VAD | 0.1 | Very light |
| Whisper-base | 0.5 | On CPU |
| Whisper-small | 1.0 | On CPU |
| Qwen-7B (4-bit) | 4.5 | Quantized |
| Qwen-7B (full) | 14 | Full precision |

### Processing Speed

Approximate times on CPU (i7-10700):
- VAD: 0.1x realtime (10s audio → 1s processing)
- Whisper-base: 1-2x realtime
- Whisper-small: 2-3x realtime
- LLM Summary: 30-60s for full day

### Optimization Strategies

1. **Use quantization**: 4-bit for LLMs, int8 for Whisper
2. **Smaller models**: base Whisper + 7B LLM is good balance
3. **Batch processing**: Process segments in batches
4. **GPU acceleration**: 5-10x speedup with CUDA
5. **Cache segments**: Save VAD output for reprocessing

## Security & Privacy

### Privacy Features

1. **On-device processing**: No cloud APIs
2. **Local storage**: All data stays on disk
3. **Automatic cleanup**: Configurable retention
4. **No network access**: Models run offline

### Data Retention

Configure in `settings.yaml`:
```yaml
retention:
  keep_audio_segments: false  # Delete after transcription
  keep_transcripts: true      # Keep for analysis
  days_to_retain: 30          # Auto-delete old data
```

### Access Control

- Files stored with user permissions
- No external network requests
- Logs contain no sensitive data

## Testing & Validation

### Unit Tests
```bash
pytest tests/test_models.py
pytest tests/test_agents.py
```

### Integration Tests
```bash
python test_setup.py  # Verify setup
python examples.py     # Run examples
```

### Validation
```bash
python main.py --mode full --audio test_data/sample.wav
```

## Future Enhancements

1. **Multi-speaker Detection**: Identify different speakers
2. **Emotion Recognition**: Detect emotional states
3. **Real-time Processing**: Live transcription during recording
4. **Mobile Support**: Android/iOS versions
5. **Web Interface**: Browser-based UI
6. **Cloud Sync**: Optional encrypted backup
7. **Advanced Analytics**: Trend analysis over time

## Troubleshooting

See `USAGE.md` for detailed troubleshooting guide.

Common issues:
- Model download failures → Use local models
- Memory errors → Enable quantization
- Slow processing → Use smaller models or GPU
- Poor transcription → Adjust VAD threshold or use larger Whisper model
