# Audio Format Support Guide

## Overview

The ADHD Audio Summarization System supports **two modes of audio input**:

1. **Live Recording Mode** - Record audio directly from microphone
2. **Audio File Mode** - Process pre-recorded audio files

## Supported Audio Formats

### ✅ Fully Supported Formats

| Format | Extension | Typical Use Case | Notes |
|--------|-----------|------------------|-------|
| **WAV** | .wav | Professional recording | ⭐ **Recommended** - No conversion, fastest processing |
| **MP3** | .mp3 | Consumer recording | Most common format, good compression |
| **FLAC** | .flac | High-quality archival | Lossless compression, large files |
| **M4A** | .m4a | Apple devices | iPhone/iPad voice memos |
| **OGG** | .ogg | Open-source | Vorbis codec |
| **AAC** | .aac | Modern devices | Advanced Audio Coding |
| **WMA** | .wma | Windows devices | Windows Media Audio |
| **OPUS** | .opus | Modern codec | Low latency, high quality |

### How It Works

The system uses `librosa` library for audio loading, which:
- **Automatically detects** the audio format
- **Converts** to the required format (16kHz mono)
- **Resamples** if necessary
- **Normalizes** audio levels

**You don't need to do anything special** - just provide the file path!

## Usage Examples

### Command Line

```bash
# WAV file (fastest, recommended)
python main.py --mode full --audio recording.wav

# MP3 file (most common)
python main.py --mode full --audio recording.mp3

# FLAC file (high quality)
python main.py --mode full --audio recording.flac

# M4A file (iPhone recordings)
python main.py --mode full --audio voice_memo.m4a

# OGG file
python main.py --mode full --audio recording.ogg

# With full path
python main.py --mode full --audio /path/to/recordings/day1.mp3

# Relative path
python main.py --mode full --audio ../recordings/interview.wav
```

### Python API

```python
from pipeline.orchestrator import PipelineOrchestrator

# Process any audio format
with PipelineOrchestrator() as orchestrator:
    # WAV
    orchestrator.run_full_pipeline(audio_source="recording.wav")
    
    # MP3
    orchestrator.run_full_pipeline(audio_source="recording.mp3")
    
    # FLAC
    orchestrator.run_full_pipeline(audio_source="recording.flac")
    
    # M4A (iPhone)
    orchestrator.run_full_pipeline(audio_source="voice_memo.m4a")
```

### Batch Processing

```python
from pipeline.orchestrator import PipelineOrchestrator
from pathlib import Path

# Process all audio files in a directory (multiple formats)
audio_dir = Path("recordings")

# Collect all supported formats
audio_files = []
for extension in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg', '*.aac']:
    audio_files.extend(audio_dir.glob(extension))

print(f"Found {len(audio_files)} audio files")

# Process each file
with PipelineOrchestrator() as orchestrator:
    for audio_file in audio_files:
        print(f"Processing {audio_file.name} ({audio_file.suffix})...")
        output_dir = orchestrator.run_full_pipeline(
            audio_source=str(audio_file)
        )
        print(f"✓ Report saved to: {output_dir}\n")
```

## Audio Specifications

### Input Requirements

**Flexible - System Auto-Converts:**
- **Sample Rate**: Any (automatically resampled to 16kHz)
- **Channels**: Mono or Stereo (automatically converted to mono)
- **Bit Depth**: Any (automatically normalized)
- **Duration**: Any length (processed in chunks)

### Processing Pipeline

```
Input Audio (any format)
    ↓
librosa.load() → Converts to 16kHz mono float32
    ↓
Voice Activity Detection (VAD)
    ↓
Speech Segments Extracted
    ↓
Whisper ASR (requires 16kHz mono)
    ↓
Transcribed Text
    ↓
LLM Summary Generation
    ↓
Daily Report
```

## Performance Considerations

### Format Comparison

| Format | Load Speed | File Size (1 hour) | Recommended For |
|--------|-----------|-------------------|-----------------|
| WAV | ⚡ Fastest | ~600 MB | Local processing |
| MP3 | 🔄 Fast | ~50-100 MB | Storage, sharing |
| FLAC | 🔄 Fast | ~300-400 MB | Archival quality |
| M4A | 🔄 Fast | ~40-80 MB | iPhone users |
| OGG | 🔄 Medium | ~50-100 MB | Open source |

**Recommendation:**
- **For processing**: Use WAV if possible (no conversion overhead)
- **For storage**: Use MP3 or M4A (smaller files)
- **For archival**: Use FLAC (lossless, compressible)

### Conversion Overhead

Format conversion adds minimal time:
- **WAV**: ~0 seconds (no conversion)
- **MP3/M4A**: ~2-5 seconds per hour of audio
- **FLAC**: ~3-7 seconds per hour of audio

The bottleneck is **VAD and ASR**, not format conversion.

## Common Use Cases

### 1. iPhone Voice Memos (.m4a)

```bash
# Transfer voice memo from iPhone to computer
# Then process directly
python main.py --mode full --audio "Voice Memo 001.m4a"
```

### 2. Zoom/Teams Recordings (.mp4, .m4a)

```bash
# Most meeting software saves as MP4 with audio
# Extract audio first or use directly if audio-only
python main.py --mode full --audio zoom_meeting.m4a
```

### 3. Professional Recordings (.wav, .flac)

```bash
# High-quality recordings from professional equipment
python main.py --mode full --audio interview.flac
```

### 4. YouTube Downloads (.mp3, .m4a)

```bash
# Audio extracted from videos
python main.py --mode full --audio downloaded_audio.mp3
```

## Troubleshooting

### Issue: "Could not load audio file"

**Solution:**
```bash
# Verify file exists
ls -l your_audio_file.mp3

# Check if librosa can load it
python -c "import librosa; librosa.load('your_audio_file.mp3', sr=16000)"

# Ensure ffmpeg is installed
ffmpeg -version
```

### Issue: "Format not supported"

**Solution:**
```bash
# Install/reinstall audio libraries
pip install librosa soundfile audioread

# Ensure ffmpeg is installed (handles most formats)
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows:
conda install -c conda-forge ffmpeg
```

### Issue: Slow processing with large files

**Solution:**
```python
# Process in chunks or split large files
from pydub import AudioSegment

# Split large MP3 into smaller chunks
audio = AudioSegment.from_mp3("large_file.mp3")
chunk_length_ms = 3600000  # 1 hour

for i, chunk in enumerate(audio[::chunk_length_ms]):
    chunk.export(f"chunk_{i}.mp3", format="mp3")
    # Process each chunk separately
```

## Advanced: Format Conversion

If you need to convert formats before processing:

### Using FFmpeg (Command Line)

```bash
# MP3 to WAV (best for processing)
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# M4A to WAV
ffmpeg -i input.m4a -ar 16000 -ac 1 output.wav

# FLAC to WAV
ffmpeg -i input.flac -ar 16000 -ac 1 output.wav

# Batch convert all MP3s in a directory
for f in *.mp3; do ffmpeg -i "$f" -ar 16000 -ac 1 "${f%.mp3}.wav"; done
```

### Using Python

```python
from pydub import AudioSegment

# Convert MP3 to WAV
audio = AudioSegment.from_mp3("input.mp3")
audio = audio.set_frame_rate(16000).set_channels(1)
audio.export("output.wav", format="wav")

# Convert M4A to WAV
audio = AudioSegment.from_file("input.m4a", format="m4a")
audio = audio.set_frame_rate(16000).set_channels(1)
audio.export("output.wav", format="wav")
```

**Note:** Pre-conversion is optional - the system handles it automatically!

## Summary

✅ **Just use your audio files directly** - no conversion needed  
✅ **All major formats supported** - WAV, MP3, FLAC, M4A, OGG, etc.  
✅ **Automatic resampling** - to 16kHz mono  
✅ **Batch processing** - handle multiple files easily  
✅ **Cross-platform** - works on Linux, macOS, Windows  

**For best performance:** Use WAV files at 16kHz mono if possible  
**For convenience:** Use any format - system handles it automatically!
