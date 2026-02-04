"""
Simple test script to verify the system is set up correctly.
"""

import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from models.base import AudioSegment, TranscriptSegment, DailySummary
        logger.info("[OK] Base models imported")
        
        from models.vad_models import create_vad_model
        logger.info("[OK] VAD models imported")
        
        from models.asr_models import create_asr_model
        logger.info("[OK] ASR models imported")
        
        from models.llm_models import create_llm_model
        logger.info("[OK] LLM models imported")
        
        from agents.recording_agent import RecordingAgent
        logger.info("[OK] Recording agent imported")
        
        from agents.vad_transcription_agents import VADAgent, TranscriptionAgent
        logger.info("[OK] VAD and Transcription agents imported")
        
        from agents.summary_agent import SummaryAgent
        logger.info("[OK] Summary agent imported")
        
        from pipeline.orchestrator import PipelineOrchestrator
        logger.info("[OK] Pipeline orchestrator imported")
        
        return True
    
    except ImportError as e:
        logger.error(f"[ERROR] Import failed: {e}")
        return False


def test_config():
    """Test that configuration can be loaded."""
    logger.info("\nTesting configuration...")
    
    try:
        import yaml
        with open('config/settings.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info("[OK] Configuration loaded successfully")
        logger.info(f"  - ASR model: {config['asr']['model_name']}")
        logger.info(f"  - LLM model: {config['llm']['model_name']}")
        logger.info(f"  - VAD model: {config['vad']['model']}")
        
        return True
    
    except Exception as e:
        logger.error(f"[ERROR] Configuration loading failed: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are installed."""
    logger.info("\nTesting dependencies...")
    
    dependencies = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'yaml': 'PyYAML',
        'torch': 'PyTorch',
        'transformers': 'Transformers',
    }
    
    all_ok = True
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            logger.info(f"[OK] {name} installed")
        except ImportError:
            logger.error(f"[ERROR] {name} NOT installed")
            all_ok = False
    
    # Optional dependencies
    optional_deps = {
        'faster_whisper': 'faster-whisper (for ASR)',
        'sounddevice': 'sounddevice (for audio recording)',
        'librosa': 'librosa (for audio processing)',
    }
    
    logger.info("\nOptional dependencies:")
    for module, name in optional_deps.items():
        try:
            __import__(module)
            logger.info(f"[OK] {name} installed")
        except ImportError:
            logger.warning(f"[WARNING] {name} NOT installed (optional)")
        except Exception as e:
            # Handle ctranslate2 ROCm SDK error on Windows
            if module == 'faster_whisper' and 'rocm_sdk_core' in str(e):
                logger.warning(f"[WARNING] {name} installed but has ctranslate2 issue (see docs/FIX_CTRANSLATE2.md)")
                logger.warning(f"   Quick fix: pip uninstall faster-whisper ctranslate2 -y && pip install openai-whisper")
            else:
                logger.warning(f"[WARNING] {name} has import error: {e}")
    
    # Check for alternative whisper implementation
    try:
        import whisper as openai_whisper
        logger.info("[OK] openai-whisper (alternative ASR) installed")
    except ImportError:
        pass
    
    return all_ok


def test_directories():
    """Test that required directories exist."""
    logger.info("\nTesting directories...")
    
    from pathlib import Path
    
    directories = [
        'config',
        'models',
        'agents',
        'pipeline',
        'data',
        'data/audio_segments',
        'data/transcripts',
        'outputs',
        'outputs/daily_reports',
        'logs',
    ]
    
    all_ok = True
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if dir_path.exists():
            logger.info(f"[OK] {dir_name}/ exists")
        else:
            logger.warning(f"[WARNING] {dir_name}/ does NOT exist (will be created automatically)")
    
    return all_ok


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("ADHD Audio Summarization System - Setup Test")
    logger.info("=" * 60)
    
    results = {
        'imports': test_imports(),
        'config': test_config(),
        'dependencies': test_dependencies(),
        'directories': test_directories(),
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("Test Results Summary")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "[OK] PASSED" if result else "[ERROR] FAILED"
        logger.info(f"{test_name.capitalize()}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n[OK] All tests passed! System is ready to use.")
        logger.info("\nNext steps:")
        logger.info("1. Install missing dependencies: pip install -r requirements.txt")
        logger.info("2. Run example: python examples.py")
        logger.info("3. Or run main pipeline: python main.py --mode full --audio record --duration 60")
        return 0
    else:
        logger.error("\n[ERROR] Some tests failed. Please install missing dependencies.")
        logger.error("Run: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
