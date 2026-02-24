"""
Interactive CLI for the LLM-powered audio processing agent.

Usage:
    python agent.py
    python agent.py --config config/settings.yaml
    python agent.py --log-level DEBUG
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging. Chat output goes to a log file; only warnings appear on screen."""
    Path("logs").mkdir(exist_ok=True)

    # Force UTF-8 on Windows console before creating any handlers
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    file_handler = logging.FileHandler(
        f"logs/agent_{datetime.now().strftime('%Y%m%d')}.log",
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))

    # Only WARNING+ goes to the terminal so it doesn't clutter the chat
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.WARNING)

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[file_handler, stream_handler],
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LLM Agent for ADHD Audio Processing System"
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)

    print("=" * 60)
    print("  ADHD Audio Agent")
    print("  On-Device LLM — Privacy Preserved")
    print("=" * 60)
    print("\nLoading model... (this may take a minute)\n")

    from agents.llm_agent import LLMAgent

    agent = LLMAgent(config_path=args.config)
    try:
        agent.load()
    except Exception as exc:
        print(f"[ERROR] Failed to load agent: {exc}")
        return 1

    print("Agent ready. Type 'quit' or 'exit' to stop.\n")
    print("Example commands:")
    print("  - Transcribe data/audio.wav")
    print("  - Generate summary outputs/daily_reports/audio/transcripts_20260212.json")
    print("  - List all output files")
    print("  - Run the full pipeline on data/audio.wav")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        try:
            response = agent.chat(user_input)
            print(f"\nAgent: {response}")
            print("-" * 60)
        except Exception as exc:
            print(f"\n[ERROR] {exc}\n")

    agent.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
