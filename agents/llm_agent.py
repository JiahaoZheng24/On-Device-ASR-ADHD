"""
LLM Agent — natural language interface to the audio processing pipeline.

Uses Qwen2.5 native function calling so the LLM decides which tool to
invoke based on what the user asks.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (Qwen2.5 / OpenAI-compatible format)
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "transcribe_audio",
            "description": (
                "Transcribe speech in an audio file (wav / mp3 / m4a) to text "
                "and save the result as a transcript JSON file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to the audio file to transcribe.",
                    }
                },
                "required": ["audio_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_summary",
            "description": (
                "Generate an LLM-powered summary report from an existing "
                "transcript JSON file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "transcript_path": {
                        "type": "string",
                        "description": "Path to the transcript JSON file.",
                    }
                },
                "required": ["transcript_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_full_pipeline",
            "description": (
                "Run the complete pipeline on an audio file: transcription + "
                "LLM summary report in one step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "audio_path": {
                        "type": "string",
                        "description": "Path to the audio file.",
                    }
                },
                "required": ["audio_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_output_files",
            "description": (
                "List all available transcript JSON and report Markdown files "
                "in the outputs directory."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful audio processing assistant for an ADHD child home "
    "monitoring system. You help users transcribe audio recordings, generate "
    "summary reports, and manage output files. All processing is done locally "
    "— data never leaves the device.\n\n"
    "When the user asks you to do something, use the available tools to "
    "accomplish the task. Respond in the same language as the user "
    "(Chinese or English)."
)

# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class LLMAgent:
    """
    Conversational agent powered by Qwen2.5 function calling.

    The LLM decides which pipeline tool to call; this class dispatches the
    call and feeds the result back so the model can reply in natural language.
    """

    def __init__(self, config_path: str = "config/settings.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        self.config_path = config_path

        self.llm = None
        self.orchestrator = None

        # Full conversation history including system prompt
        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the LLM and set up the pipeline orchestrator."""
        from models.llm_models import create_llm_model
        from pipeline.orchestrator import PipelineOrchestrator

        logger.info("Loading LLM for agent...")
        self.llm = create_llm_model(self.config.get("llm", {}))
        self.llm.load_model()

        logger.info("Setting up pipeline orchestrator...")
        self.orchestrator = PipelineOrchestrator(self.config_path)
        logger.info("Agent ready.")

    def cleanup(self) -> None:
        """Release resources."""
        if self.orchestrator:
            self.orchestrator.cleanup()

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> str:
        """
        Process one conversational turn.

        Appends the user message, lets the LLM decide whether to call a tool,
        executes it, feeds the result back, and returns the final reply.
        """
        self.messages.append({"role": "user", "content": user_message})

        # Allow up to 3 tool-call rounds per turn
        for _ in range(3):
            raw = self.llm.generate_with_tools(self.messages, TOOLS)

            match = _TOOL_CALL_RE.search(raw)
            if not match:
                # Plain assistant response — no tool call
                response = raw.strip()
                self.messages.append({"role": "assistant", "content": response})
                return response

            # --- parse tool call ---
            try:
                call = json.loads(match.group(1).strip())
                tool_name: str = call["name"]
                tool_args: Dict = call.get("arguments", {})
            except (json.JSONDecodeError, KeyError) as exc:
                logger.error("Failed to parse tool call: %s", exc)
                err = "Sorry, I encountered an error processing the tool call."
                self.messages.append({"role": "assistant", "content": err})
                return err

            logger.info("Tool call: %s(%s)", tool_name, tool_args)

            # Record the assistant's tool call in history
            self.messages.append({"role": "assistant", "content": raw.strip()})

            # Execute tool and capture result
            result = self._execute_tool(tool_name, tool_args)
            logger.info("Tool result: %s", result[:120])

            # Feed result back as a tool message
            self.messages.append(
                {"role": "tool", "content": result, "name": tool_name}
            )

        # Exhausted rounds — give a safe fallback
        fallback = (
            "I've finished processing your request. "
            "Please check the output files for results."
        )
        self.messages.append({"role": "assistant", "content": fallback})
        return fallback

    # ------------------------------------------------------------------
    # Tool dispatcher
    # ------------------------------------------------------------------

    def _execute_tool(self, name: str, args: Dict) -> str:
        try:
            if name == "transcribe_audio":
                return self._tool_transcribe_audio(args["audio_path"])
            elif name == "generate_summary":
                return self._tool_generate_summary(args["transcript_path"])
            elif name == "run_full_pipeline":
                return self._tool_run_full_pipeline(args["audio_path"])
            elif name == "list_output_files":
                return self._tool_list_output_files()
            else:
                return f"Unknown tool: {name}"
        except Exception as exc:
            logger.error("Tool '%s' failed: %s", name, exc, exc_info=True)
            return f"Tool execution failed: {exc}"

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_transcribe_audio(self, audio_path: str) -> str:
        """Transcribe only — no LLM summary."""
        path = Path(audio_path)
        if not path.exists():
            return f"Error: audio file not found: {audio_path}"

        orc = self.orchestrator

        # Lazily initialise required agents
        if orc.audio_file_agent is None:
            orc.initialize_agents(["audio_file"])
        if orc.vad_agent is None:
            orc.initialize_agents(["vad"])
        if orc.transcription_agent is None:
            orc.initialize_agents(["transcription"])

        audio, sample_rate = orc.audio_file_agent.execute(path)
        speech_segments = orc.vad_agent.execute(audio)

        if not speech_segments:
            return "No speech detected in the audio file."

        transcripts = orc.transcription_agent.execute(speech_segments)
        if not transcripts:
            return "Transcription produced no text."

        # Save under outputs/<stem>/
        stem = "".join(c for c in path.stem if c.isalnum() or c in ("_", "-"))
        out_dir = Path(self.config["system"]["output_dir"]) / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = orc.transcription_agent.save_transcripts(
            transcripts, output_subdir=out_dir
        )

        total_words = sum(len(t.text.split()) for t in transcripts)
        return (
            f"Transcription complete: {len(transcripts)} segments, "
            f"~{total_words} words.\n"
            f"Saved to: {transcript_path}"
        )

    def _tool_generate_summary(self, transcript_path: str) -> str:
        """Generate a summary from an existing transcript JSON."""
        path = Path(transcript_path)
        if not path.exists():
            return f"Error: transcript file not found: {transcript_path}"

        orc = self.orchestrator
        if orc.summary_agent is None:
            orc.initialize_agents(["summary"])
        if orc.transcription_agent is None:
            orc.initialize_agents(["transcription"])

        transcripts = orc.transcription_agent.load_transcripts(path)
        if not transcripts:
            return "No transcripts found in the file."

        summary = orc.summary_agent.execute(transcripts)
        out_dir = orc.summary_agent.save_summary(summary)

        return (
            f"Summary generated: {len(transcripts)} segments, "
            f"{summary.total_speech_duration / 60:.1f} min of speech.\n"
            f"Report saved to: {out_dir}"
        )

    def _tool_run_full_pipeline(self, audio_path: str) -> str:
        """Run full pipeline: transcription + summary."""
        if not Path(audio_path).exists():
            return f"Error: audio file not found: {audio_path}"

        # Reset so run_full_pipeline re-initialises all required agents cleanly
        self.orchestrator.is_initialized = False
        out_dir = self.orchestrator.run_full_pipeline(audio_source=audio_path)

        if out_dir:
            return f"Full pipeline complete. Reports saved to: {out_dir}"
        return "Pipeline finished but no output generated (no speech detected)."

    def _tool_list_output_files(self) -> str:
        """List output JSON and Markdown files."""
        root = Path(self.config["system"]["output_dir"])
        if not root.exists():
            return "No output directory found. Run the pipeline first."

        files = sorted(
            f for f in root.rglob("*")
            if f.is_file() and f.suffix in (".json", ".md")
        )
        if not files:
            return "No output files found yet."

        lines = [f"  {f.relative_to(root.parent)}" for f in files]
        return "Available output files:\n" + "\n".join(lines)
