"""Data analysis coding agent.

Wraps the lifted `interpreter.Interpreter` in an LLM generate -> run -> reflect
loop. Given an analysis request (and any input data files already on disk), it
asks the model to write a self-contained Python script, executes it, detects any
figures the script produced, shows them back to the (vision-capable) model for
reflection, and iterates a fixed number of times.

Pattern source: AI Scientist v2 `perform_plotting.py` (the aggregator/reflection
loop) and `perform_experiments.py` (run -> capture -> feed error back). The
execution itself is delegated to interpreter.py rather than `subprocess.run`.

Decisions
---------
- Figures are detected by diffing the working dir's image files before/after each
  run; newly appeared images are the produced figures and are surfaced both to
  the caller and (as vision input) to the model on the next reflection turn.
- Code is extracted with response.extract_code; long stdout/stderr is trimmed
  with response.trim_long_string before being fed back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llm import client_for, get_response_from_llm
from response import extract_code, trim_long_string
from interpreter import Interpreter
from config import ModelSpec

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass
class DataAgentResult:
    success: bool
    term_out: str
    figures: list[str] = field(default_factory=list)   # absolute paths
    code: str = ""


def _images_in(d: Path) -> set[Path]:
    return {p for p in d.rglob("*") if p.suffix.lower() in _IMAGE_EXTS}


class DataAgent:
    """One coding agent bound to a model spec and an execution timeout."""

    def __init__(self, spec: ModelSpec, system_prompt: str, *, timeout: int,
                 max_iters: int):
        self.spec = spec
        self.system_prompt = system_prompt
        self.timeout = timeout
        self.max_iters = max_iters

    def run(self, request: str, working_dir: Path, *, record=None) -> DataAgentResult:
        """Run the generate/execute/reflect loop in `working_dir`.

        `record(call_name, system, messages, response)` if given is called after
        every model turn so the caller can persist the raw conversation.
        """
        working_dir = Path(working_dir)
        working_dir.mkdir(parents=True, exist_ok=True)
        interp = Interpreter(working_dir, timeout=self.timeout)

        client, model = client_for(self.spec.model)
        history: list[dict] = []
        prompt = request
        last = DataAgentResult(success=False, term_out="")

        try:
            for i in range(self.max_iters):
                before = _images_in(working_dir)
                image_paths = [str(p) for p in sorted(last.figures)] if last.figures else None

                content, history = get_response_from_llm(
                    prompt, client, model, self.system_prompt,
                    image_paths=image_paths, msg_history=history,
                    temperature=self.spec.temperature, max_tokens=self.spec.max_tokens,
                )
                if record:
                    record(f"data_agent_{i}", self.system_prompt, history, content)

                code = extract_code(content)
                if not code.strip():
                    # No code this turn -> treat as the agent's final word.
                    last.term_out = content
                    break

                result = interp.run(code)
                term = trim_long_string("\n".join(result.term_out))
                new_figs = sorted(_images_in(working_dir) - before)
                success = result.exc_type is None

                last = DataAgentResult(
                    success=success, term_out=term,
                    figures=[str(p) for p in new_figs] or last.figures, code=code,
                )
                if success:
                    prompt = (
                        f"The script ran. Output:\n```\n{term}\n```\n"
                        f"{'Figures were produced; they are attached. ' if new_figs else ''}"
                        "If the analysis is complete and correct, reply with 'DONE'. "
                        "Otherwise provide an improved script in a python code block."
                    )
                else:
                    prompt = (
                        f"The script failed:\n```\n{term}\n```\n"
                        "Fix the error and provide a corrected python code block."
                    )

                if success and "DONE" in content:
                    break
        finally:
            interp.cleanup_session()

        return last
