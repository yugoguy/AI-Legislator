"""Text-extraction helpers for parsing LLM output.

Sources
-------
- AI Scientist v2 `treesearch/utils/response.py` (most functions).
- AI Scientist v1 `ai_scientist/llm.py` (`extract_json_between_markers`).

Reused (logic carried over, lightly edited)
-------------------------------------------
- wrap_code, is_valid_python_script, extract_text_up_to_code, trim_long_string
- extract_code: fenced + unfenced Python block extraction, kept only if it compiles
- extract_json_between_markers: ```json fence first, then any {...} fallback,
  with a control-character-stripping retry

Not reused
----------
- extract_jsons (the any-{...} scanner): superseded by extract_json_between_markers,
  which is what the agent loops actually need (they emit fenced JSON).

Changed
-------
- Black formatting in extract_code is OPTIONAL: if `black` is not installed the
  code is returned unformatted rather than raising ImportError. (Original
  hard-imported black.)
"""

from __future__ import annotations

import json
import re

try:
    import black

    _HAS_BLACK = True
except ImportError:  # formatting is a nicety, not a requirement
    _HAS_BLACK = False


def wrap_code(code: str, lang: str = "python") -> str:
    """Wrap a code string in a triple-backtick fence."""
    return f"```{lang}\n{code}\n```"


def is_valid_python_script(script: str) -> bool:
    """True if `script` compiles as Python."""
    try:
        compile(script, "<string>", "exec")
        return True
    except SyntaxError:
        return False


def format_code(code: str) -> str:
    """Black-format `code`; return it unchanged if Black is absent or input invalid."""
    if not _HAS_BLACK:
        return code
    try:
        return black.format_str(code, mode=black.FileMode())
    except black.parsing.InvalidInput:  # type: ignore[attr-defined]
        return code


def extract_text_up_to_code(s: str) -> str:
    """Return the natural-language text before the first code fence (or "")."""
    if "```" not in s:
        return ""
    return s[: s.find("```")].strip()


def extract_code(text: str) -> str:
    """Extract and concatenate valid Python code blocks from `text`.

    Tries fenced ```python ... ``` blocks first, then falls back to treating the
    whole string as a single block. Only blocks that compile are kept.
    """
    parsed: list[str] = []

    matches = re.findall(r"```(python)?\n*(.*?)\n*```", text, re.DOTALL)
    for m in matches:
        parsed.append(m[1])

    if not parsed:  # whole text may be code, or fences may be missing
        m = re.findall(r"^(```(python)?)?\n?(.*?)\n?(```)?$", text, re.DOTALL)
        if m:
            parsed.append(m[0][2])

    valid = [format_code(c) for c in parsed if is_valid_python_script(c)]
    return format_code("\n\n".join(valid))


def trim_long_string(string: str, threshold: int = 5100, k: int = 2500) -> str:
    """Collapse the middle of an over-long string, keeping the first/last `k` chars."""
    if len(string) > threshold:
        head, tail = string[:k], string[-k:]
        cut = len(string) - 2 * k
        return f"{head}\n ... [{cut} characters truncated] ... \n{tail}"
    return string


def extract_json_between_markers(llm_output: str) -> dict | None:
    """Pull the first parseable JSON object from an LLM response.

    Prefers ```json ... ``` fences; falls back to the first {...} span. Retries
    once after stripping control characters. Returns None if nothing parses.
    """
    matches = re.findall(r"```json(.*?)```", llm_output, re.DOTALL)
    if not matches:
        matches = re.findall(r"\{.*?\}", llm_output, re.DOTALL)

    for js in matches:
        js = js.strip()
        try:
            return json.loads(js)
        except json.JSONDecodeError:
            try:
                cleaned = re.sub(r"[\x00-\x1F\x7F]", "", js)
                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue
    return None
