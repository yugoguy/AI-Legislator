"""Research engine: the tool-choice iteration reused across stages.

A single fixed-iteration loop in which the legislator model is shown the
available tools (data-source tools + a data-analysis action) and picks one per
turn via an ACTION/ARGUMENTS protocol; the tool result is fed back into the next
turn. Reused unchanged (only the stage prompt differs) by brainstorm, research,
and refinement.

Pattern source: AI Scientist v2 `perform_ideation_temp_free.py` (the
ACTION/ARGUMENTS parse loop with tool results threaded into reflection turns).

Interfaces this depends on (DEFERRED modules slot in here)
----------------------------------------------------------
- tools: dict[str, BaseTool]  -- the Japanese data-source tools. Each conforms to
  Sakana's base_tool.BaseTool (name, description, use_tool(**kwargs) -> str).
- data_agent: a DataAgent instance, exposed to the model as the "AnalyzeData"
  action (writes/reads files under the research node's data dir).

Completion semantics (settled with the user)
---------------------------------------------
- The research node is left COMPLETE iff the model reached "Finalize", or the
  iterations were exhausted with the last action succeeding.
- Otherwise it is left INCOMPLETE (a failure left it hanging); the orchestrator
  may resume it later when the parent G is reselected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from llm import client_for, get_response_from_llm
from response import extract_json_between_markers, trim_long_string
from node import ResearchNode, COMPLETE, INCOMPLETE
from config import ModelSpec
from data_agent import DataAgent

FINALIZE = "Finalize"
ANALYZE = "AnalyzeData"


@dataclass
class ResearchResult:
    status: str                      # COMPLETE | INCOMPLETE
    summary: str = ""                # short: what was looked for + how it turned out
    materials: list[dict] = field(default_factory=list)  # gathered findings
    figures: list[str] = field(default_factory=list)


def _tool_catalog(tools: dict) -> str:
    lines = [f"- {ANALYZE}: run a Python data-analysis/visualization task "
             f"(arguments: {{\"request\": \"...\"}}).",
             f"- {FINALIZE}: finish this research "
             f"(arguments: {{\"summary\": \"one or two sentences\"}})."]
    for name, tool in tools.items():
        lines.append(f"- {name}: {tool.description}")
    return "\n".join(lines)


def run_research(
    node: ResearchNode,
    *,
    context: str,
    stage_prompt: str,
    language: str,
    spec: ModelSpec,
    tools: dict,
    data_agent: DataAgent,
    iters: int,
) -> ResearchResult:
    """Drive the ACTION/ARGUMENTS loop against `node`, recording into it."""
    node.bump_stat("attempts")
    client, model = client_for(spec.model)

    catalog = _tool_catalog(tools)
    system = (
        "You are an AI legislator conducting evidence research. Each turn, choose "
        "exactly one tool and respond ONLY as:\n"
        "ACTION: <tool name>\nARGUMENTS: <a JSON object>\n\n"
        f"Available tools:\n{catalog}\n\n"
        f"Respond in {language}."
    )
    prompt = f"{stage_prompt}\n\nContext:\n{context}\n\nBegin your research."

    history: list[dict] = []
    materials: list[dict] = []
    figures: list[str] = []
    last_ok = True
    finalized = False
    summary = ""

    for i in range(iters):
        content, history = get_response_from_llm(
            prompt, client, model, system, msg_history=history,
            temperature=spec.temperature, max_tokens=spec.max_tokens,
        )
        node.record_raw(f"research_{i}", system, history, content)

        action, args = _parse_action(content)
        node.action_log.append({"iter": i, "action": action, "arguments": args})

        if action == FINALIZE:
            summary = (args or {}).get("summary", "")
            finalized = True
            last_ok = True
            break

        if action == ANALYZE:
            request = (args or {}).get("request", "")
            res = data_agent.run(request, node.data_dir,
                                 record=node.record_raw)
            last_ok = res.success
            figures.extend(res.figures)
            materials.append({"action": ANALYZE, "request": request,
                              "ok": res.success, "output": res.term_out})
            node.source = ANALYZE
            node.query_input = request
            feedback = trim_long_string(res.term_out)
            prompt = (f"AnalyzeData {'succeeded' if res.success else 'failed'}. "
                      f"Output:\n```\n{feedback}\n```\nContinue, or Finalize.")
        elif action in tools:
            try:
                out = tools[action].use_tool(**(args or {}))
                last_ok = True
            except Exception as e:                       # tool failure
                out = f"Error using {action}: {e}"
                last_ok = False
            materials.append({"action": action, "arguments": args,
                              "ok": last_ok, "output": out})
            node.source = action
            node.query_input = str(args)
            prompt = (f"{action} returned:\n```\n{trim_long_string(str(out))}\n```\n"
                      "Continue, or Finalize.")
        else:                                            # unparseable / unknown
            last_ok = False
            prompt = ("Unrecognized ACTION. Reply strictly with ACTION/ARGUMENTS "
                      f"using one of the listed tools (in {language}).")

    status = COMPLETE if (finalized or last_ok) else INCOMPLETE
    node.outputs = trim_long_string(
        "\n\n".join(str(m.get("output", "")) for m in materials)
    )
    node.error = None if status == COMPLETE else "left incomplete"
    node.state = status

    if not summary:
        summary = _fallback_summary(materials)
    return ResearchResult(status=status, summary=summary,
                          materials=materials, figures=figures)


def _parse_action(text: str) -> tuple[str | None, dict | None]:
    """Extract ACTION name and ARGUMENTS JSON from a model response."""
    action = None
    for line in text.splitlines():
        s = line.strip()
        if s.upper().startswith("ACTION:"):
            action = s.split(":", 1)[1].strip()
            break
    args = extract_json_between_markers(text)
    if args is None and "ARGUMENTS:" in text:
        args = extract_json_between_markers(text.split("ARGUMENTS:", 1)[1])
    return action, args


def _fallback_summary(materials: list[dict]) -> str:
    if not materials:
        return "No materials were gathered."
    ok = sum(1 for m in materials if m.get("ok"))
    last = materials[-1]
    return (f"{len(materials)} action(s), {ok} succeeded; "
            f"last: {last.get('action')} -> {'ok' if last.get('ok') else 'failed'}.")
