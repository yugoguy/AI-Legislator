"""Evaluator agent.

Scores a 議案 (G node) on five criteria, each 0–1, and reduces them to a single
Q in [0,1] used as the exploitation term of the UCB selection policy (see
research_selection.py). Stateless like legislator.py: one function, given config
+ the G's proposal and research summaries.

Q is the average of the five criteria. The model is asked to emit the five
sub-scores AND their average ("final_score"); we recompute the average from the
sub-scores when all are present (guarding against a mis-averaged final), and fall
back to the emitted final_score otherwise. Sub-scores are returned too, so the
caller can log them onto the G's q_history.

Criteria (also spelled out in the prompt): grounding, specificity,
jurisdictional_fit, feasibility, potential.
"""

from __future__ import annotations

from llm import client_for, get_response_from_llm
from response import extract_json_between_markers
from config import Config, ModelSpec
from node import GNode
import prompts

_CRITERIA = ("grounding", "specificity", "jurisdictional_fit",
             "feasibility", "potential")


def _summaries_text(g: GNode) -> str:
    # Only successful findings are evidence. Failed research items (tool/process
    # errors) are excluded so they cannot drag down the proposal's score.
    ok_items = [s for s in g.research_summaries if s.get("ok")]
    if not ok_items:
        return "(no successful research gathered yet)"
    return "\n".join(f"- {s['source']}: {s['outcome']}" for s in ok_items)


def _reduce(data: dict) -> tuple[float, dict]:
    """Return (Q, sub_scores). Average the five criteria when all present."""
    subs: dict[str, float] = {}
    for k in _CRITERIA:
        v = data.get(k)
        if isinstance(v, (int, float)):
            subs[k] = max(0.0, min(1.0, float(v)))
    if len(subs) == len(_CRITERIA):
        q = sum(subs.values()) / len(_CRITERIA)
    else:  # fall back to the emitted final_score
        fs = data.get("final_score")
        q = max(0.0, min(1.0, float(fs))) if isinstance(fs, (int, float)) else 0.0
    return q, subs


def evaluate(cfg: Config, g: GNode, *, record=None) -> tuple[float, dict]:
    """Score G; return (Q in [0,1], sub_scores dict). Q=0 on parse failure."""
    spec: ModelSpec = cfg.model_for("select", "evaluator")
    lang = cfg.language_for("select")
    system = prompts.EVALUATE_SYSTEM.format(
        region=cfg.region, region_level=cfg.region_level, language=lang)
    user = prompts.EVALUATE_USER.format(
        region=cfg.region, region_level=cfg.region_level,
        proposal=g.read_proposal() or "(no proposal drafted yet)",
        summaries=_summaries_text(g),
    )
    client, model = client_for(spec.model)
    content, history = get_response_from_llm(
        user, client, model, system,
        temperature=spec.temperature, max_tokens=spec.max_tokens,
        reasoning_effort=spec.reasoning_effort,
    )
    if record:
        record("evaluate", system, history, content)

    data = extract_json_between_markers(content) or {}
    return _reduce(data)
