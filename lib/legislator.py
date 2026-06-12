"""Legislator agent.

The decision-making LLM agent. Stateless functions (each takes config + the
node/material context it needs) so the orchestrator stays the only assembly
point. All wording lives in prompts.py.

Responsibilities:
  - generate_topics: high-level topic strings for the brainstorm stage.
  - author_proposal: build a new G's proposal.md from research materials.
  - decide_action: after a completed research step, choose update/create/close.
  - rewrite_proposal: in-place revision of a G's proposal.md given a summary.
  - answer: defend a G against a parliament question.
  - reflect: post-parliament reflection that feeds refinement.
  - write_up: final submission-ready proposal.
"""

from __future__ import annotations

from llm import client_for, get_response_from_llm
from response import extract_json_between_markers, extract_text_up_to_code
from config import Config
from node import GNode
import prompts


def _call(spec, system, user, *, record=None, name="legislator", image_paths=None):
    client, model = client_for(spec.model)
    content, history = get_response_from_llm(
        user, client, model, system, image_paths=image_paths,
        temperature=spec.temperature, max_tokens=spec.max_tokens,
    )
    if record:
        record(name, system, history, content)
    return content


def generate_topics(cfg: Config) -> list[str]:
    spec = cfg.model_for("brainstorm", "legislator")
    lang = cfg.language_for("brainstorm")
    system = prompts.TOPIC_SYSTEM.format(region=cfg.region,
                                         region_level=cfg.region_level, language=lang)
    user = prompts.TOPIC_USER.format(n=cfg.num_topics, region=cfg.region,
                                     region_level=cfg.region_level, language=lang)
    content = _call(spec, system, user, name="generate_topics")
    data = extract_json_between_markers(content)
    if isinstance(data, list):
        return [str(x) for x in data][: cfg.num_topics]
    if isinstance(data, dict):  # tolerate {"topics": [...]}
        for v in data.values():
            if isinstance(v, list):
                return [str(x) for x in v][: cfg.num_topics]
    return []


def author_proposal(cfg: Config, stage: str, materials: str, g: GNode, *,
                    record=None) -> None:
    """Write g.proposal.md and set its EN/JA titles from research materials."""
    spec = cfg.model_for(stage, "legislator")
    lang = cfg.language_for(stage)
    system = prompts.BUILD_PROPOSAL_SYSTEM.format(
        region=cfg.region, region_level=cfg.region_level, language=lang)
    user = prompts.BUILD_PROPOSAL_USER.format(materials=materials, region=cfg.region)
    content = _call(spec, system, user, record=record, name="author_proposal")
    _apply_proposal(content, g)


def decide_action(cfg: Config, stage: str, proposal: str, summary: str, *,
                  record=None) -> dict:
    spec = cfg.model_for(stage, "legislator")
    lang = cfg.language_for(stage)
    system = prompts.UPDATE_DECISION_SYSTEM.format(region=cfg.region, language=lang)
    user = prompts.UPDATE_DECISION_USER.format(
        proposal=proposal, summary=summary, region=cfg.region)
    content = _call(spec, system, user, record=record, name="decide_action")
    decision = extract_json_between_markers(content) or {}
    if decision.get("action") not in ("update", "create", "close"):
        decision["action"] = "update"
    return decision


def rewrite_proposal(cfg: Config, stage: str, proposal: str, summary: str, g: GNode, *,
                     record=None) -> None:
    spec = cfg.model_for(stage, "legislator")
    lang = cfg.language_for(stage)
    system = prompts.BUILD_PROPOSAL_SYSTEM.format(
        region=cfg.region, region_level=cfg.region_level, language=lang)
    user = prompts.REWRITE_PROPOSAL_USER.format(
        proposal=proposal, summary=summary, region=cfg.region)
    content = _call(spec, system, user, record=record, name="rewrite_proposal")
    _apply_proposal(content, g)


def answer(cfg: Config, proposal: str, materials: str, question: str, *,
           record=None) -> str:
    spec = cfg.model_for("parliament", "legislator")
    lang = cfg.language_for("parliament")
    system = prompts.ANSWER_SYSTEM.format(region=cfg.region, language=lang)
    user = prompts.ANSWER_USER.format(proposal=proposal, materials=materials,
                                      question=question)
    return _call(spec, system, user, record=record, name="answer")


def reflect(cfg: Config, transcript: str, *, record=None) -> str:
    spec = cfg.model_for("parliament", "legislator")
    lang = cfg.language_for("parliament")
    system = prompts.REFLECT_SYSTEM.format(region=cfg.region, language=lang)
    user = prompts.REFLECT_USER.format(transcript=transcript)
    return _call(spec, system, user, record=record, name="reflect")


def write_up(cfg: Config, context: str, g: GNode, *, record=None) -> None:
    spec = cfg.model_for("writeup", "writeup")
    lang = cfg.language_for("writeup")
    system = prompts.WRITEUP_SYSTEM.format(
        region=cfg.region, region_level=cfg.region_level, language=lang)
    user = prompts.WRITEUP_USER.format(context=context, region=cfg.region)
    content = _call(spec, system, user, record=record, name="write_up")
    _apply_proposal(content, g)


def _apply_proposal(content: str, g: GNode) -> None:
    """Split model output into proposal Markdown + a trailing title JSON block."""
    titles = extract_json_between_markers(content) or {}
    md = extract_text_up_to_code(content) or content
    g.write_proposal(md.strip())
    # set_titles logs the change into g.title_history (incl. the first title).
    g.set_titles(titles.get("title_en"), titles.get("title_ja"))
