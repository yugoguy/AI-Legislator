"""Orchestrator: the single-pass AI Legislator pipeline.

The only assembly tier besides run.py. Owns stage sequencing, batch parallelism,
tree expansion, and the progress file. Selection algorithms and the data-source
tool registry are INJECTED (they are the deferred modules), so this file depends
on their interfaces only:

  - research_select(tree) -> g_node_id | None
        pick the next active G to work on (orchestrator skips ids already chosen
        in the current batch to guarantee no overlap). May be a stateful object
        (the UCB policy) that scores/evaluates G nodes internally and persists Q
        onto them via the tree; the orchestrator treats it as this callable only.
  - parliament_select(tree) -> list[g_node_id]
        pick the portfolio for parliament; the orchestrator closes the rest.
  - tools: dict[str, BaseTool]  -- the Japanese data-source tools.
  - pdf_renderer(markdown, out_path) -> None  -- markdown->PDF toolchain.

Single pass (no outer loop): brainstorm -> research/G loop -> parliament ->
refinement -> writeup. Each stage uses a fixed, config-driven number of node
selections. A batch runs `cfg.batch_size` work units on distinct nodes in
threads; tree mutations happen only after the batch joins.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from config import Config
from tree import Tree
from node import (GNode, ResearchNode, TopicNode, TOPIC, G, RESEARCH, PARLIAMENT,
                  ACTIVE, COMPLETE, INCOMPLETE)
import legislator
import research as research_mod
import parliament as parliament_mod
from data_agent import DataAgent
import prompts

log = logging.getLogger("ai-legislator")


def evolve(
    cfg: Config,
    *,
    tools: dict,
    research_select: Callable[[Tree], str | None],
    parliament_select: Callable[[Tree], list[str]],
    pdf_renderer: Callable[[str, Path], None],
) -> Tree:
    tree = Tree(cfg.root_dir)

    _brainstorm(cfg, tree, tools)
    _research_loop(cfg, tree, tools, research_select,
                   stage="research", n=cfg.research_selections,
                   iters=cfg.research_iters,
                   stage_prompt=prompts.GROUNDING_RESEARCH)
    _parliament(cfg, tree, parliament_select, pdf_renderer)
    _research_loop(cfg, tree, tools, research_select,
                   stage="refinement", n=cfg.refinement_selections,
                   iters=cfg.refinement_iters,
                   stage_prompt=prompts.REFINEMENT_RESEARCH)
    _writeup(cfg, tree)

    tree.update_progress()
    return tree


# --- stage: brainstorm ------------------------------------------------------

def _brainstorm(cfg: Config, tree: Tree, tools: dict) -> None:
    topics = legislator.generate_topics(cfg)
    for text in topics:
        topic = tree.new_node(TOPIC, "brainstorm", topic_text=text,
                              region=cfg.region, region_level=cfg.region_level)
        # Spawn up to g_per_topic candidate 議案 for this topic, each from its
        # own grounding research pass.
        for _ in range(cfg.g_per_topic):
            res: ResearchNode = tree.new_node(RESEARCH, "brainstorm",
                                              parent_id=topic.node_id)
            result = research_mod.run_research(
                res, context=f"Topic: {text}", topic=text,
                stage_prompt=prompts.BRAINSTORM_RESEARCH,
                language=cfg.language_for("brainstorm"),
                spec=cfg.model_for("brainstorm", "legislator"),
                region=cfg.region, region_level=cfg.region_level,
                tools=tools,
                data_agent=_data_agent(cfg, "research"),
                iters=cfg.brainstorm_iters,
            )
            tree.save(res)
            if result.status != COMPLETE:
                log.info("  [brainstorm] topic %s: research_fail (no 議案 spawned)",
                         topic.node_id)
                continue
            g: GNode = tree.new_node(G, "brainstorm", parent_id=topic.node_id,
                                     origin_topic=text)
            legislator.author_proposal(cfg, "brainstorm",
                                       _materials_str(result), g, record=g.record_raw)
            g.append_research_summary(res.node_id, res.source, res.query_input,
                                      result.summary, ok=True)
            g.bump_stat("research_count")
            tree.save(g)
            log.info("  [brainstorm] topic %s -> %s  %s",
                     topic.node_id, g.node_id, g.title_ja or "(untitled)")
            topic.bump_stat("g_spawned")
            tree.save(topic)
    tree.update_progress()


# --- stages: research / refinement loop -------------------------------------

def _research_loop(cfg: Config, tree: Tree, tools: dict,
                   research_select: Callable[[Tree], str | None], *,
                   stage: str, n: int, iters: int, stage_prompt: str) -> None:
    done = 0
    while done < n:
        batch = _pick_batch(tree, research_select, cfg.batch_size, n - done)
        if not batch:
            break

        # Pre-create the research node for each picked G (distinct dirs -> no
        # overlap), then run the batch in parallel.
        units = []
        for g_id in batch:
            g: GNode = tree.get(g_id)
            res = _resumable_research(tree, g, stage)
            units.append((g, res))

        def work(unit):
            g, res = unit
            return research_mod.run_research(
                res, context=_g_context(cfg, g), topic=g.origin_topic,
                stage_prompt=stage_prompt,
                language=cfg.language_for(stage),
                spec=cfg.model_for(stage, "legislator"),
                region=cfg.region, region_level=cfg.region_level,
                tools=tools, data_agent=_data_agent(cfg, stage), iters=iters,
            )

        with ThreadPoolExecutor(max_workers=cfg.batch_size) as ex:
            results = list(ex.map(work, units))

        # Post-batch: persist and apply legislator decisions (single-threaded).
        for (g, res), result in zip(units, results):
            tree.save(res)
            g.bump_stat("times_selected")
            g.bump_stat("research_count")
            if result.status == COMPLETE:
                action = _apply_decision(cfg, tree, g, res, result, stage)
            else:
                # Research left the node INCOMPLETE: record the failure so it is
                # visible and counts as a (failed) attempt; no proposal change.
                action = "research_fail"
                g.record_decision(stage, action, rationale=res.error or "incomplete")
            tree.save(g)
            log.info("  [%s] %s -> %s  | %s", stage, g.node_id, action,
                     g.title_ja or "(untitled)")
            done += 1
        tree.update_progress()


def _apply_decision(cfg, tree, g: GNode, res: ResearchNode, result, stage) -> str:
    """Apply the legislator's post-research action; record it on g; return it."""
    g.append_research_summary(res.node_id, res.source, res.query_input,
                              result.summary, ok=True)
    proposal = g.read_proposal() or ""
    # Withhold 'create' from the decision once the active-G cap is reached, and
    # hard-coerce any stray create to update so branching can never exceed it.
    allow_create = len(tree.g_nodes(active_only=True)) < cfg.max_g_nodes
    decision = legislator.decide_action(cfg, stage, proposal, result.summary,
                                        allow_create=allow_create,
                                        record=g.record_raw)
    action = decision.get("action", "update")
    if action == "create" and not allow_create:
        action = "update"
    rationale = decision.get("rationale", "")
    g.record_decision(stage, action, rationale)
    if action == "close":
        tree.close(g.node_id)
    elif action == "create":
        child: GNode = tree.new_node(G, stage, parent_id=res.node_id,
                                     origin_topic=g.origin_topic)
        legislator.author_proposal(cfg, stage, _materials_str(result), child,
                                   record=child.record_raw)
        child.append_research_summary(res.node_id, res.source, res.query_input,
                                      result.summary, ok=True)
        child.record_decision(stage, "create", f"branched from {g.node_id}")
        tree.save(child)
        log.info("    [%s] %s branched -> %s  %s", stage, g.node_id,
                 child.node_id, child.title_ja or "(untitled)")
    else:  # update in place
        legislator.rewrite_proposal(cfg, stage, proposal, result.summary, g,
                                    record=g.record_raw)
    return action


# --- stage: parliament ------------------------------------------------------

def _parliament(cfg: Config, tree: Tree, parliament_select, pdf_renderer) -> None:
    selected = parliament_select(tree)[: cfg.parliament_max]
    selected_set = set(selected)
    # Close every active G not selected for parliament.
    for g in tree.g_nodes(active_only=True):
        if g.node_id not in selected_set:
            tree.close(g.node_id)

    for g_id in selected:
        g: GNode = tree.get(g_id)
        parl = tree.new_node(PARLIAMENT, "parliament", parent_id=g.node_id, g_id=g_id)
        parliament_mod.run_parliament(cfg, g, parl, pdf_renderer=pdf_renderer)
        tree.save(parl)
        g.set_stat("went_to_parliament", True)
        tree.save(g)
    tree.update_progress()


# --- stage: writeup ---------------------------------------------------------

def _writeup(cfg: Config, tree: Tree) -> None:
    actives = [g for g in tree.g_nodes(active_only=True)][: cfg.writeup_max]
    for g in actives:
        parls = tree.parliament_nodes_of(g.node_id)
        reflections = "\n\n".join(p.reflection for p in parls)
        context = (f"PROPOSAL:\n{g.read_proposal() or ''}\n\n"
                   f"PARLIAMENT REFLECTIONS:\n{reflections}")
        legislator.write_up(cfg, context, g, record=g.record_raw)
        tree.save(g)
    tree.update_progress()


# --- helpers ----------------------------------------------------------------

def _data_agent(cfg: Config, stage: str) -> DataAgent:
    system = prompts.DATA_AGENT_SYSTEM.format(
        region=cfg.region, region_level=cfg.region_level)
    return DataAgent(cfg.model_for(stage, "coding"), system,
                     timeout=cfg.exec_timeout, max_iters=cfg.data_agent_iters)


def _pick_batch(tree: Tree, select, batch_size: int, remaining: int) -> list[str]:
    """Pick up to min(batch_size, remaining) DISTINCT G ids for this batch."""
    want = min(batch_size, remaining)
    picked: list[str] = []
    seen: set[str] = set()
    attempts = 0
    while len(picked) < want and attempts < want * 4:
        attempts += 1
        gid = select(tree)
        if gid is None:
            break
        if gid in seen:
            continue
        seen.add(gid)
        picked.append(gid)
    return picked


def _resumable_research(tree: Tree, g: GNode, stage: str) -> ResearchNode:
    """Resume an INCOMPLETE research node on g, else create a new one."""
    for r in tree.research_nodes_of(g.node_id):
        if r.state == INCOMPLETE:
            return r
    return tree.new_node(RESEARCH, stage, parent_id=g.node_id)


def _g_context(cfg: Config, g: GNode) -> str:
    return (f"Jurisdiction: {cfg.region} (level: {cfg.region_level}).\n"
            f"Originating topic: {g.origin_topic or '(unknown)'}\n\n"
            f"Proposal so far:\n{g.read_proposal() or '(none yet)'}\n\n"
            f"Prior research:\n" +
            "\n".join(f"- {s['source']}: {s['outcome']}" for s in g.research_summaries))


def _materials_str(result) -> str:
    return "\n\n".join(
        f"[{m.get('action')}] {'ok' if m.get('ok') else 'failed'}\n{m.get('output','')}"
        for m in result.materials
    ) or result.summary
