"""On-disk node tree.

Owns the directory layout, mints node ids, and exposes traversal accessors that
the (swappable) selection algorithms read from the root. Holds no agent logic.

Layout
------
    <root>/
        tree.json        id counters + root topic ids (tree-level state)
        progress.txt     human-readable progress + tree, refreshed after batches
        nodes/<node_id>/ one directory per node (see node.py)

Concurrency note: tree mutations (new_node/save/close) are expected to be called
from the orchestrator's single-threaded post-batch phase. Work units in a batch
operate on distinct, pre-created nodes, so parallel execution never mutates the
tree concurrently.
"""

from __future__ import annotations

import json
from pathlib import Path

from node import (
    Node, TopicNode, GNode, ResearchNode, ParliamentNode, load_node,
    TOPIC, G, RESEARCH, PARLIAMENT, ACTIVE, CLOSED,
)

_PREFIX = {TOPIC: "topic", G: "g", RESEARCH: "res", PARLIAMENT: "parl"}
_NODE_CLASS = {TOPIC: TopicNode, G: GNode, RESEARCH: ResearchNode, PARLIAMENT: ParliamentNode}


class Tree:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.nodes_dir = self.root / "nodes"
        self.nodes_dir.mkdir(parents=True, exist_ok=True)
        self._counters: dict[str, int] = {}
        self._topic_ids: list[str] = []
        self._cache: dict[str, Node] = {}
        self._load_state()

    # ---- tree-level state ----
    def _state_path(self) -> Path:
        return self.root / "tree.json"

    def _load_state(self) -> None:
        if self._state_path().exists():
            with open(self._state_path(), encoding="utf-8") as fh:
                s = json.load(fh)
            self._counters = s.get("counters", {})
            self._topic_ids = s.get("topic_ids", [])

    def _save_state(self) -> None:
        with open(self._state_path(), "w", encoding="utf-8") as fh:
            json.dump({"counters": self._counters, "topic_ids": self._topic_ids},
                      fh, ensure_ascii=False, indent=2)

    def _mint_id(self, node_type: str) -> str:
        n = self._counters.get(node_type, 0) + 1
        self._counters[node_type] = n
        return f"{_PREFIX[node_type]}_{n:04d}"

    # ---- creation / access ----
    def new_node(self, node_type: str, stage: str, parent_id: str | None = None,
                 **fields) -> Node:
        node_id = self._mint_id(node_type)
        cls = _NODE_CLASS[node_type]
        node = cls(node_id=node_id, type=node_type, stage=stage,
                   parent_id=parent_id, **fields)
        node.bind(self.nodes_dir / node_id)
        node.save()

        if parent_id is not None:
            parent = self.get(parent_id)
            if node_id not in parent.children_ids:
                parent.children_ids.append(node_id)
                parent.save()
        if node_type == TOPIC:
            self._topic_ids.append(node_id)

        self._cache[node_id] = node
        self._save_state()
        return node

    def get(self, node_id: str) -> Node:
        if node_id not in self._cache:
            self._cache[node_id] = load_node(self.nodes_dir / node_id)
        return self._cache[node_id]

    def save(self, node: Node) -> None:
        node.save()
        self._cache[node.node_id] = node

    def close(self, node_id: str) -> None:
        node = self.get(node_id)
        node.state = CLOSED
        node.save()

    # ---- traversal (read-only; selection reads these from the root) ----
    def all_ids(self) -> list[str]:
        return [p.name for p in sorted(self.nodes_dir.iterdir()) if p.is_dir()]

    def all_nodes(self) -> list[Node]:
        return [self.get(i) for i in self.all_ids()]

    def by_type(self, node_type: str) -> list[Node]:
        return [n for n in self.all_nodes() if n.type == node_type]

    def children(self, node_id: str) -> list[Node]:
        return [self.get(c) for c in self.get(node_id).children_ids]

    def topic_nodes(self) -> list[TopicNode]:
        return [self.get(i) for i in self._topic_ids]  # type: ignore[return-value]

    def g_nodes(self, active_only: bool = False) -> list[GNode]:
        gs = [n for n in self.by_type(G)]
        if active_only:
            gs = [n for n in gs if n.state == ACTIVE]
        return gs  # type: ignore[return-value]

    def research_nodes_of(self, g_id: str) -> list[ResearchNode]:
        return [c for c in self.children(g_id) if c.type == RESEARCH]  # type: ignore

    def parliament_nodes_of(self, g_id: str) -> list[ParliamentNode]:
        return [c for c in self.children(g_id) if c.type == PARLIAMENT]  # type: ignore

    # ---- root progress file ----
    def update_progress(self) -> None:
        lines = [f"# AI Legislator progress", ""]
        for t in self.topic_nodes():
            lines.append(f"[topic] {t.node_id}  {t.topic_text[:60]}")
            for g in self.children(t.node_id):
                if g.type != G:
                    continue
                tag = "closed" if g.state == CLOSED else "active"
                q = g.stats.get("Q", 0.0)
                nsel = g.stats.get("research_count", 0)
                lines.append(
                    f"  [G:{tag}] {g.node_id}  Q={q:.2f} n={nsel}  "
                    f"{(g.title_ja or '(untitled)')[:50]}"
                )
                # Title evolution (only if it actually changed over time).
                titles = [h.get("title_ja", "") for h in g.title_history]
                if len(titles) > 1:
                    lines.append("      title: " + " → ".join(
                        (tt or "(untitled)")[:30] for tt in titles))
                # Decision trail: update/create/close/research_fail per step.
                if g.decision_history:
                    trail = " ".join(f"{d['stage'][:4]}:{d['action']}"
                                     for d in g.decision_history)
                    lines.append(f"      decisions: {trail}")
                for c in self.children(g.node_id):
                    lines.append(f"    [{c.type}:{c.state}] {c.node_id}")
        (self.root / "progress.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
