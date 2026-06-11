"""Node data contract: the shared structure every tree node serializes to.

Each node is a directory on disk:

    <node_dir>/
        meta.json          structured fields (this dataclass), one shape per type
        raw/NNNN_<call>.json   every LLM/VLM conversation, append-only, for later
        ... type-specific payload files (proposal.md, code, data, assets/, ...)

Design (settled with the user)
------------------------------
- One uniform meta shape via a base dataclass: id, type, state, parent/children,
  stage provenance, timestamps, a `stats` dict (defined per-type counters) and an
  open `extra` dict (free-form, schema-free additions later).
- Four node types: topic, G (議案), research, parliament.
- A G node holds the canonical mutable `proposal.md`, its rendered `proposal.pdf`
  (the ONLY artifact parliament sees), an `assets/` dir for figures, and a short
  per-attempt `research_summaries` log (what was looked for + how it turned out).
  The full research detail lives on the research nodes, not duplicated here.
- A research node carries full detail + a complete/incomplete `state`.
- A parliament node carries the Q&A transcript + the legislator's reflection.

Evolutionary/agent logic lives in the agent and orchestrator modules, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, asdict
from datetime import datetime, timezone
from pathlib import Path

# Node type tags.
TOPIC = "topic"
G = "g"            # 議案
RESEARCH = "research"
PARLIAMENT = "parliament"

# Common states.
ACTIVE = "active"
CLOSED = "closed"
COMPLETE = "complete"
INCOMPLETE = "incomplete"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Node:
    """Base node: the uniform meta + raw/payload mechanics shared by all types."""

    node_id: str
    type: str
    stage: str                                  # stage that created the node
    state: str = ACTIVE
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    stats: dict = field(default_factory=dict)   # per-type counters
    extra: dict = field(default_factory=dict)   # open catch-all

    # ---- directory binding (not serialized; set by the tree) ----
    def __post_init__(self) -> None:
        self._dir: Path | None = None

    def bind(self, node_dir: Path) -> "Node":
        self._dir = Path(node_dir)
        return self

    @property
    def dir(self) -> Path:
        assert self._dir is not None, f"node {self.node_id} not bound to a directory"
        return self._dir

    # ---- persistence ----
    def save(self) -> None:
        self.updated_at = _now()
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "raw").mkdir(exist_ok=True)
        with open(self.dir / "meta.json", "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, ensure_ascii=False, indent=2)

    # ---- raw conversation recording (for later analysis) ----
    def record_raw(self, call_name: str, system_message, messages, response) -> None:
        raw_dir = self.dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(raw_dir.glob("*.json")))
        payload = {
            "call": call_name,
            "timestamp": _now(),
            "system_message": system_message,
            "messages": messages,
            "response": response,
        }
        with open(raw_dir / f"{idx:04d}_{call_name}.json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)

    # ---- arbitrary payload files in the node directory ----
    def write_payload(self, name: str, text: str) -> Path:
        p = self.dir / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def read_payload(self, name: str) -> str | None:
        p = self.dir / name
        return p.read_text(encoding="utf-8") if p.exists() else None

    # ---- small helpers ----
    def touch(self) -> None:
        self.updated_at = _now()

    def bump_stat(self, key: str, by: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + by

    def set_stat(self, key: str, value) -> None:
        self.stats[key] = value


@dataclass
class TopicNode(Node):
    topic_text: str = ""
    region: str = ""
    region_level: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stats.setdefault("g_spawned", 0)


@dataclass
class GNode(Node):
    """議案 node. Body lives in proposal.md; parliament sees proposal.pdf only."""

    title_en: str = ""
    title_ja: str = ""
    # Short per-research-attempt log: {research_id, source, query, outcome, ok}.
    research_summaries: list[dict] = field(default_factory=list)

    PROPOSAL_MD = "proposal.md"
    PROPOSAL_PDF = "proposal.pdf"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stats.setdefault("times_selected", 0)
        self.stats.setdefault("went_to_parliament", False)

    @property
    def assets_dir(self) -> Path:
        d = self.dir / "assets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def proposal_md_path(self) -> Path:
        return self.dir / self.PROPOSAL_MD

    @property
    def proposal_pdf_path(self) -> Path:
        return self.dir / self.PROPOSAL_PDF

    def write_proposal(self, markdown: str) -> None:
        self.write_payload(self.PROPOSAL_MD, markdown)

    def read_proposal(self) -> str | None:
        return self.read_payload(self.PROPOSAL_MD)

    def append_research_summary(self, research_id: str, source: str,
                                query: str, outcome: str, ok: bool) -> None:
        self.research_summaries.append(
            {"research_id": research_id, "source": source,
             "query": query, "outcome": outcome, "ok": ok}
        )

    def render_pdf(self, renderer) -> Path:
        """Render proposal.md -> proposal.pdf via an injected `renderer(md, path)`.

        The markdown->pdf toolchain is a swappable choice held in config, not
        baked here. `renderer` is `(markdown_text, out_path) -> None`.
        """
        md = self.read_proposal() or ""
        renderer(md, self.proposal_pdf_path)
        return self.proposal_pdf_path


@dataclass
class ResearchNode(Node):
    """One research attempt on a parent G node. Full detail + complete/incomplete."""

    source: str = ""             # which data source / action was used last
    query_input: str = ""        # the query the legislator issued
    action_log: list[dict] = field(default_factory=list)  # per-iteration record
    outputs: str = ""            # collected textual outputs / findings
    error: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.state = self.state if self.state in (COMPLETE, INCOMPLETE) else INCOMPLETE
        self.stats.setdefault("attempts", 0)

    @property
    def data_dir(self) -> Path:
        d = self.dir / "data"
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass
class ParliamentNode(Node):
    """質疑応答 record for a G node, plus the legislator's next-stage reflection."""

    g_id: str = ""
    transcript: list[dict] = field(default_factory=list)  # [{round, question, answer}]
    reflection: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.stats.setdefault("rounds", 0)


# Type tag -> class, for the loader.
_CLASSES = {TOPIC: TopicNode, G: GNode, RESEARCH: ResearchNode, PARLIAMENT: ParliamentNode}


def load_node(node_dir: Path | str) -> Node:
    """Reconstruct a Node subclass from its on-disk meta.json and bind its dir."""
    node_dir = Path(node_dir)
    with open(node_dir / "meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    cls = _CLASSES[meta["type"]]
    names = {f.name for f in fields(cls)}
    obj = cls(**{k: v for k, v in meta.items() if k in names})
    return obj.bind(node_dir)
