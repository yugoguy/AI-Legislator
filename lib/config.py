"""Configuration: every non-trivial choice is a knob here.

Mirrors the discipline of the NEAT `config.py`: low-level modules take explicit
arguments and never import this file; the orchestrator unpacks these and passes
them down. Frozen dataclass, single source of truth.

Two things are intentionally granular per the design:
  - models: chosen per (stage, role) pair, each expected to be vision-capable.
  - language: chosen per stage (gives "inner vs final" as a special case).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Stage / role vocabularies (kept as constants so callers don't pass raw strings).
STAGES = ("brainstorm", "research", "parliament", "refinement", "writeup")
ROLES = ("legislator", "parliament", "coding", "writeup", "evaluator")

# Reference list of model strings to choose from when configuring `models` below.
# NOT enforced: routing is by substring (see llm.py) and the provider API is the
# real validator, so any current model string works even if absent here. This is
# just a convenience menu. Verified current as of June 2026; it will drift.
AVAILABLE_MODELS = (
    # Anthropic — Active, all vision-capable
    "claude-opus-4-8",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
    "claude-fable-5",

    # OpenAI — GPT-5 family; USD / 1M tokens: input / cached input / output
    "gpt-5-nano",        # $0.05 / $0.005 / $0.40
    "gpt-5.4-nano",      # $0.20 / $0.02  / $1.25
    "gpt-5-mini",        # $0.25 / $0.025 / $2.00
    "gpt-5.4-mini",      # $0.75 / $0.075 / $4.50
    "gpt-5",             # $1.25 / $0.125 / $10.00
    "gpt-5.1",           # $1.25 / $0.125 / $10.00
    "gpt-5.2",           # $1.75 / $0.175 / $14.00
    "gpt-5.4",           # $2.50 / $0.25  / $15.00
    "gpt-5.5",           # $5.00 / $0.50  / $30.00

    # OpenAI — older GPT / vision-capable; USD / 1M tokens: input / cached input / output
    "gpt-4.1-nano",      # $0.10 / $0.025 / $0.40
    "gpt-4o-mini",       # $0.15 / $0.075 / $0.60
    "gpt-4.1-mini",      # $0.40 / $0.10  / $1.60
    "gpt-4.1",           # $2.00 / $0.50  / $8.00
    "gpt-4o",            # $2.50 / $1.25  / $10.00

    # OpenAI — reasoning models; USD / 1M tokens: input / cached input / output
    "o1-mini",           # $1.10 / $0.55  / $4.40
    "o3-mini",           # $1.10 / $0.55  / $4.40
    "o4-mini",           # $1.10 / $0.275 / $4.40
    "o3",                # $2.00 / $0.50  / $8.00
    "o1",                # $15.00 / $7.50 / $60.00
)


@dataclass(frozen=True)
class ModelSpec:
    """One model selection: the model string plus its sampling knobs."""

    model: str
    temperature: float = 0.8
    max_tokens: int = 4096
    # Reasoning-model effort ("minimal"|"low"|"medium"|"high"). Sent ONLY to
    # reasoning models (gpt-5*/o1/o3/o4) by llm.py; ignored for others. "low"
    # keeps hidden reasoning from eating the whole max_tokens budget on long
    # generations (the v1.3 empty-proposal bug). None => don't send the param.
    reasoning_effort: str | None = "minimal"


def _default_models() -> dict[str, ModelSpec]:
    """Default per-(stage, role) models, keyed "stage:role".

    max_tokens is raised on roles that emit a FULL proposal/script; reasoning
    tokens are drawn from this budget, so the short 4096 default truncated long
    writes to empty in v1.3. Long-write roles: legislator (author/rewrite at
    brainstorm/research/refinement) and writeup. coding (data agent) is medium.
    Short roles (parliament Q&A, evaluator, decide_action) keep the 4096 default.
    """
    LONG = 16000        # full 議案 Markdown (~9k chars observed) + reasoning headroom
    MED = 8000          # data-analysis scripts
    return {
        "brainstorm:legislator": ModelSpec("gpt-5-mini", max_tokens=LONG),
        "research:legislator": ModelSpec("gpt-5-mini", max_tokens=LONG),
        "research:coding": ModelSpec("gpt-5-mini", max_tokens=MED),
        "parliament:parliament": ModelSpec("gpt-5-mini"),
        "parliament:legislator": ModelSpec("gpt-5-mini"),  # short defense answers
        "refinement:legislator": ModelSpec("gpt-5-mini", max_tokens=LONG),
        "refinement:coding": ModelSpec("gpt-5-mini", max_tokens=MED),
        "writeup:writeup": ModelSpec("gpt-5-mini", max_tokens=LONG),
        # The Evaluator scores each 議案 for the UCB selection policy. Role is
        # stage-independent (scoring is the same wherever selection runs), so it
        # is keyed under a single pseudo-stage "select".
        "select:evaluator": ModelSpec("gpt-5-mini"),
    }


def _default_languages() -> dict[str, str]:
    """Per-stage output language. Inner stages vs final write-up set separately."""
    return {
        "brainstorm": "Japanese",
        "research": "Japanese",
        "parliament": "Japanese",
        "refinement": "Japanese",
        "writeup": "Japanese",
        # Evaluator reasons internally; its output is a parsed JSON score, so the
        # language only affects its rationale text. Japanese keeps it consistent.
        "select": "Japanese",
    }


@dataclass(frozen=True)
class Config:
    # --- Run / IO ---
    root_dir: str = "./run"          # the on-disk node tree lives here
    seed: int = 0
    verbose: bool = True

    # --- Target jurisdiction (drives topic generation) ---
    region: str = "横浜市"
    region_level: str = "市区町村"   # e.g. 市区町村 / 都道府県 / 国

    # --- Models (per stage:role) and language (per stage) ---
    models: dict[str, ModelSpec] = field(default_factory=_default_models)
    languages: dict[str, str] = field(default_factory=_default_languages)

    # --- Stage sizes / iteration counts ---
    num_topics: int = 8              # initial high-level topic nodes
    g_per_topic: int = 1             # candidate 議案 spawned per topic at brainstorm
    brainstorm_iters: int = 10        # research iterations per brainstorm grounding
    research_selections: int = 40    # node selections in the research/G loop
    research_iters: int = 8          # conversation iterations per research execution
    parliament_max: int = 6          # G nodes taken to parliament (others closed)
    parliament_rounds: int = 3       # Q&A rounds per G
    refinement_selections: int = 8   # node selections in the refinement loop
    refinement_iters: int = 10      # conversation iterations per refinement research
    writeup_max: int = 3             # active G nodes written up at most

    # Cap on simultaneously ACTIVE 議案. Once reached, the legislator's per-step
    # decision is offered only update|close (no 'create'), and the orchestrator
    # hard-coerces any stray 'create' to 'update'. Brainstorm seeding is NOT
    # gated by this (it creates the initial num_topics * g_per_topic directly);
    # the cap only limits research/refinement branching. Closing a G frees a slot.
    max_g_nodes: int = 16

    # --- Selection policy (UCB over active 議案) ---
    # Pick score = Q(g) + ucb_c * sqrt(ln N_i / n_i), normalized over active G to
    # a sampling distribution. Q in [0,1] from the Evaluator; n_i = research count
    # of g; N_i = selection rounds g has been present for. See research_selection.
    ucb_c: float = 0.5               # exploration weight (UCB1 standard is sqrt(2))
    eval_every: int = 2              # re-score a g with the Evaluator every N picks of it

    # --- Parallelism ---
    batch_size: int = 2              # parallel work units per batch (no node overlap)

    # --- Data agent / execution ---
    exec_timeout: int = 1800         # per code execution, seconds
    data_agent_iters: int = 4        # generate->run->reflect rounds

    # --- Web search (SearchWeb tool, provider-native server-side search) ---
    # Model that executes web searches, independent of the legislator model. Any
    # Anthropic model supports the native web-search tool; for an OpenAI executor
    # this must be a search-capable model (e.g. "gpt-4o-mini-search-preview").
    web_search_model: str = "gpt-4o-mini-search-preview"
    web_search_max_results: int = 10

    # --- Local-government data (SearchLocalGov bridge tool) ---
    # Per-region local datasets (assembly bills/petitions + minutes). The bridge
    # (local_data.py) reads this to decide availability and where to load from;
    # a region-specific fetcher owns the data-structure details. "path" is a
    # local data directory (downloaded dataset). Add regions as more are built.
    local_sources: dict = field(default_factory=lambda: {
        "横浜市": {"available": True, "path": "./data/yokohama_gikai_db"},
    })
    local_max_results: int = 5

    # --- Resource ceiling (advisory; enforced by the orchestrator) ---
    max_total_llm_calls: int = 0     # 0 = unlimited

    # --- accessors -------------------------------------------------------

    def model_for(self, stage: str, role: str) -> ModelSpec:
        """ModelSpec for a (stage, role); falls back to the legislator default."""
        return self.models.get(f"{stage}:{role}") or self.models["brainstorm:legislator"]

    def language_for(self, stage: str) -> str:
        return self.languages.get(stage, "Japanese")
