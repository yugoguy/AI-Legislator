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
ROLES = ("legislator", "parliament", "coding", "writeup")


@dataclass(frozen=True)
class ModelSpec:
    """One model selection: the model string plus its sampling knobs."""

    model: str
    temperature: float = 0.75
    max_tokens: int = 4096


def _default_models() -> dict[str, ModelSpec]:
    """Default per-(stage, role) models, keyed "stage:role".

    Model strings are placeholders chosen in config, NOT gated by an allow-list
    (see llm.py). Swap freely; each should be vision-capable.
    """
    legislator = ModelSpec("claude-3-5-sonnet-20241022")
    coding = ModelSpec("claude-3-5-sonnet-20241022")
    parliament = ModelSpec("claude-3-5-sonnet-20241022")
    writeup = ModelSpec("claude-3-5-sonnet-20241022")
    return {
        "brainstorm:legislator": legislator,
        "research:legislator": legislator,
        "research:coding": coding,
        "parliament:parliament": parliament,
        "parliament:legislator": legislator,
        "refinement:legislator": legislator,
        "refinement:coding": coding,
        "writeup:writeup": writeup,
    }


def _default_languages() -> dict[str, str]:
    """Per-stage output language. Inner stages vs final write-up set separately."""
    return {
        "brainstorm": "Japanese",
        "research": "Japanese",
        "parliament": "Japanese",
        "refinement": "Japanese",
        "writeup": "Japanese",
    }


@dataclass(frozen=True)
class Config:
    # --- Run / IO ---
    root_dir: str = "./run"          # the on-disk node tree lives here
    seed: int = 0
    verbose: bool = True

    # --- Target jurisdiction (drives topic generation) ---
    region: str = "東京都"
    region_level: str = "都道府県"   # e.g. 市区町村 / 都道府県 / 国

    # --- Models (per stage:role) and language (per stage) ---
    models: dict[str, ModelSpec] = field(default_factory=_default_models)
    languages: dict[str, str] = field(default_factory=_default_languages)

    # --- Stage sizes / iteration counts ---
    num_topics: int = 5              # initial high-level topic nodes
    research_selections: int = 20    # node selections in the research/G loop
    research_iters: int = 6          # conversation iterations per research execution
    parliament_max: int = 5          # G nodes taken to parliament (others closed)
    parliament_rounds: int = 3       # Q&A rounds per G
    refinement_selections: int = 8   # node selections in the refinement loop
    refinement_iters: int = 6        # conversation iterations per refinement research
    writeup_max: int = 5             # active G nodes written up at most

    # --- Parallelism ---
    batch_size: int = 3              # parallel work units per batch (no node overlap)

    # --- Data agent / execution ---
    exec_timeout: int = 1200         # per code execution, seconds
    data_agent_iters: int = 4        # generate->run->reflect rounds

    # --- Resource ceiling (advisory; enforced by the orchestrator) ---
    max_total_llm_calls: int = 0     # 0 = unlimited

    # --- accessors -------------------------------------------------------

    def model_for(self, stage: str, role: str) -> ModelSpec:
        """ModelSpec for a (stage, role); falls back to the legislator default."""
        return self.models.get(f"{stage}:{role}") or self.models["brainstorm:legislator"]

    def language_for(self, stage: str) -> str:
        return self.languages.get(stage, "Japanese")
