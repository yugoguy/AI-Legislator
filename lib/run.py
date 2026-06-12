"""Entry point + debug runner.

Builds the Config, wires the (now-implemented) selection algorithms and the
data-source tool registry, and launches the orchestrator. The only thing that
must be supplied externally is an LLM API key (e.g. ANTHROPIC_API_KEY) plus, for
the e-Stat tool, ESTAT_APP_ID; the Diet / e-Gov / web-scrape tools need no key.

Run:
    python run.py            # normal run
    DEBUG=1 python run.py    # tiny sizes for a cheap end-to-end smoke test
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from config import Config
from orchestrator import evolve
import tools as tools_mod
import evaluator as evaluator_mod
from research_selection import make_research_select
from parliament_selection import make_parliament_select


def build_config(debug: bool) -> Config:
    cfg = Config()
    if debug:
        # Shrink every dimension so a full pass is cheap to watch end-to-end.
        cfg = replace(
            cfg,
            root_dir="./run_debug",
            num_topics=2,
            research_selections=2,
            research_iters=2,
            parliament_max=2,
            parliament_rounds=1,
            refinement_selections=1,
            refinement_iters=2,
            writeup_max=2,
            batch_size=2,
            data_agent_iters=2,
        )
    return cfg


# --- PDF renderer (the one external toolchain; swappable) -------------------

def pdf_renderer(markdown_text: str, out_path: Path) -> None:
    """markdown -> HTML -> PDF via weasyprint; raises clearly if deps absent."""
    try:
        import markdown as md_lib
        from weasyprint import HTML
    except ImportError as e:
        raise RuntimeError(
            "PDF rendering requires `markdown` and `weasyprint`. Install them or "
            "swap pdf_renderer for another markdown->PDF toolchain."
        ) from e
    html = md_lib.markdown(markdown_text, extensions=["tables"])
    HTML(string=html, base_url=str(Path(out_path).parent)).write_pdf(str(out_path))


def main() -> None:
    debug = os.getenv("DEBUG") == "1"
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("ai-legislator")

    cfg = build_config(debug)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "run_debug" if debug else "run"
    cfg = replace(cfg, root_dir=f"./{prefix}_{timestamp}")

    # Mirror all terminal logging into a run.log inside the run directory.
    Path(cfg.root_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(Path(cfg.root_dir) / "run.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    log.info("logging to %s", Path(cfg.root_dir) / "run.log")

    tools = tools_mod.build_default_tools(
        web_search_model=cfg.web_search_model,
        web_search_max_results=cfg.web_search_max_results,
        region=cfg.region,
        local_sources=cfg.local_sources,
        local_max_results=cfg.local_max_results,
    )
    log.info("tools available: %s", ", ".join(sorted(tools)) or "(none)")

    research_select = make_research_select(cfg, evaluator_mod.evaluate, seed=cfg.seed)
    parliament_select = make_parliament_select(seed=cfg.seed)

    log.info("starting run (debug=%s) -> %s", debug, cfg.root_dir)
    evolve(
        cfg,
        tools=tools,
        research_select=research_select,
        parliament_select=parliament_select,
        pdf_renderer=pdf_renderer,
    )
    log.info("done. tree + progress at: %s", cfg.root_dir)
    print(f"Done. Tree + progress at: {cfg.root_dir}")


if __name__ == "__main__":
    main()
