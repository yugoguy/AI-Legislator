"""Debug harness: fake LLM only.

Everything else uses the real code path:
- real tools from tools.build_default_tools()
- real pdf_renderer from run.py
- real orchestrator/parliament/research/data-agent/interpreter
"""

from __future__ import annotations

import itertools
import json
import shutil
import sys
import types
from dataclasses import replace
from pathlib import Path

OUT_DIR = Path("./run_debug").resolve()


def _stub_sdks() -> None:
    for n in ("anthropic", "openai", "backoff"):
        sys.modules.setdefault(n, types.ModuleType(n))

    import backoff
    backoff.on_exception = lambda *a, **k: (lambda f: f)
    backoff.expo = object()

    import openai
    import anthropic

    for e in (
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "OpenAI",
    ):
        if not hasattr(openai, e):
            setattr(openai, e, type(e, (Exception,), {}))

    for e in ("RateLimitError", "APITimeoutError", "Anthropic"):
        if not hasattr(anthropic, e):
            setattr(anthropic, e, type(e, (Exception,), {}))


_stub_sdks()

from config import Config
from tools import build_default_tools
import node
import legislator
import research
import parliament
import data_agent
import orchestrator
from data_agent import DataAgent
import prompts
from run import pdf_renderer as real_pdf_renderer


_CALLS = {"n": 0}
_decide = itertools.cycle(["update", "create"])
_research_actions = None


_SAMPLE_ARGS = {
    "SearchEStat": {"query": "高齢者 人口"},
    "SearchKokkai": {"query": "高齢化"},
    "SearchEgovLaw": {"query": "介護保険"},
    "SearchWeb": {"query": "横浜市 高齢化"},
    "SearchLocalGov": {"action": "search", "query": "高齢化"},
    "FetchWebPage": {"url": "https://www.bousai.go.jp/"},
}


def _plot_code() -> str:
    return (
        "```python\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "x = [1, 2, 3, 4]\n"
        "y = [10, 14, 13, 18]\n"
        "\n"
        "plt.figure()\n"
        "plt.plot(x, y, marker='o')\n"
        "plt.title('Debug figure')\n"
        "plt.xlabel('Step')\n"
        "plt.ylabel('Value')\n"
        "plt.tight_layout()\n"
        "plt.savefig('debug_plot.png')\n"
        "print('saved debug_plot.png')\n"
        "```"
    )


def fake_llm(
    msg,
    client,
    model,
    system_message,
    *,
    image_paths=None,
    msg_history=None,
    temperature=0.75,
    max_tokens=4096,
    reasoning_effort=None,
):
    global _research_actions

    _CALLS["n"] += 1
    s, u = system_message or "", msg or ""

    if "high-level policy topic" in s:
        content = '```json\n["高齢化対策", "防災インフラ整備", "子育て支援"]\n```'

    elif "conducting evidence research" in s:
        if "Begin your research" in u and _research_actions is not None:
            name, args = next(_research_actions)
            content = f"ACTION: {name}\nARGUMENTS:\n```json\n{args}\n```"
        elif "Begin your research" in u:
            content = 'ACTION: SearchKokkai\nARGUMENTS:\n```json\n{"query": "高齢化"}\n```'
        else:
            content = (
                "ACTION: Finalize\nARGUMENTS:\n```json\n"
                '{"summary": "関連する公的情報を確認した。"}\n```'
            )

    elif "data-analysis coding agent" in s:
        content = _plot_code()

    elif "drafting a 議案" in s or "submission-ready" in s:
        content = (
            "# 高齢者支援条例案 / Elderly Support Ordinance\n\n"
            "## 本文\n本文テキスト...\n\n"
            "## 提案理由\n理由...\n\n"
            "## 出典\n- 国会会議録\n\n"
            '```json\n{"title_en": "Elderly Support Ordinance", '
            '"title_ja": "高齢者支援条例案"}\n```'
        )

    elif "deciding what to do with a 議案" in s:
        content = (
            '```json\n{"action": "%s", "rationale": "デバッグ"}\n```'
            % next(_decide)
        )

    elif "member of parliament" in s:
        content = "この議案の財源の裏付けは何か？"

    elif "defending your 議案" in s:
        content = "既存予算の再配分で財源を確保します。"

    elif "reflecting after a parliamentary" in s:
        content = "次段階で財源データをe-Statと国会会議録で補強する。"

    elif "evaluator scoring a 議案" in s:
        content = (
            '```json\n{"grounding": 0.4, "specificity": 0.5, '
            '"jurisdictional_fit": 0.6, "feasibility": 0.5, "potential": 0.6, '
            '"final_score": 0.52}\n```'
        )

    else:
        content = "（デバッグ応答）"

    hist = (msg_history or []) + [
        {"role": "user", "content": u},
        {"role": "assistant", "content": content},
    ]
    return content, hist


def fake_client(model):
    return None, model


def _patch_llm() -> None:
    import evaluator
    for mod in (legislator, research, parliament, data_agent, evaluator):
        mod.get_response_from_llm = fake_llm
        mod.client_for = fake_client
    # SearchWeb delegates to llm.web_search (a real provider call); stub it so the
    # debug smoke test needs no API key and no network for web search.
    import websearch
    websearch.web_search = lambda query, model, **kw: (
        True,
        f"(debug) results for {query}:\n- 横浜市 高齢化 統計: https://www.city.yokohama.lg.jp/",
        [{"title": "横浜市 高齢化 統計", "url": "https://www.city.yokohama.lg.jp/"}],
    )


def _build_tools() -> dict:
    cfg = Config()
    return build_default_tools(
        web_search_model=cfg.web_search_model,
        web_search_max_results=cfg.web_search_max_results,
        region=cfg.region,
        local_sources=cfg.local_sources,
        local_max_results=cfg.local_max_results,
    )


def check_tools(tools: dict) -> None:
    print("[B] calling REAL tools live:")
    for name, tool in tools.items():
        args = _SAMPLE_ARGS.get(name, {})
        out = str(tool.use_tool(**args)).replace("\n", " ")
        print(f"    {name}({args}) -> {out[:160]}")


def run_pipeline(tools: dict) -> Path:
    global _research_actions

    seq = [
        (n, json.dumps(_SAMPLE_ARGS.get(n, {}), ensure_ascii=False))
        for n in tools
    ]
    _research_actions = itertools.cycle(seq) if seq else None

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    iters = max(3, len(tools) + 1)
    cfg = replace(
        Config(),
        root_dir=str(OUT_DIR),
        num_topics=2,
        research_selections=2,
        research_iters=iters,
        parliament_max=2,
        parliament_rounds=1,
        refinement_selections=1,
        refinement_iters=iters,
        writeup_max=2,
        batch_size=2,
        data_agent_iters=2,
    )

    from research_selection import make_research_select
    from parliament_selection import make_parliament_select
    import evaluator as evaluator_mod

    tree = orchestrator.evolve(
        cfg,
        tools=tools,
        research_select=make_research_select(cfg, evaluator_mod.evaluate, seed=0),
        parliament_select=make_parliament_select(seed=0),
        pdf_renderer=real_pdf_renderer,
    )

    topics = tree.topic_nodes()
    gs = tree.g_nodes()
    researches = tree.by_type(node.RESEARCH)
    parls = tree.by_type(node.PARLIAMENT)

    assert len(topics) == 2
    assert gs and any(g.read_proposal() for g in gs)
    assert any(r.state == node.COMPLETE for r in researches)
    assert parls and all(p.transcript and p.reflection for p in parls)
    assert (OUT_DIR / "progress.txt").exists()
    assert list(OUT_DIR.rglob("raw/*.json"))

    print(
        f"[C] topics={len(topics)} G={len(gs)} "
        f"(states={ {g.state for g in gs} }) "
        f"research={len(researches)} parliament={len(parls)} "
        f"llm_calls={_CALLS['n']}"
    )
    return OUT_DIR


def check_data_agent() -> None:
    work = OUT_DIR / "_data_agent_check"
    agent = DataAgent(
        Config().model_for("research", "coding"),
        prompts.DATA_AGENT_SYSTEM,
        timeout=60,
        max_iters=2,
    )
    res = agent.run("Make a figure from the data.", work)

    assert res.success, f"data agent failed: {res.term_out}"
    assert res.figures, f"data agent produced no figures. term_out={res.term_out}"

    print(f"[D] data agent ran real code; figures={[Path(f).name for f in res.figures]}")


def _print_tree(root: Path, max_depth: int = 3) -> None:
    print(f"\nCreated files under {root}:")
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if len(rel.parts) > max_depth:
            continue
        indent = "  " * len(rel.parts)
        print(f"  {indent}{rel.parts[-1]}{'/' if p.is_dir() else ''}")


def main() -> None:
    _patch_llm()

    print("=== AI Legislator debug: FAKE LLM ONLY, REAL TOOLS ===")

    tools = _build_tools()
    print(f"[A] tools in use: {sorted(tools)}")

    check_tools(tools)

    root = run_pipeline(tools)
    check_data_agent()
    _print_tree(root)

    print("\n=== ALL CHECKS PASSED ===")
    print(f"Browse the run at: {root}")


if __name__ == "__main__":
    main()
