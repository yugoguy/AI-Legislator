"""Offline debug harness — exercise the whole system WITHOUT real LLMs.

Run:
    python debug_run.py                 # fully offline: fake LLM + fake tools
    REAL_TOOLS=1 python debug_run.py    # fake LLM, but REAL data-source tools
                                        #   (needs network; e-Stat also needs
                                        #    ESTAT_APP_ID, others need no key)

The LLM is ALWAYS faked (we are debugging plumbing + tool integration, not the
model). Output is written to ./run_debug/ so you can open the resulting node
tree, proposals, raw conversations, and tool outputs. With REAL_TOOLS=1 the
agent's tool calls hit the live APIs and real responses are saved into the
research nodes.

API keys: only e-Stat needs one (ESTAT_APP_ID). 国会会議録 / e-Gov / web-scrape
need none. Missing keys just disable that one tool (with a warning).
"""

from __future__ import annotations

import itertools
import os
import shutil
import sys
import tempfile
import types
from dataclasses import replace
from pathlib import Path

REAL_TOOLS = os.getenv("REAL_TOOLS") == "1"
OUT_DIR = Path("./run_debug").resolve()


# --- A. stub provider SDKs so the system imports with no installs/keys --------
def _stub_sdks() -> None:
    for n in ("anthropic", "openai", "backoff"):
        sys.modules.setdefault(n, types.ModuleType(n))
    import backoff
    backoff.on_exception = lambda *a, **k: (lambda f: f)
    backoff.expo = object()
    import openai, anthropic
    for e in ("RateLimitError", "APITimeoutError", "APIConnectionError",
              "InternalServerError", "OpenAI"):
        if not hasattr(openai, e):
            setattr(openai, e, type(e, (Exception,), {}))
    for e in ("RateLimitError", "APITimeoutError", "Anthropic"):
        if not hasattr(anthropic, e):
            setattr(anthropic, e, type(e, (Exception,), {}))
    pil = types.ModuleType("PIL"); img = types.ModuleType("PIL.Image"); pil.Image = img
    sys.modules.setdefault("PIL", pil); sys.modules.setdefault("PIL.Image", img)


_stub_sdks()

from config import Config                       # noqa: E402
from tools import BaseTool                       # noqa: E402
import node                                      # noqa: E402
import legislator, research, parliament, data_agent, orchestrator  # noqa: E402
from data_agent import DataAgent                 # noqa: E402
import prompts                                   # noqa: E402


# --- scripted fake model -----------------------------------------------------
_CALLS = {"n": 0}
_decide = itertools.cycle(["update", "create"])
# In REAL_TOOLS mode, cycle the agent through every available tool before
# finalizing, so each real API is actually hit and its output saved.
_research_actions: "itertools.cycle | None" = None


def fake_llm(msg, client, model, system_message, *, image_paths=None,
             msg_history=None, temperature=0.75, max_tokens=4096):
    _CALLS["n"] += 1
    s, u = system_message or "", msg or ""

    if "high-level policy topic" in s:
        content = '```json\n["高齢化対策", "防災インフラ整備", "子育て支援"]\n```'
    elif "conducting evidence research" in s:
        if "Begin your research" in u and _research_actions is not None:
            name, args = next(_research_actions)
            content = f'ACTION: {name}\nARGUMENTS:\n```json\n{args}\n```'
        elif "Begin your research" in u:
            content = 'ACTION: SearchKokkai\nARGUMENTS:\n```json\n{"query": "高齢化"}\n```'
        else:
            content = ('ACTION: Finalize\nARGUMENTS:\n```json\n'
                       '{"summary": "関連する公的情報を確認した。"}\n```')
    elif "data-analysis coding agent" in s:
        if "reply with 'DONE'" in u or "Fix the error" in u:
            content = "解析は完了しました。これ以上のコードは不要です。"
        else:
            content = ("```python\n"
                       "with open('out.png', 'wb') as f:\n"
                       "    f.write(b'\\x89PNG\\r\\n')\n"
                       "print('wrote figure')\n"
                       "```")
    elif "drafting a 議案" in s or "submission-ready" in s:
        content = ("# 高齢者支援条例案 / Elderly Support Ordinance\n\n"
                   "## 本文\n本文テキスト...\n\n## 提案理由\n理由...\n\n"
                   "## 出典\n- 国会会議録\n\n"
                   '```json\n{"title_en": "Elderly Support Ordinance", '
                   '"title_ja": "高齢者支援条例案"}\n```')
    elif "deciding what to do with a 議案" in s:
        content = ('```json\n{"action": "%s", "rationale": "デバッグ"}\n```'
                   % next(_decide))
    elif "member of parliament" in s:
        content = "この議案の財源の裏付けは何か？"
    elif "defending your 議案" in s:
        content = "既存予算の再配分で財源を確保します。"
    elif "reflecting after a parliamentary" in s:
        content = "次段階で財源データをe-Statと国会会議録で補強する。"
    else:
        content = "（デバッグ応答）"

    hist = (msg_history or []) + [{"role": "user", "content": u},
                                  {"role": "assistant", "content": content}]
    return content, hist


def fake_client(model):
    return None, model


def _patch_llm() -> None:
    for mod in (legislator, research, parliament, data_agent):
        mod.get_response_from_llm = fake_llm
        mod.client_for = fake_client


# --- fake tools + noop pdf ----------------------------------------------------
class FakeTool(BaseTool):
    def __init__(self, name, desc):
        super().__init__(name, desc,
                         [{"name": "query", "type": "str", "description": "q"}])

    def use_tool(self, **kwargs):
        return f"[fake {self.name}] result for {kwargs}"


def fake_pdf(markdown_text, out_path):
    Path(out_path).write_bytes(b"%PDF-1.4 debug\n")


# Sample arguments to drive each real tool by name.
_SAMPLE_ARGS = {
    "SearchEStat": {"query": "高齢者 人口"},
    "SearchKokkai": {"query": "高齢化"},
    "SearchEgovLaw": {"query": "介護保険"},
    "FetchWebPage": {"url": "https://www.bousai.go.jp/"},
}


def _build_tools() -> dict:
    if not REAL_TOOLS:
        return {t.name: t for t in (FakeTool("SearchKokkai", "Diet"),
                                    FakeTool("SearchEgovLaw", "laws"),
                                    FakeTool("FetchWebPage", "web"))}
    from tools import build_default_tools
    return build_default_tools()


# --- B. tool check -----------------------------------------------------------
def check_tools(tools: dict) -> None:
    if REAL_TOOLS:
        print("[B] calling REAL tools live:")
        for name, tool in tools.items():
            args = _SAMPLE_ARGS.get(name, {})
            out = str(tool.use_tool(**args)).replace("\n", " ")
            print(f"    {name}({args}) -> {out[:160]}")
        return
    # offline: mock requests and assert the real tools build correct URLs
    try:
        import requests
    except ImportError:
        print("[B] requests not installed; skipping URL check.")
        return
    calls = []

    def fake_get(url, **kw):
        calls.append(url)

        class R:
            def raise_for_status(self): pass
            def json(self): return {}
            text = ""; encoding = "utf-8"; apparent_encoding = "utf-8"
        return R()

    orig = requests.get
    requests.get = fake_get
    try:
        os.environ.setdefault("ESTAT_APP_ID", "X")
        import estat, kokkai, egov, webscrape
        estat.EstatTool().use_tool(query="高齢者")
        kokkai.KokkaiTool().use_tool(query="少子化")
        egov.EgovLawTool().use_tool(query="介護保険")
        webscrape.WebScrapeTool().use_tool(url="https://example.lg.jp/x")
    finally:
        requests.get = orig
    assert any("e-stat.go.jp" in u for u in calls)
    assert any("kokkai.ndl.go.jp" in u for u in calls)
    assert any("laws.e-gov.go.jp" in u for u in calls)
    print(f"[B] (offline) real tools built {len(calls)} correct request URLs.")


# --- C. full pipeline --------------------------------------------------------
def run_pipeline(tools: dict) -> Path:
    global _research_actions
    if REAL_TOOLS:
        seq = [(n, __import__("json").dumps(_SAMPLE_ARGS.get(n, {}), ensure_ascii=False))
               for n in tools]
        _research_actions = itertools.cycle(seq) if seq else None

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    iters = max(3, len(tools) + 1) if REAL_TOOLS else 2
    cfg = replace(Config(), root_dir=str(OUT_DIR), num_topics=2,
                  research_selections=2, research_iters=iters,
                  parliament_max=2, parliament_rounds=1,
                  refinement_selections=1, refinement_iters=iters,
                  writeup_max=2, batch_size=2, data_agent_iters=2)

    from research_selection import make_research_select
    from parliament_selection import make_parliament_select

    tree = orchestrator.evolve(
        cfg, tools=tools,
        research_select=make_research_select(seed=0),
        parliament_select=make_parliament_select(seed=0),
        pdf_renderer=fake_pdf,
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

    print(f"[C] topics={len(topics)} G={len(gs)} (states={ {g.state for g in gs} }) "
          f"research={len(researches)} parliament={len(parls)} "
          f"llm_calls={_CALLS['n']}")
    return OUT_DIR


# --- D. real interpreter via DataAgent ---------------------------------------
def check_data_agent() -> None:
    work = OUT_DIR / "_data_agent_check"
    agent = DataAgent(Config().model_for("research", "coding"),
                      prompts.DATA_AGENT_SYSTEM, timeout=60, max_iters=2)
    res = agent.run("Make a figure from the data.", work)
    assert res.success and res.figures, f"data agent failed: {res.term_out}"
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
    print(f"=== AI Legislator offline debug (REAL_TOOLS={REAL_TOOLS}) ===")
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
