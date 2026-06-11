"""Parliament (質疑応答) stage.

For a selected G node: render its proposal to PDF, then run a fixed number of
Q&A rounds. The parliament model asks a sharp question each round while seeing
the proposal (the PDF rasterized to page images, so vision is real, not the raw
Markdown); the legislator answers from the G's gathered materials. The transcript
is stored on a parliament node, and a legislator reflection is generated to feed
the refinement stage.

One model is prompted for the questioner side; the legislator module supplies the
answers. Reflection is the legislator's, per the settled design (no separate
summary doc).
"""

from __future__ import annotations

from pathlib import Path

import legislator
from llm import client_for, get_response_from_llm
from config import Config
from node import GNode, ParliamentNode
import prompts


def _pdf_to_images(pdf_path: Path, out_dir: Path) -> list[str]:
    """Rasterize PDF pages to PNGs for vision input.

    Uses PyMuPDF (fitz) if available; returns [] on absence so the caller can
    fall back to text. PDF rendering is the one external toolchain choice.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    doc = fitz.open(pdf_path)
    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=150)
        p = out_dir / f"page_{i:02d}.png"
        pix.save(p)
        paths.append(str(p))
    doc.close()
    return paths


def run_parliament(cfg: Config, g: GNode, parl: ParliamentNode, *,
                   pdf_renderer) -> None:
    """Run the Q&A loop for `g`, recording into `parl`."""
    g.render_pdf(pdf_renderer)
    page_images = _pdf_to_images(g.proposal_pdf_path, parl.dir / "pdf_pages")

    spec = cfg.model_for("parliament", "parliament")
    lang = cfg.language_for("parliament")
    q_system = prompts.QUESTION_SYSTEM.format(language=lang)
    client, model = client_for(spec.model)

    proposal = g.read_proposal() or ""
    materials = _materials_text(g)
    history: list[dict] = []

    for r in range(cfg.parliament_rounds):
        q_user = prompts.QUESTION_USER.format(
            title=(g.title_ja or g.title_en or g.node_id), round=r + 1)
        question, history = get_response_from_llm(
            q_user, client, model, q_system,
            image_paths=page_images or None, msg_history=history,
            temperature=spec.temperature, max_tokens=spec.max_tokens,
        )
        parl.record_raw(f"question_{r}", q_system, history, question)

        ans = legislator.answer(cfg, proposal, materials, question,
                                record=parl.record_raw)
        parl.transcript.append({"round": r + 1, "question": question, "answer": ans})
        # Feed the answer back so the next question builds on it.
        history = history + [{"role": "user", "content": f"(legislator answered: {ans})"}]

    parl.bump_stat("rounds", cfg.parliament_rounds)
    transcript_text = "\n\n".join(
        f"Q{t['round']}: {t['question']}\nA{t['round']}: {t['answer']}"
        for t in parl.transcript
    )
    parl.reflection = legislator.reflect(cfg, transcript_text, record=parl.record_raw)


def _materials_text(g: GNode) -> str:
    """Compact view of the G's research summaries for the answerer."""
    return "\n".join(
        f"- [{'ok' if s['ok'] else 'failed'}] {s['source']}: {s['outcome']}"
        for s in g.research_summaries
    )
