"""Yokohama (横浜市) local-data fetcher.

Region-specific implementation of local_data.LocalFetcher against the
yokohama_gikai_db dataset built by yokohama_gikai_scraper.py. THIS is the only
file that knows the dataset's structure; the bridge (local_data.py) stays
format-agnostic.

Dataset (a local directory, cfg.local_sources["横浜市"]["path"]):
  yokohama_gikai_bills.json            list of bills/petitions (record_id keyed)
  bill_to_session_minutes_files.json   record_id -> that bill's SESSION minute files
  yokohama_gikai_minutes.json          downloaded minute files (metadata)
  minutes/<local_text_path>            the actual minute .txt files
  pdfs/...                             attached bill PDFs (referenced, not read here)

Matching is session-level: a bill maps to its whole session's minute files (not
a bill-specific transcript). To find where a bill is actually discussed, the
fetcher substring-searches those files.

Three capabilities (LocalFetcher contract):
  search_bills(query)            keyword OR-union over title/summary/item_type
  get_bill(record_id)            full detail + attached PDFs + minutes availability
  get_minutes(record_id, hit)    two-step:
      hit == ""  -> LOCATE: list numbered occurrences of the bill in its session
                    minutes (compact previews), so the agent picks one
      hit == "N" -> READ: the full text window around occurrence N

Minute-text handling:
  - Skip the agenda/junk header: real proceedings start at the SECOND "開議"
    (the first is inside the 議事日程 listing). Slice from there; fall back to the
    whole file if there are fewer than two.
  - Search term is derived from the bill, not the agent: the number-core
    (digits + 号, normalized to half/full width) when the bill has a number,
    else the bill title. The agent may override by passing `keyword`.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

from local_data import LocalFetcher

# Tunables (kept here; the bridge passes none of these through).
_WINDOW = 1500          # chars before/after a hit returned on READ
_PREVIEW = 50           # chars of context shown per hit on LOCATE
_MAX_HITS = 12          # max occurrences listed on LOCATE
_FILE_HEAD_CAP = 1200   # chars returned if a bill has no usable search term


def _norm(s: str) -> str:
    """Normalize width/case so fullwidth digits match halfwidth, etc."""
    return unicodedata.normalize("NFKC", s or "")


class YokohamaFetcher(LocalFetcher):
    region = "横浜市"

    def __init__(self, path: str | Path):
        self.root = Path(path)
        bills = json.loads(self._read(self.root / "yokohama_gikai_bills.json"))
        mapping = json.loads(self._read(self.root / "bill_to_session_minutes_files.json"))
        self.bills = {b["record_id"]: b for b in bills}
        self.minutes_map = {m["record_id"]: m for m in mapping}

    @staticmethod
    def _read(p: Path) -> str:
        if not p.exists():
            raise FileNotFoundError(p)
        return p.read_text(encoding="utf-8")

    # -- search ------------------------------------------------------------

    def search_bills(self, query: str, max_results: int = 10) -> str:
        terms = [_norm(t) for t in query.split() if t.strip()]
        if not terms:
            return "Provide a search keyword."
        seen: set[str] = set()
        hits: list[dict] = []
        for term in terms:                      # OR-union across terms
            for b in self.bills.values():
                rid = b["record_id"]
                if rid in seen:
                    continue
                hay = _norm(f"{b.get('title','')} {b.get('summary','')} "
                            f"{b.get('item_type','')}")
                if term in hay:
                    seen.add(rid)
                    hits.append(b)
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break
        if not hits:
            return f"「{query}」に該当する議案・請願は見つかりませんでした。"
        lines = [f"{len(hits)}件:"]
        for b in hits:
            has_min = self._has_minutes(b["record_id"])
            lines.append(
                f"- [{b['record_id']}] {b.get('number') or '（番号なし）'} "
                f"{b.get('title','')} | 結果:{b.get('result') or '—'} "
                f"| {b.get('session_title','')} | 会議録:{'有' if has_min else '無'}"
            )
        return "\n".join(lines)

    # -- bill detail -------------------------------------------------------

    def get_bill(self, record_id: str) -> str:
        b = self.bills[record_id]               # KeyError -> bridge ok=False
        files = b.get("files") or []
        pdfs = "\n".join(f"  - {f.get('text') or f.get('local_path')}: {f.get('url')}"
                         for f in files if f.get("kind") == "pdf") or "  （なし）"
        mins = self._minute_files(record_id)
        min_list = "\n".join(
            f"  - {m.get('meeting_date_text','?')} {m.get('council_title','')}"
            for m in mins) or "  （この会期の会議録は未取得）"
        return (
            f"議案番号: {b.get('number') or '（番号なし）'}\n"
            f"件名: {b.get('title','')}\n"
            f"種別: {b.get('item_type') or '—'}\n"
            f"結果: {b.get('result') or '—'}\n"
            f"提出: {b.get('submitter_group') or '—'} {b.get('submitted_date_text') or ''}\n"
            f"会期: {b.get('session_title','')}（{b.get('session_year','')}）\n"
            f"概要: {b.get('summary') or '（なし）'}\n"
            f"出典URL: {b.get('session_url','')}\n"
            f"添付PDF:\n{pdfs}\n"
            f"会期の会議録ファイル:\n{min_list}\n"
            f"※会議録の該当箇所を読むには get_minutes(record_id) で位置一覧を取得。"
        )

    # -- minutes: locate / read -------------------------------------------

    def get_minutes(self, record_id: str, query: str = "") -> str:
        """Read a bill's session minutes, located by the BILL'S OWN number.

        The agent never supplies a search keyword: it identifies the bill by
        `record_id` (obtained from search), and the fetcher locates that bill in
        its session's minutes using the bill's議案番号 core (e.g. 市第123号議案
        -> "123号"). `query` is used ONLY to read a specific occurrence:

          query == ""       -> LOCATE: list numbered occurrences of this bill.
          query == "hit=N"  -> READ: text window around occurrence N.

        Bills without a 議案番号 (請願・専決処分報告 等) cannot be located by number;
        the agent can still see the session's minute files via get_bill.
        Occurrence order is deterministic (file order, then position in file).
        """
        if record_id not in self.bills:
            raise KeyError(record_id)
        files = self._minute_files(record_id)
        if not files:
            return "この議案の会期に対応する会議録は取得されていません。"

        term = self._bill_number_core(record_id)
        if term is None:
            return ("この議案には議案番号がなく、会議録内の該当箇所を番号で特定でき"
                    "ません。get_bill で会期の会議録ファイル一覧を確認してください。")

        read_n = self._as_hit_index(query)
        occ = self._occurrences(record_id, term)

        if not occ:
            return (f"会議録内で議案番号「{term}」に一致する箇所は見つかりません"
                    "でした。")

        if read_n is None:
            # LOCATE: numbered list of occurrences with tiny previews.
            lines = [f"議案番号「{term}」の出現箇所 {len(occ)} 件"
                     "（読むには query=\"hit=<番号>\" を指定）:"]
            for i, o in enumerate(occ[:_MAX_HITS], 1):
                lines.append(f"[{i}] {o['date']} {o['council']} … {o['preview']}")
            return "\n".join(lines)

        # READ: the window around occurrence read_n.
        if not (1 <= read_n <= len(occ)):
            return f"hit 番号は 1〜{len(occ)} で指定してください。"
        o = occ[read_n - 1]
        return (
            f"〔会議録抜粋〕{o['date']} {o['council']}\n"
            f"ファイル: {o['text_path']}\n出典URL: {o['source_url']}\n"
            f"--- 抜粋（前後{_WINDOW}字）---\n{o['window']}"
        )

    # -- internals ---------------------------------------------------------

    def _has_minutes(self, record_id: str) -> bool:
        return bool(self._minute_files(record_id))

    def _minute_files(self, record_id: str) -> list[dict]:
        rec = self.minutes_map.get(record_id) or {}
        return rec.get("candidate_minute_files") or []

    def _bill_number_core(self, record_id: str) -> str | None:
        """The bill's 議案番号 core used to locate it in minutes: digits + 号
        (NFKC-normalized so fullwidth digits match), or None if it has no number."""
        num = self.bills[record_id].get("number")
        if not num:
            return None
        m = re.search(r"(\d+)\s*号", _norm(num))
        return m.group(1) + "号" if m else None

    @staticmethod
    def _as_hit_index(query: str):
        """Return N if query is 'hit=N' or a bare integer, else None."""
        q = (query or "").strip()
        m = re.fullmatch(r"(?:hit=)?(\d+)", q)
        return int(m.group(1)) if m else None

    def _body(self, text: str) -> str:
        """Slice from the 2nd 開議 (skip the agenda/junk header)."""
        pos = [m.start() for m in re.finditer("開議", text)]
        return text[pos[1]:] if len(pos) >= 2 else text

    def _occurrences(self, record_id: str, term: str | None) -> list[dict]:
        """All occurrences of `term` across the session's minute files.

        Deterministic order: file order in the mapping, then position in file.
        Each occurrence carries a preview and a full window (computed once).
        """
        if not term:
            return []
        out: list[dict] = []
        for f in self._minute_files(record_id):
            tp = f.get("local_text_path")
            if not tp:
                continue
            p = self.root / tp
            if not p.exists():
                continue
            body = self._body(_norm(p.read_text(encoding="utf-8")))
            for m in re.finditer(re.escape(term), body):
                i = m.start()
                out.append({
                    "date": f.get("meeting_date_text", "?"),
                    "council": f.get("council_title", ""),
                    "text_path": tp,
                    "source_url": f.get("source_url", ""),
                    "preview": body[i:i + _PREVIEW].replace("\n", " "),
                    "window": body[max(0, i - _WINDOW):i + _WINDOW],
                })
                if len(out) >= _MAX_HITS:
                    return out
        return out
