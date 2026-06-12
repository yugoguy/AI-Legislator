"""e-Gov 法令API (Japanese statutes) tool — API v2, full-text search.

Finds and fetches Japanese laws/ordinances. Two modes:
  - search (default): FULL-TEXT keyword search over law CONTENT (not just names).
  - fetch: full text of a law by id/number.

Endpoints (e-Gov 法令API v2; base https://laws.e-gov.go.jp/api/2):
  - Search: GET /api/2/keyword   -> 全文検索 (searches law body text), JSON.
  - Fetch:  GET /api/2/law_data/{law_id_or_num}?response_format=json
            -> a JSON tree of {"tag","children",...} nodes.

History: the previous version searched law NAMES only (/api/2/laws?law_title=),
which is 法令名検索, not キーワード検索. This version uses /api/2/keyword so a
query matches text appearing inside the statutes, like the legacy v1 /keyword.

The v2 list/keyword responses nest identity fields in separate objects
(`law_info.law_id`, `revision_info.law_title`), so the parser reads those
explicitly and also falls back to a generic tree walk if the shape differs.
"""

from __future__ import annotations

import backoff
import requests

from tools import BaseTool, ToolResult

_V2 = "https://laws.e-gov.go.jp/api/2"
_TITLE_KEYS = ("law_title", "LawName", "law_name", "title")
_ID_KEYS = ("law_id", "law_num", "LawId", "law_no", "LawNo", "law_number")


class EgovLawTool(BaseTool):
    def __init__(self, max_results: int = 10):
        super().__init__(
            name="SearchEgovLaw",
            description=(
                "Full-text search of Japanese statutes (e-Gov 法令, API v2). "
                "Provide a Japanese `query` to find laws whose CONTENT mentions "
                "it, or a `law_id` (法令ID/法令番号) to fetch a law's full text."
            ),
            parameters=[
                {"name": "query", "type": "str",
                 "description": "Keyword searched within law text (Japanese)."},
                {"name": "law_id", "type": "str",
                 "description": "Law id or number to fetch full text (optional)."},
            ],
        )
        self.max_results = max_results

    def use_tool(self, query: str = "", law_id: str = "") -> ToolResult:
        try:
            if law_id:
                return ToolResult(True, self._format_text(self._fetch(law_id)))
            if query:
                return ToolResult(True, self._format_search(self._search(query)))
            return ToolResult(False, "Provide either `query` or `law_id`.")
        except requests.HTTPError as e:
            return ToolResult(False, f"e-Gov HTTP error: {e}")
        except requests.RequestException as e:
            return ToolResult(False, f"e-Gov request error: {e}")

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _search(self, query: str) -> dict:
        rsp = requests.get(
            f"{_V2}/keyword",
            params={"keyword": query, "limit": self.max_results,
                    "response_format": "json"},
            timeout=30,
        )
        rsp.raise_for_status()
        return rsp.json()

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _fetch(self, law_id: str) -> dict:
        rsp = requests.get(f"{_V2}/law_data/{law_id}",
                           params={"response_format": "json"}, timeout=30)
        rsp.raise_for_status()
        return rsp.json()

    def _format_search(self, payload) -> str:
        items = payload.get("items") or payload.get("laws") or []
        out = []
        for it in items[: self.max_results]:
            info = it.get("law_info", it) if isinstance(it, dict) else {}
            rev = it.get("revision_info", it) if isinstance(it, dict) else {}
            lid = _first(info, _ID_KEYS) or _first(it, _ID_KEYS)
            title = _first(rev, _TITLE_KEYS) or _first(it, _TITLE_KEYS)
            snippet = _snippet(it)
            if lid or title:
                line = f"[{lid or '?'}] {title or '?'}"
                out.append(line + (f" … {snippet}" if snippet else ""))
        if not out:  # shape differed; fall back to a generic walk
            out = [f"[{i}] {t}" for t, i in _walk_pairs(payload)][: self.max_results]
        return "\n".join(out) if out else "No matching laws found."

    @staticmethod
    def _format_text(payload) -> str:
        text = _collect_text(payload).strip()
        return text[:4000] if text else "No law text returned."


def _first(d, keys):
    if isinstance(d, dict):
        for k in keys:
            if d.get(k):
                return str(d[k])
    return None


def _snippet(item) -> str:
    """Pull any sentence/highlight text the keyword hit returned."""
    for k in ("sentence", "text", "snippet", "highlight", "matched_text"):
        v = item.get(k) if isinstance(item, dict) else None
        if isinstance(v, str) and v.strip():
            return v.strip().replace("\n", " ")[:160]
    return ""


def _walk_pairs(node, found=None):
    if found is None:
        found = []
    if isinstance(node, dict):
        t = _first(node, _TITLE_KEYS)
        i = _first(node, _ID_KEYS)
        if t and i:
            found.append((t, i))
        for v in node.values():
            _walk_pairs(v, found)
    elif isinstance(node, list):
        for x in node:
            _walk_pairs(x, found)
    return found


def _collect_text(node) -> str:
    if isinstance(node, str):
        return node + " "
    if isinstance(node, list):
        return "".join(_collect_text(x) for x in node)
    if isinstance(node, dict):
        if "children" in node:
            return _collect_text(node["children"])
        return "".join(_collect_text(v) for v in node.values())
    return ""
