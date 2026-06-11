"""e-Gov 法令API (Japanese statutes) tool.

Finds and fetches the text of Japanese laws/ordinances. Two modes:
  - search (default): keyword search -> matching laws (id + name).
  - fetch: full text of a law by id/number.

API facts (verified June 2026):
  - e-Gov 法令API v2 released 2025-03; base https://laws.e-gov.go.jp/api/2 ,
    full text via /law_data/{law_id_or_num}?response_format=json (clean JSON tree).
  - Keyword search uses the still-available v1 endpoint
    https://laws.e-gov.go.jp/api/1/keyword?keyword=... (returns JSON). The v2
    search resource (see /api/2/swagger-ui) can be swapped in here later.

This interprets the plan's "e-gov" source as the 法令 (statute) API, the
actionable one for a legislator. Pattern mirrors Sakana semantic_scholar.py.
"""

from __future__ import annotations

import backoff
import requests

from tools import BaseTool

_V1 = "https://laws.e-gov.go.jp/api/1"
_V2 = "https://laws.e-gov.go.jp/api/2"


class EgovLawTool(BaseTool):
    def __init__(self, max_results: int = 10):
        super().__init__(
            name="SearchEgovLaw",
            description=(
                "Search and fetch Japanese statutes (e-Gov 法令). Provide a "
                "Japanese `query` to find laws, or a `law_id` (法令ID/番号) to "
                "fetch a law's full text."
            ),
            parameters=[
                {"name": "query", "type": "str",
                 "description": "Keyword to search law text/names (Japanese)."},
                {"name": "law_id", "type": "str",
                 "description": "Law id or number to fetch full text (optional)."},
            ],
        )
        self.max_results = max_results

    def use_tool(self, query: str = "", law_id: str = "") -> str:
        try:
            if law_id:
                return self._format_text(self._fetch(law_id))
            if query:
                return self._format_search(self._search(query))
            return "Provide either `query` or `law_id`."
        except requests.HTTPError as e:
            return f"e-Gov HTTP error: {e}"

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _search(self, query: str) -> dict:
        rsp = requests.get(f"{_V1}/keyword", params={"keyword": query}, timeout=30)
        rsp.raise_for_status()
        return rsp.json()

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _fetch(self, law_id: str) -> dict:
        rsp = requests.get(f"{_V2}/law_data/{law_id}",
                           params={"response_format": "json"}, timeout=30)
        rsp.raise_for_status()
        return rsp.json()

    def _format_search(self, payload: dict) -> str:
        # v1 keyword wraps results under DataRoot/ApplData/LawNameListInfo.
        appl = payload.get("DataRoot", {}).get("ApplData", {})
        items = appl.get("LawNameListInfo", [])
        if isinstance(items, dict):
            items = [items]
        if not items:
            return "No matching laws found."
        out = []
        for it in items[: self.max_results]:
            out.append(f"[{it.get('LawId','?')}] {it.get('LawName','?')} "
                       f"({it.get('LawNo','')})")
        return "\n".join(out)

    @staticmethod
    def _format_text(payload: dict) -> str:
        text = _collect_text(payload)
        return text[:4000] if text else "No law text returned."


def _collect_text(node) -> str:
    """Recursively join text from the v2 law_data JSON tree."""
    if isinstance(node, str):
        return node + " "
    if isinstance(node, list):
        return "".join(_collect_text(x) for x in node)
    if isinstance(node, dict):
        if "children" in node:
            return _collect_text(node["children"])
        return "".join(_collect_text(v) for v in node.values())
    return ""
