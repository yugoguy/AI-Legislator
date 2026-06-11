"""e-Gov 法令API (Japanese statutes) tool — API v2.

Finds and fetches the text of Japanese laws/ordinances. Two modes:
  - search (default): law-list search -> matching laws (id/number + name).
  - fetch: full text of a law by id/number.

Why this was rewritten
----------------------
The previous version mixed API versions: it fetched via v2 but SEARCHED via
`https://laws.e-gov.go.jp/api/1/keyword`, which 404s — the v1 `keyword` resource
is not served on the v2 host (`laws.e-gov.go.jp` serves `/api/2/...`; the legacy
v1 API lived on the `elaws.e-gov.go.jp` host). This version is ALL v2.

API facts (verified June 2026):
  - Base: https://laws.e-gov.go.jp/api/2  (v2 released 2025-03-19;
    OpenAPI/Swagger UI at https://laws.e-gov.go.jp/api/2/swagger-ui).
  - Search: GET /api/2/laws  (law list with name filtering).
  - Fetch:  GET /api/2/law_data/{law_id_or_num}?response_format=json
            -> a JSON tree of {"tag","children",...} nodes.

Both responses are parsed defensively (keys located by walking the tree) so the
tool tolerates minor field-name differences across the v2 schema, and search
results are additionally filtered locally by the query as a safety net. If the
live schema differs, only the small parser helpers here need adjusting.
"""

from __future__ import annotations

import backoff
import requests

from tools import BaseTool

_V2 = "https://laws.e-gov.go.jp/api/2"

# Candidate field names for title / id, used by the defensive parser.
_TITLE_KEYS = ("law_title", "LawName", "law_name", "title")
_ID_KEYS = ("law_id", "LawId", "law_num", "law_no", "LawNo", "law_number")


class EgovLawTool(BaseTool):
    def __init__(self, max_results: int = 10):
        super().__init__(
            name="SearchEgovLaw",
            description=(
                "Search and fetch Japanese statutes (e-Gov 法令, API v2). Provide "
                "a Japanese `query` to find laws by name, or a `law_id` "
                "(法令ID/法令番号) to fetch a law's full text."
            ),
            parameters=[
                {"name": "query", "type": "str",
                 "description": "Keyword to search law names (Japanese)."},
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
                return self._format_search(self._search(query), query)
            return "Provide either `query` or `law_id`."
        except requests.HTTPError as e:
            return f"e-Gov HTTP error: {e}"
        except requests.RequestException as e:
            return f"e-Gov request error: {e}"

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _search(self, query: str) -> dict:
        rsp = requests.get(
            f"{_V2}/laws",
            params={"law_title": query, "limit": self.max_results,
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

    def _format_search(self, payload, query: str) -> str:
        pairs = _find_law_pairs(payload)
        # Local safety-net filter; fall back to all results if it empties out.
        filtered = [(t, i) for (t, i) in pairs if query in t] or pairs
        seen, out = set(), []
        for title, lid in filtered:
            if lid in seen:
                continue
            seen.add(lid)
            out.append(f"[{lid}] {title}")
            if len(out) >= self.max_results:
                break
        return "\n".join(out) if out else "No matching laws found."

    @staticmethod
    def _format_text(payload) -> str:
        text = _collect_text(payload).strip()
        return text[:4000] if text else "No law text returned."


def _find_law_pairs(node, found=None) -> list[tuple[str, str]]:
    """Walk the JSON and collect (title, id) pairs by candidate key names."""
    if found is None:
        found = []
    if isinstance(node, dict):
        title = next((node[k] for k in _TITLE_KEYS if node.get(k)), None)
        lid = next((node[k] for k in _ID_KEYS if node.get(k)), None)
        if title and lid:
            found.append((str(title), str(lid)))
        for v in node.values():
            _find_law_pairs(v, found)
    elif isinstance(node, list):
        for x in node:
            _find_law_pairs(x, found)
    return found


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
