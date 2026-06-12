"""国会会議録検索システム (National Diet proceedings) API tool.

Searches Diet speech records by keyword / speaker / house / date range.

API facts (verified against kokkai.ndl.go.jp/api.html):
  - No authentication required.
  - Speech-unit endpoint: https://kokkai.ndl.go.jp/api/speech
    (also meeting / meeting_list for whole-meeting output).
  - JSON via recordPacking=json. Up to 100 speech records per request.
  - Params: any (keyword), speaker, nameOfHouse (衆議院/参議院/両院),
    from / until (YYYY-MM-DD), maximumRecords, startRecord.

Pattern: subclasses tools.BaseTool; mirrors Sakana semantic_scholar.py.
"""

from __future__ import annotations

import backoff
import requests

from tools import BaseTool, ToolResult

_URL = "https://kokkai.ndl.go.jp/api/speech"


class KokkaiTool(BaseTool):
    def __init__(self, max_results: int = 10):
        super().__init__(
            name="SearchKokkai",
            description=(
                "Search Japanese National Diet proceedings (国会会議録). Provide a "
                "Japanese `query` keyword; optionally `speaker`, `house` "
                "(衆議院/参議院), and `since`/`until` dates (YYYY-MM-DD)."
            ),
            parameters=[
                {"name": "query", "type": "str",
                 "description": "Keyword to search speech text (Japanese)."},
                {"name": "speaker", "type": "str", "description": "Speaker name (optional)."},
                {"name": "house", "type": "str",
                 "description": "衆議院 / 参議院 / 両院 (optional)."},
                {"name": "since", "type": "str", "description": "From date YYYY-MM-DD (optional)."},
                {"name": "until", "type": "str", "description": "Until date YYYY-MM-DD (optional)."},
            ],
        )
        self.max_results = max_results

    def use_tool(self, query: str = "", speaker: str = "", house: str = "",
                 since: str = "", until: str = "") -> ToolResult:
        if not (query or speaker):
            return ToolResult(False, "Provide at least `query` or `speaker`.")
        base = {
            "recordPacking": "json",
            "maximumRecords": self.max_results,
        }
        if speaker:
            base["speaker"] = speaker
        if house:
            base["nameOfHouse"] = house
        if since:
            base["from"] = since
        if until:
            base["until"] = until
        try:
            return ToolResult(True, self._search_union(query, base))
        except requests.HTTPError as e:
            return ToolResult(False, f"Kokkai HTTP error: {e}")
        except requests.RequestException as e:
            return ToolResult(False, f"Kokkai request error: {e}")

    def _search_union(self, query: str, base: dict) -> str:
        """Search each whitespace-separated keyword separately and union results.

        The `any` parameter ANDs multiple terms, which usually over-narrows to
        zero hits. Splitting into one call per term and unioning (dedup by
        speechID, capped at max_results) approximates an OR search. Speaker/house/
        date filters in `base` apply to every sub-call.
        """
        terms = query.split() if query else [""]
        if len(terms) <= 1:
            params = dict(base)
            if query:
                params["any"] = query
            return self._format_records(_kokkai_records(self._get(params)))
        seen: set[str] = set()
        merged: list[dict] = []
        for term in terms:
            params = dict(base, any=term)
            for r in _kokkai_records(self._get(params)):
                sid = r.get("speechID", "")
                if sid and sid in seen:
                    continue
                seen.add(sid)
                merged.append(r)
                if len(merged) >= self.max_results:
                    break
            if len(merged) >= self.max_results:
                break
        return self._format_records(merged)

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _get(self, params: dict) -> dict:
        rsp = requests.get(_URL, params=params, timeout=30)
        rsp.raise_for_status()
        return rsp.json()

    @staticmethod
    def _format_records(records: list) -> str:
        if not records:
            return "No Diet speech records found."
        out = [f"showing {len(records)} record(s):"]
        for r in records:
            speech = (r.get("speech") or "").replace("\n", " ")
            out.append(
                f"- {r.get('date','?')} {r.get('nameOfHouse','')} "
                f"{r.get('nameOfMeeting','')} | {r.get('speaker','?')}: "
                f"{speech[:200]}"
            )
        return "\n".join(out)


def _kokkai_records(payload: dict) -> list:
    """Extract the speechRecord list from a kokkai API payload."""
    return payload.get("speechRecord", []) or []
