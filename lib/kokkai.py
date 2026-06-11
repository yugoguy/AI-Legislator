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

from tools import BaseTool

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
                 since: str = "", until: str = "") -> str:
        if not (query or speaker):
            return "Provide at least `query` or `speaker`."
        params = {
            "recordPacking": "json",
            "maximumRecords": self.max_results,
        }
        if query:
            params["any"] = query
        if speaker:
            params["speaker"] = speaker
        if house:
            params["nameOfHouse"] = house
        if since:
            params["from"] = since
        if until:
            params["until"] = until
        try:
            return self._format(self._get(params))
        except requests.HTTPError as e:
            return f"Kokkai HTTP error: {e}"

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _get(self, params: dict) -> dict:
        rsp = requests.get(_URL, params=params, timeout=30)
        rsp.raise_for_status()
        return rsp.json()

    @staticmethod
    def _format(payload: dict) -> str:
        total = payload.get("numberOfRecords", 0)
        records = payload.get("speechRecord", []) or []
        if not records:
            return "No Diet speech records found."
        out = [f"{total} record(s) matched; showing {len(records)}:"]
        for r in records:
            speech = (r.get("speech") or "").replace("\n", " ")
            out.append(
                f"- {r.get('date','?')} {r.get('nameOfHouse','')} "
                f"{r.get('nameOfMeeting','')} | {r.get('speaker','?')}: "
                f"{speech[:200]}"
            )
        return "\n".join(out)
