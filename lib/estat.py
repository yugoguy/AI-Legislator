"""e-Stat (政府統計の総合窓口) API tool.

Wraps the e-Stat REST API v3.0. Two modes:
  - search (default): getStatsList by keyword -> matching statistical tables.
  - fetch: getStatsData by statsDataId -> the table's values.

API facts (verified against the e-Stat manual 3.0):
  - Base: https://api.e-stat.go.jp/rest/3.0/app/json/
  - appId is REQUIRED (free registration). Read from env ESTAT_APP_ID.
  - getStatsList key params: searchWord, statsCode, limit.
  - getStatsData key params: statsDataId, limit, metaGetFlg.

Pattern: subclasses tools.BaseTool; mirrors Sakana semantic_scholar.py
(requests + backoff + use_tool returning a formatted string).
"""

from __future__ import annotations

import os
from typing import Any

import backoff
import requests

from tools import BaseTool

_BASE = "https://api.e-stat.go.jp/rest/3.0/app/json"


class EstatTool(BaseTool):
    def __init__(self, max_results: int = 10):
        super().__init__(
            name="SearchEStat",
            description=(
                "Search Japanese government statistics (e-Stat). Provide a "
                "Japanese keyword `query` to find statistical tables, or a "
                "`stats_data_id` to fetch a specific table's values."
            ),
            parameters=[
                {"name": "query", "type": "str",
                 "description": "Keyword to search statistical tables (Japanese)."},
                {"name": "stats_data_id", "type": "str",
                 "description": "A table id to fetch values for (optional)."},
            ],
        )
        self.max_results = max_results
        self.app_id = os.getenv("ESTAT_APP_ID")

    def use_tool(self, query: str = "", stats_data_id: str = "") -> str:
        if not self.app_id:
            return "e-Stat error: ESTAT_APP_ID is not set."
        try:
            if stats_data_id:
                return self._format_data(self._get_data(stats_data_id))
            if query:
                return self._format_list(self._get_list(query))
            return "Provide either `query` or `stats_data_id`."
        except requests.HTTPError as e:
            return f"e-Stat HTTP error: {e}"

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _get_list(self, query: str) -> dict:
        rsp = requests.get(
            f"{_BASE}/getStatsList",
            params={"appId": self.app_id, "searchWord": query,
                    "limit": self.max_results, "lang": "J"},
            timeout=30,
        )
        rsp.raise_for_status()
        return rsp.json()

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _get_data(self, stats_data_id: str) -> dict:
        rsp = requests.get(
            f"{_BASE}/getStatsData",
            params={"appId": self.app_id, "statsDataId": stats_data_id,
                    "limit": max(self.max_results, 100), "lang": "J", "metaGetFlg": "Y"},
            timeout=30,
        )
        rsp.raise_for_status()
        return rsp.json()

    @staticmethod
    def _format_list(payload: dict) -> str:
        root = payload.get("GET_STATS_LIST", {})
        info = root.get("DATALIST_INF", {})
        tables = info.get("TABLE_INF", [])
        if isinstance(tables, dict):
            tables = [tables]
        if not tables:
            return "No statistical tables found."
        out = []
        for t in tables:
            tid = t.get("@id", "?")
            name = _txt(t.get("STAT_NAME"))
            title = _txt(t.get("TITLE"))
            org = _txt(t.get("GOV_ORG"))
            out.append(f"[{tid}] {name} / {title} ({org})")
        return "\n".join(out)

    @staticmethod
    def _format_data(payload: dict) -> str:
        root = payload.get("GET_STATS_DATA", {})
        stat = root.get("STATISTICAL_DATA", {})
        values = stat.get("DATA_INF", {}).get("VALUE", [])
        if isinstance(values, dict):
            values = [values]
        if not values:
            return "No data values returned."
        out = []
        for v in values[:50]:
            out.append(f"{ {k: v[k] for k in v if k != '$'} } = {v.get('$')}")
        return "\n".join(out)


def _txt(node: Any) -> str:
    """e-Stat returns either a string or {"@code":..., "$": text}."""
    if isinstance(node, dict):
        return str(node.get("$", ""))
    return "" if node is None else str(node)
