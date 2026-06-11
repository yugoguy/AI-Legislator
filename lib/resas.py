"""RESAS (地域経済分析システム) API tool.

================================ DISCONTINUED ================================
The RESAS API was shut down on 2025-03-24 by the Cabinet Office. Calls to the
endpoint below will fail. The Cabinet Office points to the MLIT data platform
(国土交通省データプラットフォーム, GraphQL) as the successor for prefecture /
municipality and regional-economy data.

This file is kept (per the source list) implementing the documented v1 REST
interface so the call shape is preserved and the base URL can be swapped for the
successor, OR removed entirely. It is OFF by default in tools.build_default_tools
(`include_resas=False`).
=============================================================================

Original API facts (for reference):
  - Base: https://opendata.resas-portal.go.jp/api/v1/
  - Auth header: X-API-KEY (from env RESAS_API_KEY).
  - Resources e.g. prefectures, cities, population/composition/perYear, etc.
"""

from __future__ import annotations

import os

import backoff
import requests

from tools import BaseTool

_BASE = "https://opendata.resas-portal.go.jp/api/v1"


class ResasTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="QueryRESAS",
            description=(
                "[DISCONTINUED 2025-03-24] Query RESAS regional-economy data by "
                "`path` (e.g. 'population/composition/perYear') and `params` dict. "
                "Endpoint is offline; swap base URL for the MLIT successor if used."
            ),
            parameters=[
                {"name": "path", "type": "str",
                 "description": "API resource path under /api/v1/."},
                {"name": "params", "type": "dict",
                 "description": "Query parameters for the resource (optional)."},
            ],
        )
        self.api_key = os.getenv("RESAS_API_KEY")

    def use_tool(self, path: str = "", params: dict | None = None) -> str:
        if not self.api_key:
            return "RESAS error: RESAS_API_KEY is not set (service discontinued)."
        if not path:
            return "Provide a resource `path`."
        try:
            return str(self._get(path, params or {}))
        except requests.HTTPError as e:
            return (f"RESAS HTTP error: {e}. Note: the RESAS API was discontinued "
                    "on 2025-03-24; use the MLIT data platform instead.")

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def _get(self, path: str, params: dict) -> dict:
        rsp = requests.get(f"{_BASE}/{path.lstrip('/')}",
                           headers={"X-API-KEY": self.api_key},
                           params=params, timeout=30)
        rsp.raise_for_status()
        return rsp.json()
