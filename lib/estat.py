"""e-Stat (政府統計の総合窓口) API tool.

Wraps the e-Stat REST API v3.0. Two modes: search (getStatsList by keyword to
matching statistical tables) and fetch (getStatsData by statsDataId to a table's
values). appId is required (free registration) and read from ESTAT_APP_ID.

A fetch pulls the WHOLE table by following e-Stat's NEXT_KEY pagination, then
writes it to the caller's data directory as raw JSON plus a decoded CSV, and
returns the saved paths. The data agent reads those files; without this the agent
receives an empty working directory and can only analyze numbers retyped by hand.
"""

import csv
import json
import os
from pathlib import Path
from typing import Any

import backoff
import requests

from tools import BaseTool, ToolResult

_BASE = "https://api.e-stat.go.jp/rest/3.0/app/json"
_PAGE = 100000          # e-Stat's per-request maximum
_MAX_PAGES = 50         # hard stop, so a huge table cannot spin forever


class EstatTool(BaseTool):
    def __init__(self, max_results: int = 10):
        super().__init__(
            name="SearchEStat",
            description=(
                "Search Japanese government statistics (e-Stat). Provide a "
                "Japanese keyword `query` to find statistical tables, or a "
                "`stats_data_id` to fetch a table. A fetch downloads the whole "
                "table and SAVES it to the working directory as .json and .csv, "
                "so AnalyzeData can read it with pandas."
            ),
            parameters=[
                {"name": "query", "type": "str",
                 "description": "Keyword to search statistical tables (Japanese)."},
                {"name": "stats_data_id", "type": "str",
                 "description": "A table id to download and save (optional)."},
            ],
        )
        self.max_results = max_results
        self.app_id = os.getenv("ESTAT_APP_ID")
        # Where a fetched table is written. Set per research node by the caller.
        self.data_dir: Path | None = None

    def set_data_dir(self, data_dir: Path | str) -> None:
        """Point the tool at the current research node's data directory."""
        self.data_dir = Path(data_dir)

    def _redact(self, text: str) -> str:
        """Mask the appId so it never reaches recorded text (error strings carry
        the full request URL). Does not affect what is sent to e-Stat."""
        return text.replace(self.app_id, "***") if self.app_id else text

    def use_tool(self, query: str = "", stats_data_id: str = "") -> ToolResult:
        if not self.app_id:
            return ToolResult(False, "e-Stat error: ESTAT_APP_ID is not set.")
        try:
            if stats_data_id:
                return self._fetch_and_save(stats_data_id)
            if query:
                return ToolResult(True, self._search_union(query))
            return ToolResult(False, "Provide either `query` or `stats_data_id`.")
        except requests.HTTPError as e:
            return ToolResult(False, self._redact(f"e-Stat HTTP error: {e}"))
        except requests.RequestException as e:
            return ToolResult(False, self._redact(f"e-Stat request error: {e}"))

    # search

    def _search_union(self, query: str) -> str:
        """Search each whitespace-separated term separately and union results.

        e-Stat's searchWord treats a multi-term string as AND, which usually
        over-narrows to zero hits. One call per term, unioned and deduplicated by
        table id, approximates an OR search.
        """
        terms = query.split()
        if len(terms) <= 1:
            return self._format_tables(_estat_tables(self._get_list(query)))
        seen: set[str] = set()
        merged: list[dict] = []
        for term in terms:
            for t in _estat_tables(self._get_list(term)):
                tid = t.get("@id", "")
                if tid and tid in seen:
                    continue
                seen.add(tid)
                merged.append(t)
                if len(merged) >= self.max_results:
                    break
            if len(merged) >= self.max_results:
                break
        return self._format_tables(merged)

    # fetch + save

    def _fetch_and_save(self, stats_data_id: str) -> ToolResult:
        """Download the whole table, write it to data_dir, report the saved paths."""
        values, class_map, table_name = self._get_all_data(stats_data_id)
        if not values:
            # A valid call that legitimately found nothing is still a success.
            return ToolResult(True, f"Table {stats_data_id} returned no data values.")

        if self.data_dir is None:
            # No directory to write to: fall back to a text preview only.
            return ToolResult(True, self._preview(values, class_map, stats_data_id,
                                                  table_name, saved=None))

        self.data_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.data_dir / f"estat_{stats_data_id}.json"
        csv_path = self.data_dir / f"estat_{stats_data_id}.csv"

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump({"statsDataId": stats_data_id, "table": table_name,
                       "class_map": class_map, "values": values},
                      fh, ensure_ascii=False)
        _write_csv(csv_path, values, class_map)

        return ToolResult(True, self._preview(values, class_map, stats_data_id,
                                              table_name,
                                              saved=(csv_path, json_path)))

    def _get_all_data(self, stats_data_id: str) -> tuple[list[dict], dict, str]:
        """Follow NEXT_KEY until the table is exhausted.

        Returns (all value records, dimension code->label maps, table name).
        """
        values: list[dict] = []
        class_map: dict[str, dict[str, str]] = {}
        table_name = ""
        start: int | None = None

        for _ in range(_MAX_PAGES):
            payload = self._get_data(stats_data_id, start)
            stat = payload.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {})
            if not class_map:
                class_map = _class_map(stat)
                table_name = _table_name(stat)

            chunk = stat.get("DATA_INF", {}).get("VALUE", [])
            if isinstance(chunk, dict):
                chunk = [chunk]
            values.extend(chunk)

            # NEXT_KEY is present only while more rows remain.
            next_key = stat.get("RESULT_INF", {}).get("NEXT_KEY")
            if not next_key:
                break
            start = int(next_key)

        return values, class_map, table_name

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
    def _get_data(self, stats_data_id: str, start: int | None = None) -> dict:
        params = {"appId": self.app_id, "statsDataId": stats_data_id,
                  "limit": _PAGE, "lang": "J", "metaGetFlg": "Y"}
        if start is not None:
            params["startPosition"] = start
        rsp = requests.get(f"{_BASE}/getStatsData", params=params, timeout=60)
        rsp.raise_for_status()
        return rsp.json()

    # formatting

    @staticmethod
    def _format_tables(tables: list) -> str:
        if not tables:
            return "No statistical tables found."
        out = []
        for t in tables:
            tid = t.get("@id", "?")
            name = _txt(t.get("STAT_NAME"))
            title = _txt(t.get("TITLE"))
            org = _txt(t.get("GOV_ORG"))
            out.append(f"[{tid}] {name} / {title} ({org})")
        out.append("\nTo download a table, call SearchEStat with stats_data_id=<id>.")
        return "\n".join(out)

    @staticmethod
    def _preview(values: list[dict], class_map: dict, stats_data_id: str,
                 table_name: str, saved: tuple[Path, Path] | None) -> str:
        """Tell the model what was downloaded and, above all, where it now lives."""
        lines = [f"Downloaded {len(values)} rows from table {stats_data_id}"
                 + (f" ({table_name})" if table_name else "") + "."]
        if saved:
            csv_path, json_path = saved
            lines.append(f"Saved to: {csv_path.name} and {json_path.name} "
                         f"in the working directory.")
            lines.append("AnalyzeData can read it, e.g. "
                         f"pd.read_csv('{csv_path.name}').")
        else:
            lines.append("(Not saved: no working directory was set.)")
        lines.append("")
        lines.append("Columns: " + ", ".join(_columns(values, class_map)))
        lines.append("First rows:")
        for row in _decoded_rows(values[:5], class_map):
            lines.append("  " + " | ".join(f"{k}={v}" for k, v in row.items()))
        return "\n".join(lines)


# payload helpers

def _estat_tables(payload: dict) -> list:
    """Extract the TABLE_INF list from a getStatsList payload."""
    root = payload.get("GET_STATS_LIST", {})
    tables = root.get("DATALIST_INF", {}).get("TABLE_INF", [])
    if isinstance(tables, dict):
        tables = [tables]
    return tables or []


def _table_name(stat: dict) -> str:
    return _txt(stat.get("TABLE_INF", {}).get("TITLE"))


def _class_map(stat: dict) -> dict[str, dict[str, str]]:
    """Build {dimension id: {code: label}} from CLASS_INF (needs metaGetFlg=Y).

    A VALUE record keys its dimensions as "@cat01", "@area", "@time" and so on;
    this maps those codes back to human labels for the CSV.
    """
    objs = stat.get("CLASS_INF", {}).get("CLASS_OBJ", [])
    if isinstance(objs, dict):
        objs = [objs]
    out: dict[str, dict[str, str]] = {}
    for obj in objs:
        dim_id = obj.get("@id")
        classes = obj.get("CLASS", [])
        if isinstance(classes, dict):
            classes = [classes]
        out[dim_id] = {c.get("@code"): c.get("@name", "") for c in classes}
    return out


def _columns(values: list[dict], class_map: dict) -> list[str]:
    keys = [k for k in values[0].keys() if k != "$"]
    return [k.lstrip("@") for k in keys] + ["value"]


def _decoded_rows(values: list[dict], class_map: dict) -> list[dict]:
    """Turn raw VALUE records into rows with labels instead of bare codes."""
    rows = []
    for v in values:
        row: dict[str, Any] = {}
        for k, code in v.items():
            if k == "$":
                continue
            dim = k.lstrip("@")
            label = class_map.get(dim, {}).get(code, code)
            row[dim] = label
        row["value"] = v.get("$")
        rows.append(row)
    return rows


def _write_csv(path: Path, values: list[dict], class_map: dict) -> None:
    rows = _decoded_rows(values, class_map)
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _txt(node: Any) -> str:
    """e-Stat returns either a string or {"@code": ..., "$": text}."""
    if isinstance(node, dict):
        return str(node.get("$", ""))
    return "" if node is None else str(node)
