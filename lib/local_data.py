"""Local-government data: the general bridge tool (region-agnostic).

This is the BRIDGE between the agent's research loop and whatever region-specific
data happens to be available. It exposes ONE tool, `SearchLocalGov`, that the
agent uses to search local bills (議案・請願), inspect a bill's detail, and read a
session's minutes (会議録) — without the bridge knowing anything about how any
particular region stores its data.

Design
------
- `LocalFetcher` is the abstract contract a region must implement. It is
  deliberately small and format-agnostic: search returns opaque "hit" dicts the
  bridge only formats, never interprets structurally.
- Concrete fetchers (e.g. yokohama_fetcher.YokohamaFetcher) live in their own
  files and own ALL data-structure-specific logic. Adding another city later =
  implement LocalFetcher against that city's data + register it; this file does
  not change.
- The bridge selects the fetcher whose region matches cfg.region and is marked
  available in config (config.local_sources). If none is available the tool says
  so plainly rather than guessing.

Action protocol (single tool, sub-action by `action` arg)
---------------------------------------------------------
  action="search"  query=<keyword>      -> matching bills (id, title, result, ...)
  action="bill"    record_id=<id>       -> one bill's full detail (+ minutes avail)
  action="minutes" record_id=<id>       -> LOCATE: numbered occurrences of the bill
                                           in its session minutes
  action="minutes" record_id=<id> query="hit=N"
                                        -> READ: text window around occurrence N

Keyword search MUST be single-concept-friendly: fetchers union over space-split
terms (OR), so long multi-term queries do not over-narrow (same fix as the
e-Stat / Kokkai tools).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from tools import BaseTool, ToolResult


class LocalFetcher(ABC):
    """Region-specific local-data fetcher. One implementation per region.

    Methods return human-readable strings (the bridge wraps them in ToolResult).
    A failure to load data or an unknown id should raise; the bridge converts
    that into ok=False so it reads as a research-process failure, not evidence.
    """

    #: Human label for the region this fetcher serves (e.g. "横浜市").
    region: str = ""

    @abstractmethod
    def search_bills(self, query: str, max_results: int = 10) -> str:
        """Keyword search over bills/petitions; return a formatted hit list.

        Implementations should split `query` on whitespace and UNION matches per
        term (OR semantics) so multi-term queries do not collapse to zero hits.
        Each hit line should carry at least the bill's stable id, its title, the
        result, the session, and whether session minutes are available.
        """

    @abstractmethod
    def get_bill(self, record_id: str) -> str:
        """Full detail of one bill by its stable id (title, summary, result,
        submitter, date, attached documents, and which session minutes exist)."""

    @abstractmethod
    def get_minutes(self, record_id: str, query: str = "") -> str:
        """Read a bill's session minutes. Two-step, driven by `query`:

          query == ""        -> LOCATE: list numbered occurrences of this bill in
                                its session's minutes (compact previews) so the
                                agent can choose which to read.
          query == "hit=N"   -> READ: return the text window around occurrence N.

        The agent identifies the bill by `record_id` only; the implementation
        derives the search term from the bill (its number). The agent does not
        supply a keyword."""


class LocalGovTool(BaseTool):
    """The single agent-facing bridge tool over a region's local data."""

    def __init__(self, fetcher: LocalFetcher | None, region: str,
                 max_results: int = 10):
        self.fetcher = fetcher
        self.region = region
        self.max_results = max_results
        avail = fetcher is not None
        desc = (
            f"Search {region}'s OWN local-assembly records — bills and petitions "
            f"(議案・請願) and their meeting minutes (会議録). "
            "Use `action`: 'search' with a `query` keyword to find bills; 'bill' "
            "with a `record_id` to read one bill's full detail; 'minutes' with a "
            "`record_id` to find where that bill is discussed in its session "
            "minutes — it returns a numbered list of occurrences, then call again "
            "with `query`=\"hit=<N>\" to read the text around occurrence N. This "
            "is the authoritative source for THIS jurisdiction's legislative "
            "record — prefer it over web search for local bills and debate."
        )
        if not avail:
            desc = (f"(Unavailable for {region}: no local dataset is configured. "
                    "Use other sources.) ") + desc
        super().__init__(
            name="SearchLocalGov",
            description=desc,
            parameters=[
                {"name": "action", "type": "str",
                 "description": "One of: search | bill | minutes."},
                {"name": "query", "type": "str",
                 "description": "For action=search: a single-concept keyword "
                                "(multiple terms are OR-matched). For "
                                "action=minutes: leave empty to list where this "
                                "bill is discussed, then pass \"hit=<N>\" to read "
                                "occurrence N. (Minutes are located by the bill's "
                                "own number; you do not supply a keyword.)"},
                {"name": "record_id", "type": "str",
                 "description": "Bill id for action=bill or action=minutes "
                                "(from a search result)."},
            ],
        )

    def use_tool(self, action: str = "", query: str = "",
                 record_id: str = "") -> ToolResult:
        if self.fetcher is None:
            return ToolResult(False,
                              f"No local dataset is configured for {self.region}.")
        action = (action or "").strip().lower()
        try:
            if action == "search":
                if not query:
                    return ToolResult(False, "action=search needs a `query`.")
                return ToolResult(True, self.fetcher.search_bills(query, self.max_results))
            if action == "bill":
                if not record_id:
                    return ToolResult(False, "action=bill needs a `record_id`.")
                return ToolResult(True, self.fetcher.get_bill(record_id))
            if action == "minutes":
                if not record_id:
                    return ToolResult(False, "action=minutes needs a `record_id`.")
                return ToolResult(True, self.fetcher.get_minutes(record_id, query))
            return ToolResult(False,
                              "Unknown `action`. Use search | bill | minutes.")
        except FileNotFoundError as e:
            return ToolResult(False, f"Local data not found: {e}")
        except KeyError as e:
            return ToolResult(False, f"No such record: {e}")
        except Exception as e:                       # data/parse failure
            return ToolResult(False, f"Local data error: {e}")


def build_local_tool(region: str, local_sources: dict,
                     max_results: int = 10) -> LocalGovTool | None:
    """Construct the bridge tool for `region` from config.local_sources.

    `local_sources` maps region -> {"available": bool, "path": <data dir>}.
    Returns a LocalGovTool wired to the matching region fetcher, or None if the
    region has no configured/available dataset. The mapping of region -> fetcher
    class lives here (the one place that knows which fetchers exist); each fetcher
    itself stays in its own file.
    """
    spec = (local_sources or {}).get(region)
    if not spec or not spec.get("available"):
        return None

    fetcher: LocalFetcher | None = None
    if region == "横浜市":
        try:
            from yokohama_fetcher import YokohamaFetcher
            fetcher = YokohamaFetcher(spec["path"])
        except Exception:
            fetcher = None
    # elif region == "...":  add other regions' fetchers here.

    if fetcher is None:
        return None
    return LocalGovTool(fetcher, region, max_results=max_results)
