"""Web search tool (provider-native, returns real pages).

The fix for the v1.1 failure where the agent invented URLs for FetchWebPage and
then mistook a 404 for evidence. This tool takes a `query` and returns real
candidate pages — titles, URLs, and short summaries — by delegating to the LLM
provider's own server-side web-search tool (Anthropic web_search_20250305 /
OpenAI web_search_options) via llm.web_search. The agent never guesses a URL: it
searches, sees real results, and can then FetchWebPage one of the returned URLs.

The executor model is configured separately (cfg web-search spec), so searches
can run on a cheap search-capable model independent of the legislator model. For
OpenAI it must be a search model string (e.g. gpt-4o-mini-search-preview); any
Anthropic model supports the native web-search tool directly.
"""

from __future__ import annotations

from tools import BaseTool, ToolResult
from llm import web_search


class WebSearchTool(BaseTool):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", max_results: int = 5):
        super().__init__(
            name="SearchWeb",
            description=(
                "Search the web for real pages by `query` and get back titles, "
                "URLs, and summaries. Use this to DISCOVER local-government and "
                "other pages — then FetchWebPage one of the returned URLs. Never "
                "invent a URL; search first."
            ),
            parameters=[
                {"name": "query", "type": "str",
                 "description": "What to search for (Japanese keywords work best)."},
            ],
        )
        self.model = model
        self.max_results = max_results

    def use_tool(self, query: str = "") -> ToolResult:
        if not query:
            return ToolResult(False, "Provide a `query`.")
        ok, text, _sources = web_search(query, self.model,
                                        max_results=self.max_results)
        return ToolResult(ok, text)
