"""Tool interface + registry.

`BaseTool` is lifted verbatim from AI Scientist `ai_scientist/tools/base_tool.py`
(the abstract contract every data-source tool conforms to). `build_default_tools`
assembles the concrete Japanese data-source tools into the {name: tool} registry
that research.py consumes, skipping any whose credentials are absent.

Each concrete tool lives in its own file (estat.py, kokkai.py, egov.py,
webscrape.py) and subclasses BaseTool, mirroring the structure of
Sakana's semantic_scholar.py.
"""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """What every tool returns: an explicit success flag plus text for the model.

    `ok` distinguishes a genuine TOOL/PROCESS failure (HTTP error, page not
    found, missing credential, malformed request) from a successful call — even
    one that legitimately found nothing. This matters because the research loop
    uses `ok` to decide whether a step counts as progress: a 404 must NOT be
    recorded as successful research, while an empty-but-valid result still is.

    `text` is the human-readable string shown to the model either way (an error
    message is useful feedback to the agent's PROCESS, but is not evidence about
    any proposal — see the prompts).
    """

    ok: bool
    text: str

    def __str__(self) -> str:
        return self.text


class BaseTool(ABC):
    """Abstract base class for a tool (lifted from Sakana base_tool.py).

    - name: the tool's invocation name (used as the ACTION token).
    - description: one line shown to the model in the tool catalog.
    - parameters: list of {"name","type","description"} dicts.

    use_tool returns a ToolResult(ok, text). `ok=False` signals a tool/process
    failure (not evidence about a proposal); `ok=True` covers success including
    a valid "nothing found".
    """

    def __init__(self, name: str, description: str, parameters: list[dict[str, Any]]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def use_tool(self, **kwargs) -> "ToolResult":
        """Run the tool. Subclasses return a ToolResult(ok, text)."""
        raise NotImplementedError


def build_default_tools(web_search_model: str = "claude-haiku-4-5-20251001",
                        web_search_max_results: int = 5,
                        region: str = "",
                        local_sources: dict | None = None,
                        local_max_results: int = 10) -> dict[str, "BaseTool"]:
    """Assemble the data-source registry, skipping tools missing credentials."""
    from estat import EstatTool
    from kokkai import KokkaiTool
    # from egov import EgovLawTool       # disabled: e-Gov tool currently not working
    # from webscrape import WebScrapeTool  # disabled: replaced by SearchWeb (no URL guessing)
    from websearch import WebSearchTool
    from local_data import build_local_tool

    tools: dict[str, BaseTool] = {}

    if os.getenv("ESTAT_APP_ID"):
        t = EstatTool()
        tools[t.name] = t
    else:
        warnings.warn("ESTAT_APP_ID not set; e-Stat tool disabled.")

    # Local-government data bridge for the target region (authoritative local
    # bills + minutes), if a dataset is configured/available for it.
    local = build_local_tool(region, local_sources or {}, max_results=local_max_results)
    if local is not None:
        tools[local.name] = local
    else:
        warnings.warn(f"No local dataset available for region {region!r}; "
                      "SearchLocalGov disabled.")

    web = WebSearchTool(model=web_search_model, max_results=web_search_max_results)
    tools[web.name] = web                       # discovery via real web search

    t = KokkaiTool()                            # national Diet records
    tools[t.name] = t

    return tools
