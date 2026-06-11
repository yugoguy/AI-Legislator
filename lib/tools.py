"""Tool interface + registry.

`BaseTool` is lifted verbatim from AI Scientist `ai_scientist/tools/base_tool.py`
(the abstract contract every data-source tool conforms to). `build_default_tools`
assembles the concrete Japanese data-source tools into the {name: tool} registry
that research.py consumes, skipping any whose credentials are absent.

Each concrete tool lives in its own file (estat.py, kokkai.py, egov.py,
resas.py, webscrape.py) and subclasses BaseTool, mirroring the structure of
Sakana's semantic_scholar.py.
"""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for a tool (lifted from Sakana base_tool.py).

    - name: the tool's invocation name (used as the ACTION token).
    - description: one line shown to the model in the tool catalog.
    - parameters: list of {"name","type","description"} dicts.
    """

    def __init__(self, name: str, description: str, parameters: list[dict[str, Any]]):
        self.name = name
        self.description = description
        self.parameters = parameters

    @abstractmethod
    def use_tool(self, **kwargs) -> Any:
        """Run the tool. Subclasses return a human-readable string for the model."""
        raise NotImplementedError


def build_default_tools(*, include_resas: bool = False) -> dict[str, "BaseTool"]:
    """Assemble the data-source registry, skipping tools missing credentials.

    `include_resas` is off by default because the RESAS API was discontinued on
    2025-03-24 (see resas.py); enable only if you have a working endpoint/key.
    """
    from estat import EstatTool
    from kokkai import KokkaiTool
    from egov import EgovLawTool
    from webscrape import WebScrapeTool

    tools: dict[str, BaseTool] = {}

    if os.getenv("ESTAT_APP_ID"):
        t = EstatTool()
        tools[t.name] = t
    else:
        warnings.warn("ESTAT_APP_ID not set; e-Stat tool disabled.")

    for cls in (KokkaiTool, EgovLawTool, WebScrapeTool):  # no key required
        t = cls()
        tools[t.name] = t

    if include_resas:
        from resas import ResasTool

        if os.getenv("RESAS_API_KEY"):
            t = ResasTool()
            tools[t.name] = t
        else:
            warnings.warn("RESAS_API_KEY not set; RESAS tool disabled.")

    return tools
