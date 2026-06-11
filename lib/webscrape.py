"""Web scrape tool.

Generic fetch + text extraction for the non-API sources in the plan: local
assembly 議事録, 請願・陳情, パブリックコメント, 予算・決算 and other pages
published only as HTML on municipal/prefectural sites.

Returns the main textual content of a single URL. HTML is reduced to text with
BeautifulSoup when available, else a regex tag-strip fallback. Output is trimmed
to keep the model context bounded.

Caution: respecting each site's robots.txt and terms of use is the operator's
responsibility; this tool performs a plain GET with a descriptive User-Agent.
"""

from __future__ import annotations

import re

import backoff
import requests

from tools import BaseTool

_UA = "AI-Legislator/0.1 (research; contact: operator)"
_MAX_CHARS = 6000


class WebScrapeTool(BaseTool):
    def __init__(self, max_chars: int = _MAX_CHARS):
        super().__init__(
            name="FetchWebPage",
            description=(
                "Fetch a single web page by `url` and return its main text. Use "
                "for non-API local-government pages (議事録, 請願・陳情, "
                "パブリックコメント, 予算・決算, etc.)."
            ),
            parameters=[
                {"name": "url", "type": "str", "description": "The page URL to fetch."},
            ],
        )
        self.max_chars = max_chars

    def use_tool(self, url: str = "") -> str:
        if not url:
            return "Provide a `url`."
        try:
            html = self._get(url)
        except requests.HTTPError as e:
            return f"Fetch HTTP error: {e}"
        except requests.RequestException as e:
            return f"Fetch error: {e}"
        return self._to_text(html)[: self.max_chars]

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=4)
    def _get(self, url: str) -> str:
        rsp = requests.get(url, headers={"User-Agent": _UA}, timeout=30)
        rsp.raise_for_status()
        rsp.encoding = rsp.apparent_encoding or rsp.encoding
        return rsp.text

    @staticmethod
    def _to_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator="\n")
        except ImportError:
            text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html,
                          flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
        lines = [ln.strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln)
