"""Multi-provider LLM/VLM client with conversation history and usage tracking.

Sources (three-way merge)
-------------------------
- AI Scientist v1 `ai_scientist/llm.py`: provider routing in create_client and
  the (content, new_msg_history) threading of get_response_from_llm.
- AI Scientist v2 `ai_scientist/vlm.py`: image -> base64 encoding and the
  image-block message construction (vision support).
- AI Scientist v2 `ai_scientist/utils/token_tracker.py`: per-model token
  accounting (counts only; see "Changed / dropped").

Reused
------
- create_client: pick an SDK client + model string from the model name.
- get_response_from_llm: one turn that threads msg_history and returns
  (content, new_msg_history).
- encode_image_to_base64: PIL normalize -> JPEG -> base64.

Added (in none of the sources individually)
--------------------------------------------
- Unified vision across BOTH providers. v1 had no images; v2's vlm.py built
  OpenAI `image_url` blocks only. Here Anthropic receives base64 `image` source
  blocks and OpenAI receives `image_url` blocks, from the same `image_paths`
  argument, so any role can be vision-capable regardless of provider.

Changed / dropped
-----------------
- Pricing tables removed. Per-token prices go stale and are deployment-specific;
  the tracker records token COUNTS only. Cost, if needed, is computed upstream
  from rates supplied via config. (token_tracker.py hardcoded 2024 USD prices.)
- tiktoken removed: counts come from the API response usage, not local
  re-tokenization.
- Ensemble/batch helper (get_batch_responses_from_llm) dropped: unused here.
- AVAILABLE_LLMS allow-list dropped: routing is by substring, so model strings
  are chosen in config (a per-(stage, role) hyperparameter) rather than gated by
  a list that ages out.
- JSON extraction (extract_json_between_markers) lives in response.py now.

Provider routing
----------------
"claude" in model -> Anthropic. Everything else -> an OpenAI-compatible client
(OpenAI by default, or another endpoint via base_url in create_client). o1/o3
reasoning models take the documented system-as-user / no-temperature path.
"""

from __future__ import annotations

import base64
import io
import os

import anthropic
import backoff
import openai
from PIL import Image

DEFAULT_MAX_TOKENS = 4096

_RETRY_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
)


# --- image encoding ---------------------------------------------------------

def encode_image_to_base64(image_path: str) -> str:
    """Load an image, normalize to RGB JPEG, return base64 (no data: prefix)."""
    with Image.open(image_path) as img:
        rgb = img.convert("RGB")
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# --- token tracking ---------------------------------------------------------

class TokenTracker:
    """Accumulate per-model input/output token counts across the run.

    Counts only, no pricing (stale-prone; rates are supplied upstream). Reads
    either the Anthropic (input_tokens/output_tokens) or OpenAI
    (prompt_tokens/completion_tokens) usage shape off the raw response.
    """

    def __init__(self):
        self.counts: dict[str, dict[str, int]] = {}

    def record(self, model: str, response) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        in_tok = getattr(usage, "input_tokens", None)
        out_tok = getattr(usage, "output_tokens", None)
        if in_tok is None:  # OpenAI-shaped usage
            in_tok = getattr(usage, "prompt_tokens", 0)
            out_tok = getattr(usage, "completion_tokens", 0)
        slot = self.counts.setdefault(model, {"input": 0, "output": 0, "calls": 0})
        slot["input"] += int(in_tok or 0)
        slot["output"] += int(out_tok or 0)
        slot["calls"] += 1

    def summary(self) -> dict:
        return {m: dict(v) for m, v in self.counts.items()}


tracker = TokenTracker()


# --- client creation --------------------------------------------------------

def create_client(model: str):
    """Return (client, model_string) for `model`.

    Extend with additional `elif` branches (base_url overrides) for other
    OpenAI-compatible providers as needed.
    """
    if "claude" in model:
        return anthropic.Anthropic(), model
    if model.startswith("deepseek"):
        return (
            openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            ),
            model,
        )
    return openai.OpenAI(), model  # OpenAI / OpenAI-compatible default


_CLIENT_CACHE: dict[str, tuple] = {}


def client_for(model: str):
    """create_client with per-model caching (one SDK client per model)."""
    if model not in _CLIENT_CACHE:
        _CLIENT_CACHE[model] = create_client(model)
    return _CLIENT_CACHE[model]


# --- single-turn call -------------------------------------------------------

@backoff.on_exception(backoff.expo, _RETRY_EXCEPTIONS)
def get_response_from_llm(
    msg: str,
    client,
    model: str,
    system_message: str,
    *,
    image_paths: list[str] | None = None,
    msg_history: list[dict] | None = None,
    temperature: float = 0.75,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str | None = None,
) -> tuple[str, list[dict]]:
    """One turn against `model`, threading `msg_history`.

    Returns (content, new_msg_history). `image_paths`, if given, attaches images
    to THIS user turn in the calling provider's native block format. The returned
    history is in that provider's format and should be passed back unchanged for
    the next turn of the same conversation.
    """
    if msg_history is None:
        msg_history = []
    image_paths = image_paths or []

    if "claude" in model:
        content_blocks: list[dict] = [{"type": "text", "text": msg}]
        for p in image_paths:
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": encode_image_to_base64(p),
                    },
                }
            )
        new_msg_history = msg_history + [{"role": "user", "content": content_blocks}]



        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = "".join(b.text for b in response.content if b.type == "text")
        new_msg_history = new_msg_history + [
            {"role": "assistant", "content": [{"type": "text", "text": content}]}
        ]

    else:  # OpenAI-compatible
        if image_paths:
            user_content: list | str = [{"type": "text", "text": msg}]
            for p in image_paths:
                b64 = encode_image_to_base64(p)
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    }
                )
        else:
            user_content = msg
        new_msg_history = msg_history + [{"role": "user", "content": user_content}]

        if model.startswith(("gpt-5", "o1", "o3", "o4")):
            # Newer/reasoning models: use max_completion_tokens; no temperature.
            # reasoning_effort is a reasoning-only knob; pass it only when set so
            # non-reasoning models never receive an unsupported parameter.
            messages = [{"role": "user", "content": system_message}, *new_msg_history]
            kwargs = dict(
                model=model, messages=messages,
                max_completion_tokens=max_tokens, n=1, seed=0,
            )
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**kwargs)
        else:
            messages = [{"role": "system", "content": system_message}, *new_msg_history]
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                seed=0,
            )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]

    tracker.record(model, response)
    return content, new_msg_history


# --- native web search (provider server-side tool) --------------------------

def web_search(
    query: str,
    model: str,
    *,
    max_results: int = 5,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> tuple[bool, str, list[dict]]:
    """Run one web search via the calling provider's server-side search tool.

    Routes by model name (same convention as get_response_from_llm): a "claude"
    model uses Anthropic's web_search_20250305 tool; anything else uses an
    OpenAI chat-completions search model (web_search_options). Returns
    (ok, text, sources) where sources is a list of {title, url} dicts. ok=False
    means the search itself failed (provider error / no usable result), which the
    research loop treats as a process failure, not evidence.

    The executor model is configured independently (see config); for OpenAI it
    must be a search-capable model string (e.g. gpt-4o-mini-search-preview).
    """
    client, model = client_for(model)
    if "claude" in model:
        return _web_search_anthropic(client, model, query, max_results, max_tokens)
    return _web_search_openai(client, model, query, max_tokens)


def _web_search_anthropic(client, model, query, max_results, max_tokens):
    instruction = (
        f"Search the web for: {query}\n\n"
        "Report what you find as a concise list of the most relevant real pages, "
        "each with its title, URL, and a one-line summary of what it contains. "
        "Prefer official Japanese government / municipal sources."
    )
    try:
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": instruction}],
            tools=[{"type": "web_search_20250305", "name": "web_search",
                    "max_uses": max_results}],
        )
    except Exception as e:                       # network / API failure
        return False, f"Web search failed: {e}", []

    tracker.record(model, response)
    text_parts: list[str] = []
    sources: list[dict] = []
    searched = False
    for block in response.content:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(block.text)
        elif btype == "web_search_tool_result":
            searched = True
            inner = getattr(block, "content", None)
            # Error result is a single object with an error_code.
            if isinstance(inner, dict) or getattr(inner, "type", "") == \
                    "web_search_tool_result_error":
                code = getattr(inner, "error_code", None) or (
                    inner.get("error_code") if isinstance(inner, dict) else None)
                return False, f"Web search error: {code or 'unknown'}", []
            for item in (inner or []):
                url = getattr(item, "url", None)
                title = getattr(item, "title", None)
                if url:
                    sources.append({"title": title or "", "url": url})

    if not searched and not sources:
        return False, "Web search returned no results.", []
    text = "\n".join(t for t in text_parts if t).strip()
    listing = "\n".join(f"- {s['title']}: {s['url']}" for s in sources)
    body = (text + ("\n\nSources:\n" + listing if listing else "")).strip()
    return True, body or "Web search returned no usable text.", sources


def _web_search_openai(client, model, query, max_tokens):
    instruction = (
        f"Search the web for: {query}\n\n"
        "Report the most relevant real pages with title, URL, and a one-line "
        "summary each. Prefer official Japanese government / municipal sources."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": instruction}],
            web_search_options={},
            max_tokens=max_tokens,
        )
    except Exception as e:
        return False, f"Web search failed: {e}", []

    tracker.record(model, response)
    msg = response.choices[0].message
    text = msg.content or ""
    sources: list[dict] = []
    for ann in (getattr(msg, "annotations", None) or []):
        if getattr(ann, "type", None) == "url_citation":
            uc = getattr(ann, "url_citation", None)
            url = getattr(uc, "url", None)
            title = getattr(uc, "title", None)
            if url:
                sources.append({"title": title or "", "url": url})
    if not text and not sources:
        return False, "Web search returned no results.", []
    listing = "\n".join(f"- {s['title']}: {s['url']}" for s in sources)
    body = (text + ("\n\nSources:\n" + listing if listing else "")).strip()
    return True, body, sources
