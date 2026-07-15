"""Thin wrappers around the Anthropic SDK.

Three call shapes cover the whole compiler:
- structured(): one-shot call constrained to a JSON schema (fast passes)
- agent_loop(): manual tool-use loop (parser-writer, wiki, query agents)
- stream_chat(): streaming turn for the interactive interview
"""

from __future__ import annotations

import json
from typing import Any, Callable

import anthropic

from .config import FRONTIER_MODEL

_client: anthropic.Anthropic | None = None
_async_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def async_client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic()
    return _async_client


def _first_text(content: list[Any]) -> str:
    return next((b.text for b in content if b.type == "text"), "")


def structured(
    model: str,
    system: str,
    prompt: str | list[dict],
    schema: dict,
    max_tokens: int = 8192,
) -> dict:
    """One-shot structured extraction. Returns parsed JSON matching schema."""
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    response = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return json.loads(_first_text(response.content))


async def structured_async(
    model: str,
    system: str,
    prompt: str | list[dict],
    schema: dict,
    max_tokens: int = 8192,
) -> dict:
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    response = await async_client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    return json.loads(_first_text(response.content))


def agent_loop(
    system: str,
    user_message: str,
    tools: list[dict],
    tool_impls: dict[str, Callable[..., str]],
    model: str = FRONTIER_MODEL,
    max_iterations: int = 40,
    max_tokens: int = 16000,
    verbose: bool = False,
) -> str:
    """Manual agentic loop: run tools until the model stops calling them.

    tool_impls maps tool name -> callable taking the tool input as kwargs
    and returning a string result.
    """
    messages: list[dict] = [{"role": "user", "content": user_message}]

    for _ in range(max_iterations):
        response = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            thinking={"type": "adaptive"},
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "pause_turn":
            messages.append({"role": "assistant", "content": response.content})
            continue

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            return _first_text(response.content)

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for tu in tool_uses:
            if verbose:
                preview = json.dumps(tu.input)[:120]
                print(f"    [{tu.name}] {preview}")
            impl = tool_impls.get(tu.name)
            if impl is None:
                out, is_error = f"Unknown tool: {tu.name}", True
            else:
                try:
                    out, is_error = str(impl(**tu.input)), False
                except Exception as e:
                    out, is_error = f"Tool error: {e}", True
            results.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": out, "is_error": is_error}
            )
        messages.append({"role": "user", "content": results})

    return "Agent stopped: hit max iterations."


def stream_chat(
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str = FRONTIER_MODEL,
    max_tokens: int = 16000,
):
    """One streaming turn. Prints text as it arrives, returns the final message."""
    kwargs: dict = dict(
        model=model,
        max_tokens=max_tokens,
        system=system,
        thinking={"type": "adaptive"},
        messages=messages,
    )
    if tools:
        kwargs["tools"] = tools
    with client().messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        return stream.get_final_message()
