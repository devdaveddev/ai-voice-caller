"""LLM wrapper (Anthropic, streaming). Swap the class to change provider."""
import json
import logging
import re
from typing import AsyncIterator

from anthropic import AsyncAnthropic

from .config import env

log = logging.getLogger("llm")

OUTCOME_PROMPT = """You analyse a finished phone call transcript between an AI agent and a customer.
Return ONLY a JSON object with keys:
"committed" (true/false: customer gave a firm commitment to pay),
"committed_date" (string or null, as spoken, e.g. "5 Nov"),
"committed_amount" (string or null),
"refused" (true/false),
"reason" (short string or null: customer's stated reason/objection),
"summary" (one short sentence)."""


def _merge_roles(messages: list[dict]) -> list[dict]:
    """Anthropic wants alternating roles; merge consecutive same-role turns (happens after barge-in)."""
    out: list[dict] = []
    for m in messages:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + " " + m["content"]}
        else:
            out.append(dict(m))
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(call connected)"})
    return out


class AnthropicLLM:
    def __init__(self):
        self.client = AsyncAnthropic(api_key=env("ANTHROPIC_API_KEY"))
        self.model = env("ANTHROPIC_MODEL", "claude-sonnet-5-5")

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        async with self.client.messages.stream(
            model=self.model, max_tokens=200, system=system, messages=_merge_roles(messages)
        ) as s:
            async for text in s.text_stream:
                yield text

    async def extract_outcome(self, transcript: str) -> dict:
        r = await self.client.messages.create(
            model=self.model, max_tokens=300, system=OUTCOME_PROMPT,
            messages=[{"role": "user", "content": transcript or "(empty call)"}],
        )
        text = "".join(b.text for b in r.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.S)
        return json.loads(m.group(0)) if m else {"error": "unparseable", "raw": text}


class MockLLM:
    """Offline stand-in (LLM_PROVIDER=mock) for plumbing tests. Not a real conversation."""

    def __init__(self):
        self.turn = 0

    async def stream(self, system: str, messages: list[dict]) -> AsyncIterator[str]:
        self.turn += 1
        if self.turn < 3:
            parts = ["जी ठीक है। ", "आप ईएमआई कब तक भर पाएँगे?"]
        else:
            parts = ["ठीक है, मैंने 5 तारीख को 4,200 रुपये नोट कर लिए हैं। ", "धन्यवाद, नमस्ते। END_", "CALL"]
        for p in parts:
            yield p

    async def extract_outcome(self, transcript: str) -> dict:
        return {"committed": self.turn >= 3, "committed_date": "5th",
                "committed_amount": "4,200", "refused": False, "reason": None, "summary": "mock"}


def make_llm():
    return MockLLM() if env("LLM_PROVIDER") == "mock" else AnthropicLLM()
