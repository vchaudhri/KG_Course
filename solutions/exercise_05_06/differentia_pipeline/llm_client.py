"""
llm_client.py
-------------
The only file that imports google-genai. Every other component talks to
an LLMClient, not to Gemini directly -- swapping providers later means
writing one new subclass here, touching nothing else.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod


class LLMCallError(Exception):
    """Raised when an LLM call fails after all retries, or returns a
    response that can't be parsed as valid JSON."""


def strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text.strip())
    text = re.sub(r"```$", "", text.strip())
    return text.strip()


def parse_json_response(raw: str) -> dict | list:
    try:
        return json.loads(strip_fences(raw))
    except json.JSONDecodeError as e:
        raise LLMCallError(f"Response was not valid JSON: {e}\nRaw response: {raw[:500]!r}") from e


class LLMClient(ABC):
    model: str

    @abstractmethod
    def call(self, system_prompt: str, user_prompt: str, *, max_output_tokens: int = 1000) -> str:
        """Return the raw text response. Implementations are responsible for
        their own retry logic and should raise LLMCallError on final failure."""
        raise NotImplementedError


class GeminiLLMClient(LLMClient):
    def __init__(self, model: str = "gemini-2.5-flash", api_key: str | None = None,
                 max_retries: int = 3, temperature: float = 0.0):
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.temperature = temperature
        self._client = None  # lazily constructed so importing this module never requires google-genai

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key) if self.api_key else genai.Client()
        return self._client

    def call(self, system_prompt: str, user_prompt: str, *, max_output_tokens: int = 1000) -> str:
        from google.genai import types

        client = self._get_client()
        token_budget = max_output_tokens
        last_err: Exception | str | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = client.models.generate_content(
                    model=self.model,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=self.temperature,
                        max_output_tokens=token_budget,
                        response_mime_type="application/json",
                        # Gemini 2.5 models "think" by default, and those tokens eat
                        # into max_output_tokens, truncating the JSON before it's
                        # written. Not needed for this kind of extraction/mapping task.
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                candidates = getattr(resp, "candidates", None) or []
                finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
                finish_reason_name = getattr(finish_reason, "name", str(finish_reason))
                raw = resp.text or ""

                if finish_reason_name == "MAX_TOKENS" or not raw:
                    last_err = f"response truncated (MAX_TOKENS, budget was {token_budget})"
                    token_budget *= 2
                    time.sleep(min(2 ** attempt, 10))
                    continue

                return raw
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))

        raise LLMCallError(f"LLM call failed after {self.max_retries} attempts: {last_err}")
