import os
import re
import time
from typing import Optional

from huggingface_hub import InferenceClient

from .base_assistant import BaseAssistant, AssistantResponse
from ..tools.basic_tools import get_current_datetime, calculator, get_weather


class OSSAssistant(BaseAssistant):
    """Qwen2.5-0.5B-Instruct via Hugging Face Inference API."""

    MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self, system_prompt: Optional[str] = None, hf_token: Optional[str] = None):
        super().__init__(system_prompt)
        token = hf_token or os.getenv("HF_TOKEN")
        self.client = InferenceClient(model=self.MODEL_ID, token=token)

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> AssistantResponse:
        tool_context = self._detect_tools(user_message)
        augmented = f"{tool_context}\n\nUser query: {user_message}" if tool_context else user_message

        # Build message list: system + history (without the new user turn) + augmented turn
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.get_history_as_dicts())
        messages.append({"role": "user", "content": augmented})

        start = time.time()
        try:
            resp = self.client.chat_completion(
                messages=messages,
                max_tokens=512,
                temperature=0.7,
            )
            latency_ms = (time.time() - start) * 1000
            content = resp.choices[0].message.content or ""
            tokens = resp.usage.total_tokens if resp.usage else 0

        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            content = f"[OSS model error: {exc}]"
            tokens = 0

        # Persist to history using the original (not augmented) user message
        self.add_message("user", user_message)
        self.add_message("assistant", content)

        return AssistantResponse(
            content=content,
            model=self.MODEL_ID,
            latency_ms=latency_ms,
            tokens_used=tokens,
            cost_usd=0.0,
        )

    def get_model_name(self) -> str:
        return "Qwen2.5-0.5B-Instruct (Open Source)"

    # ── Simple pattern-based tool injection ────────────────────────────────────

    def _detect_tools(self, message: str) -> str:
        parts = []

        # Current date/time
        if re.search(r"\b(what|current|today|now)\b.{0,20}\b(time|date|day)\b", message, re.I):
            result = get_current_datetime()
            parts.append(f"[Tool: get_current_datetime] → {result}")

        # Calculator
        calc_m = re.search(
            r"(?:calculate|compute|what\s+is|evaluate)\s+([\d\s+\-*/().^%a-zA-Z]+)",
            message, re.I,
        )
        if calc_m:
            result = calculator(calc_m.group(1).strip())
            if result.get("success"):
                parts.append(f"[Tool: calculator] {calc_m.group(1).strip()} = {result['result']}")

        # Weather
        weather_m = re.search(r"weather\s+(?:in\s+)?([A-Z][a-zA-Z\s]{2,30})", message)
        if weather_m:
            result = get_weather(weather_m.group(1).strip())
            parts.append(f"[Tool: get_weather] {result}")

        return "\n".join(parts)
