import os
import time
from typing import Optional

from groq import Groq

from .base_assistant import BaseAssistant, AssistantResponse


class FrontierAssistant(BaseAssistant):
    """Llama 3.3 70B via Groq API (free tier, no billing required)."""

    MODEL_ID = "llama-3.3-70b-versatile"

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        use_tools: bool = True,
    ):
        super().__init__(system_prompt)
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.use_tools = use_tools

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> AssistantResponse:
        messages = [{"role": "system", "content": self.system_prompt}]
        for m in self.conversation_history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": user_message})

        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.MODEL_ID,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
            )
            latency_ms = (time.time() - start) * 1000
            content = response.choices[0].message.content or ""
            tokens = response.usage.total_tokens if response.usage else 0

        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            content = f"[Groq error: {exc}]"
            tokens = 0

        self.add_message("user", user_message)
        self.add_message("assistant", content)

        return AssistantResponse(
            content=content,
            model=self.MODEL_ID,
            latency_ms=latency_ms,
            tokens_used=tokens,
            cost_usd=0.0,  # free tier
        )

    def get_model_name(self) -> str:
        return "Llama 3.3 70B (Frontier via Groq, Free Tier)"
