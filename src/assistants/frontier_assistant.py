import os
import time
from typing import Optional

from google import genai
from google.genai import types

from .base_assistant import BaseAssistant, AssistantResponse


class FrontierAssistant(BaseAssistant):
    """Gemini 2.0 Flash via Google AI free tier with automatic tool use."""

    MODEL_ID = "gemini-2.0-flash"

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        use_tools: bool = True,
    ):
        super().__init__(system_prompt)
        self.client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        self.use_tools = use_tools

        self._tools: list = []
        if use_tools:
            from ..tools.basic_tools import calculator, get_current_datetime, get_weather
            self._tools = [calculator, get_current_datetime, get_weather]

    # ── Public API ─────────────────────────────────────────────────────────────

    def chat(self, user_message: str) -> AssistantResponse:
        # Gemini uses role "model" instead of "assistant"
        history = [
            types.Content(
                role="model" if m.role == "assistant" else m.role,
                parts=[types.Part(text=m.content)],
            )
            for m in self.conversation_history
        ]

        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt,
            tools=self._tools if self.use_tools and self._tools else None,
        )

        chat_session = self.client.chats.create(
            model=self.MODEL_ID,
            config=config,
            history=history,
        )

        start = time.time()
        try:
            response = chat_session.send_message(user_message)
            latency_ms = (time.time() - start) * 1000
            final_text = response.text or ""

            tokens_used = 0
            if getattr(response, "usage_metadata", None):
                tokens_used = (
                    (response.usage_metadata.prompt_token_count or 0)
                    + (response.usage_metadata.candidates_token_count or 0)
                )

        except Exception as exc:
            latency_ms = (time.time() - start) * 1000
            final_text = f"[Gemini error: {exc}]"
            tokens_used = 0

        self.add_message("user", user_message)
        self.add_message("assistant", final_text)

        return AssistantResponse(
            content=final_text,
            model=self.MODEL_ID,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=0.0,  # free tier
            metadata={"tokens_used": tokens_used},
        )

    def get_model_name(self) -> str:
        return "Gemini 2.0 Flash (Frontier, Free Tier)"
