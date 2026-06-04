"""
Hugging Face Spaces deployment — Qwen2.5-0.5B-Instruct personal assistant.
Deploy this folder as a Gradio Space:
  1. Create a new Space at https://huggingface.co/spaces
  2. Select Gradio SDK
  3. Upload this file + requirements.txt
  4. Add HF_TOKEN as a Space secret (optional but recommended)
"""
import os
import sys

import gradio as gr

# When running inside Spaces the parent src/ dir is not on the path by default
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.assistants.oss_assistant import OSSAssistant
from src.guardrails.safety import check_input_safety, sanitize_response
from src.observability.logger import ObservabilityLogger

# ── Globals ───────────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")
assistant = OSSAssistant(hf_token=HF_TOKEN)
logger = ObservabilityLogger(log_dir="/tmp/oss_logs")

# ── Chat function ─────────────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    """
    Gradio ChatInterface callback.
    history is a list of [user, assistant] string pairs.
    """
    is_safe, reason = check_input_safety(message)
    if not is_safe:
        return f"⚠️ Safety guardrail triggered: {reason}"

    # Rebuild assistant state from Gradio history
    assistant.clear_history()
    for human, ai in history:
        assistant.add_message("user", human)
        assistant.add_message("assistant", ai)

    resp = assistant.chat(message)
    safe_resp = sanitize_response(resp.content)

    logger.log(
        session_id="spaces",
        model=resp.model,
        user_message=message,
        assistant_response=safe_resp,
        latency_ms=resp.latency_ms,
        tokens_used=resp.tokens_used,
        is_safe_input=True,
        is_safe_output=(safe_resp == resp.content),
    )

    latency = f"\n\n---\n_⏱ {resp.latency_ms:.0f} ms | 🔢 {resp.tokens_used} tokens_"
    return safe_resp + latency


# ── Gradio UI ─────────────────────────────────────────────────────────────────
demo = gr.ChatInterface(
    fn=chat,
    title="🤖 Qwen2.5 Personal Assistant",
    description=(
        "**Open Source AI Assistant** powered by "
        "[Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) "
        "via the Hugging Face Inference API.\n\n"
        "Supports multi-turn conversation, basic tool detection (calculator, date/time, weather), "
        "and input/output safety guardrails."
    ),
    examples=[
        "What is the capital of France?",
        "Calculate sqrt(144) + 7",
        "What day of the week is it today?",
        "Explain quantum entanglement in simple terms",
        "Help me write a short professional email declining a meeting",
    ],
)

if __name__ == "__main__":
    demo.launch(share=False)
