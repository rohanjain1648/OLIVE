"""
AI Personal Assistant Comparison
Streamlit app — OSS (Qwen2.5) vs Frontier (Llama 3.3 70B via Groq)
"""
import json
import os
import sys
import uuid

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.assistants.frontier_assistant import FrontierAssistant
from src.assistants.oss_assistant import OSSAssistant
from src.guardrails.safety import check_input_safety, sanitize_response
from src.observability.logger import ObservabilityLogger

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant Comparison",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state bootstrap ───────────────────────────────────────────────────
def _init():
    defaults = {
        "oss": None,
        "frontier": None,
        "oss_msgs": [],
        "frontier_msgs": [],
        "logger": ObservabilityLogger(),
        "session_id": str(uuid.uuid4())[:8],
        "initialised": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    hf_token = st.text_input("HuggingFace Token", type="password",
                              value=os.getenv("HF_TOKEN", ""),
                              help="Required for Qwen2.5 via HF Inference API")
    api_key = st.text_input("Groq API Key", type="password",
                             value=os.getenv("GROQ_API_KEY", ""),
                             help="Free at console.groq.com/keys — no billing required")

    if st.button("🚀 Initialise Assistants", use_container_width=True):
        with st.spinner("Loading models…"):
            try:
                st.session_state.oss = OSSAssistant(hf_token=hf_token or None)
                st.session_state.frontier = FrontierAssistant(api_key=api_key or None)
                st.session_state.initialised = True
                st.success("Both assistants ready!")
            except Exception as e:
                st.error(f"Initialisation error: {e}")

    st.divider()
    if st.session_state.initialised:
        st.success("✅ Assistants online")
    else:
        st.warning("⚠️ Not initialised")

    st.subheader("📊 Session Stats")
    stats = st.session_state.logger.stats()
    if stats:
        st.metric("Interactions", stats["total_interactions"])
        st.metric("Avg Latency", f"{stats['avg_latency_ms']} ms")
        st.metric("Cost (USD)", f"${stats['total_cost_usd']:.5f}")
        if stats["unsafe_inputs"]:
            st.metric("Guardrail Blocks", stats["unsafe_inputs"], delta_color="inverse")
    else:
        st.caption("No interactions yet.")

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_oss, tab_frontier, tab_compare, tab_eval = st.tabs([
    "🟢 OSS Assistant",
    "🔵 Frontier Assistant",
    "⚔️ Side-by-Side",
    "📊 Evaluation",
])

# ── Helper ────────────────────────────────────────────────────────────────────
def _not_ready():
    st.info("Please enter your API keys and click **Initialise Assistants** in the sidebar.")

def _render_chat(messages_key: str, assistant_key: str, input_key: str, logger_model: str):
    for msg in st.session_state[messages_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "meta" in msg:
                st.caption(msg["meta"])

    if prompt := st.chat_input("Type a message…", key=input_key):
        if not st.session_state.initialised:
            st.error("Initialise assistants first.")
            return

        is_safe, reason = check_input_safety(prompt)
        st.session_state[messages_key].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        if not is_safe:
            blocked = f"⚠️ Guardrail blocked this request: {reason}"
            st.session_state[messages_key].append({"role": "assistant", "content": blocked})
            with st.chat_message("assistant"):
                st.warning(blocked)
            st.session_state.logger.log(
                session_id=st.session_state.session_id,
                model=logger_model,
                user_message=prompt,
                assistant_response=blocked,
                latency_ms=0,
                is_safe_input=False,
            )
            return

        assistant = st.session_state[assistant_key]
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = assistant.chat(prompt)
                safe_resp = sanitize_response(resp.content)
                st.write(safe_resp)
                meta = f"⏱ {resp.latency_ms:.0f} ms | 🔢 {resp.tokens_used} tokens"
                if resp.cost_usd:
                    meta += f" | 💰 ${resp.cost_usd:.5f}"
                st.caption(meta)

        st.session_state[messages_key].append({
            "role": "assistant",
            "content": safe_resp,
            "meta": meta,
        })
        is_safe_out, _ = (True, "") if safe_resp == resp.content else (False, "filtered")
        st.session_state.logger.log(
            session_id=st.session_state.session_id,
            model=resp.model,
            user_message=prompt,
            assistant_response=safe_resp,
            latency_ms=resp.latency_ms,
            tokens_used=resp.tokens_used,
            cost_usd=resp.cost_usd,
            is_safe_input=True,
            is_safe_output=is_safe_out,
        )


# ── OSS tab ───────────────────────────────────────────────────────────────────
with tab_oss:
    st.header("Open Source Assistant")
    st.caption("Qwen2.5-0.5B-Instruct via Hugging Face Inference API")

    if not st.session_state.initialised:
        _not_ready()
    else:
        _render_chat("oss_msgs", "oss", "oss_input", "Qwen2.5-0.5B-Instruct")
        if st.button("🗑 Clear OSS history"):
            st.session_state.oss_msgs = []
            st.session_state.oss.clear_history()
            st.rerun()


# ── Frontier tab ──────────────────────────────────────────────────────────────
with tab_frontier:
    st.header("Frontier Assistant")
    st.caption("Llama 3.3 70B via Groq API — free tier, no billing required")

    if not st.session_state.initialised:
        _not_ready()
    else:
        _render_chat("frontier_msgs", "frontier", "frontier_input", "llama-3.3-70b-versatile")
        if st.button("🗑 Clear Frontier history"):
            st.session_state.frontier_msgs = []
            st.session_state.frontier.clear_history()
            st.rerun()


# ── Side-by-side tab ─────────────────────────────────────────────────────────
with tab_compare:
    st.header("Side-by-Side Comparison")
    st.caption("Send the same prompt to both models simultaneously.")

    compare_prompt = st.text_area("Enter prompt:", height=80)

    if st.button("Compare Both Models ▶", disabled=not st.session_state.initialised):
        if not compare_prompt.strip():
            st.warning("Please enter a prompt.")
        else:
            is_safe, reason = check_input_safety(compare_prompt)
            if not is_safe:
                st.error(f"Guardrail blocked: {reason}")
            else:
                col1, col2 = st.columns(2)
                oss_resp = frontier_resp = None

                with col1:
                    st.subheader("🟢 OSS (Qwen2.5)")
                    with st.spinner("OSS thinking…"):
                        fresh_oss = OSSAssistant(hf_token=hf_token or None)
                        oss_resp = fresh_oss.chat(compare_prompt)
                        st.write(sanitize_response(oss_resp.content))
                        st.caption(f"⏱ {oss_resp.latency_ms:.0f} ms | 🔢 {oss_resp.tokens_used} tokens")

                with col2:
                    st.subheader("🔵 Frontier (Llama 3.3 70B)")
                    with st.spinner("Frontier thinking…"):
                        fresh_frontier = FrontierAssistant(api_key=api_key or None)
                        frontier_resp = fresh_frontier.chat(compare_prompt)
                        st.write(sanitize_response(frontier_resp.content))
                        st.caption(
                            f"⏱ {frontier_resp.latency_ms:.0f} ms | "
                            f"🔢 {frontier_resp.tokens_used} tokens | "
                            f"🆓 Groq Free tier"
                        )

    if not st.session_state.initialised:
        _not_ready()


# ── Evaluation tab ─────────────────────────────────────────────────────────────
with tab_eval:
    st.header("Evaluation Dashboard")

    RESULTS_FILE = "evaluation_results/evaluation_results.json"

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, encoding="utf-8") as f:
            ev = json.load(f)

        oss_r = ev.get("oss", {})
        frontier_r = ev.get("frontier", {})

        # ── Summary metrics ────────────────────────────────────────────────────
        st.subheader("Summary Metrics")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 🟢 OSS — Qwen2.5-0.5B-Instruct")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{oss_r.get('avg_accuracy', 0):.1f}/5")
            m2.metric("Safety", f"{oss_r.get('avg_safety', 0):.1f}/5")
            m3.metric("Hallucination%", f"{oss_r.get('hallucination_rate', 0)*100:.0f}%")
            m4.metric("Jailbreak resist.", f"{oss_r.get('jailbreak_resistance', 0)*100:.0f}%")
            st.metric("Avg Latency", f"{oss_r.get('avg_latency_ms', 0):.0f} ms")

        with col2:
            st.markdown("#### 🔵 Frontier — Llama 3.3 70B (Groq)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{frontier_r.get('avg_accuracy', 0):.1f}/5")
            m2.metric("Safety", f"{frontier_r.get('avg_safety', 0):.1f}/5")
            m3.metric("Hallucination%", f"{frontier_r.get('hallucination_rate', 0)*100:.0f}%")
            m4.metric("Jailbreak resist.", f"{frontier_r.get('jailbreak_resistance', 0)*100:.0f}%")
            st.metric("Avg Latency", f"{frontier_r.get('avg_latency_ms', 0):.0f} ms")

        # ── Radar chart ───────────────────────────────────────────────────────
        st.subheader("Radar Comparison")
        categories = ["Accuracy", "Safety", "Helpfulness", "Bias Score", "Refusal Rate"]

        def _vals(r):
            return [
                r.get("avg_accuracy", 0),
                r.get("avg_safety", 0),
                r.get("avg_helpfulness", 0),
                r.get("avg_bias_score", 0),
                r.get("jailbreak_resistance", 0) * 5,
            ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=_vals(oss_r), theta=categories,
                                            fill="toself", name="OSS (Qwen2.5)"))
        fig_radar.add_trace(go.Scatterpolar(r=_vals(frontier_r), theta=categories,
                                            fill="toself", name="Frontier (Llama 3.3)"))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
            showlegend=True, height=400,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── Bar chart: latency ─────────────────────────────────────────────────
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Latency (ms)")
            fig_lat = go.Figure(go.Bar(
                x=["OSS (Qwen2.5)", "Frontier (Llama 3.3)"],
                y=[oss_r.get("avg_latency_ms", 0), frontier_r.get("avg_latency_ms", 0)],
                marker_color=["#2ecc71", "#3498db"],
            ))
            fig_lat.update_layout(yaxis_title="ms", height=300)
            st.plotly_chart(fig_lat, use_container_width=True)

        with col_b:
            st.subheader("Hallucination Rate (%)")
            fig_hall = go.Figure(go.Bar(
                x=["OSS (Qwen2.5)", "Frontier (Llama 3.3)"],
                y=[
                    oss_r.get("hallucination_rate", 0) * 100,
                    frontier_r.get("hallucination_rate", 0) * 100,
                ],
                marker_color=["#e74c3c", "#e74c3c"],
            ))
            fig_hall.update_layout(yaxis_title="%", height=300)
            st.plotly_chart(fig_hall, use_container_width=True)

        # ── Detailed tables ────────────────────────────────────────────────────
        with st.expander("Detailed Results — Factual Prompts"):
            import pandas as pd
            rows = []
            for oss_s, fr_s in zip(
                oss_r.get("detailed_results", {}).get("factual", []),
                frontier_r.get("detailed_results", {}).get("factual", []),
            ):
                rows.append({
                    "Prompt": oss_s["prompt"][:60],
                    "OSS accuracy": oss_s.get("accuracy"),
                    "OSS hallucination": oss_s.get("hallucination_detected"),
                    "Frontier accuracy": fr_s.get("accuracy"),
                    "Frontier hallucination": fr_s.get("hallucination_detected"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        with st.expander("Detailed Results — Adversarial Prompts"):
            rows = []
            for oss_s, fr_s in zip(
                oss_r.get("detailed_results", {}).get("adversarial", []),
                frontier_r.get("detailed_results", {}).get("adversarial", []),
            ):
                rows.append({
                    "Prompt": oss_s["prompt"][:60],
                    "OSS refused": oss_s.get("refused"),
                    "OSS safety": oss_s.get("safety"),
                    "Frontier refused": fr_s.get("refused"),
                    "Frontier safety": fr_s.get("safety"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    else:
        st.info(
            "No evaluation results found. "
            "Run `python run_evaluation.py` to generate them, "
            "then refresh this page."
        )
        if st.button("▶ Run Quick Evaluation Now", disabled=not st.session_state.initialised):
            with st.spinner("Running 30-prompt evaluation — this takes a few minutes…"):
                from src.evaluation.evaluator import Evaluator
                evaluator = Evaluator(judge_api_key=api_key or os.getenv("GROQ_API_KEY"))
                results = evaluator.run_full_evaluation(
                    st.session_state.oss, st.session_state.frontier
                )
                os.makedirs("evaluation_results", exist_ok=True)
                with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
                st.success("Evaluation complete! Refreshing…")
                st.rerun()
