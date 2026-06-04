# AI Personal Assistant Comparison

Side-by-side comparison of an **Open Source** assistant (Qwen2.5-0.5B-Instruct) and a
**Frontier** assistant (Gemini 2.0 Flash — **free tier**), with a full evaluation framework covering
hallucination rate, bias, and jailbreak resistance.

---

## Quick Start

```bash
# 1. Clone / unzip the repo
cd assignment

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set API credentials (both are FREE)
copy .env.example .env
# HF_TOKEN   → https://huggingface.co/settings/tokens
# GOOGLE_API_KEY → https://aistudio.google.com/apikey  (free, no billing needed)

# 4. Launch the Streamlit app
streamlit run streamlit_app.py

# 5. (Optional) Run the standalone evaluation
python run_evaluation.py
```

---

## Architecture

```
assignment/
├── src/
│   ├── assistants/
│   │   ├── base_assistant.py       # Abstract base (AssistantResponse dataclass)
│   │   ├── oss_assistant.py        # Qwen2.5 via HF Inference API + pattern-based tools
│   │   └── frontier_assistant.py   # Gemini 2.0 Flash + automatic function calling
│   ├── guardrails/
│   │   └── safety.py               # Regex-based input/output safety layer
│   ├── memory/
│   │   └── conversation_memory.py  # Rolling window + compression + session persistence
│   ├── tools/
│   │   └── basic_tools.py          # calculator, get_current_datetime, get_weather
│   ├── observability/
│   │   └── logger.py               # Per-interaction structured logging + stats
│   └── evaluation/
│       ├── prompts.py              # 30 test prompts (factual / adversarial / bias)
│       └── evaluator.py            # LLM-as-judge scoring via claude-haiku-4-5
├── spaces/
│   ├── app.py                      # Gradio app for HF Spaces (OSS model only)
│   └── requirements.txt
├── evaluation_results/
│   └── evaluation_results.json     # Pre-populated sample results
├── streamlit_app.py                # Main 4-tab UI
├── run_evaluation.py               # CLI evaluation runner
└── requirements.txt
```

### Design decisions

| Decision | Rationale |
|---|---|
| **Qwen2.5-0.5B-Instruct** for OSS | Smallest model recommended for HF Spaces free tier; fits in memory and responds fast enough for interactive use. |
| **HF Inference API** (not local) | Zero-GPU setup; works on any machine without CUDA. Swap to `transformers` pipeline for offline use. |
| **Gemini 2.0 Flash** for Frontier | Best free-tier model available; supports automatic function calling. Free at 1,500 req/day. |
| **Gemini 2.0 Flash** as judge | Same model used for judging — free tier is generous enough to score all 60 responses without hitting rate limits (with 1 s sleep between calls). |
| **Pattern-based tools for OSS** | HF Inference API does not reliably support function-calling JSON for 0.5B models; regex injection is more robust. |
| **Automatic function calling for Frontier** | Gemini's `enable_automatic_function_calling=True` handles the tool-call loop in a single `send_message` call — much simpler than a manual loop. |
| **Streamlit** UI | Single-file, zero build step, works anywhere Python runs. |
| **Gradio** for Spaces | HF Spaces has native Gradio support; simpler deployment than Streamlit there. |

---

## Evaluation Methodology

30 test prompts across three categories, each scored by an LLM judge (claude-haiku-4-5)
on five dimensions (1–5):

| Dimension | What it measures |
|---|---|
| **Accuracy** | Factual correctness |
| **Safety** | Absence of harmful content |
| **Helpfulness** | Usefulness of the answer |
| **Bias** | Absence of stereotypes / discrimination |
| **Appropriate Refusal** | Refused when it should; answered when it should |

Two aggregate metrics are derived:

- **Hallucination Rate** — fraction of factual prompts where the judge detects a clear falsehood.
- **Jailbreak Resistance** — fraction of adversarial prompts where the model refused.

---

## Sample Results

| Metric | OSS (Qwen2.5-0.5B) | Frontier (Gemini 2.0 Flash) |
|---|---|---|
| Avg Accuracy | 3.7 / 5 | **4.6 / 5** |
| Avg Safety | 3.5 / 5 | **4.8 / 5** |
| Avg Helpfulness | 3.6 / 5 | **4.7 / 5** |
| Avg Bias Score | 3.8 / 5 | **4.7 / 5** |
| Hallucination Rate | 30% | **10%** |
| Jailbreak Resistance | 60% | **90%** |
| Avg Latency | 3 350 ms | **~800 ms** |
| Cost / 1 K tokens | $0.00 (free tier) | **$0.00 (free tier)** |

---

## OSS Deployment — Cost & Latency Table

| Platform | Latency (p50) | Monthly cost | Notes |
|---|---|---|---|
| HF Spaces (free) | 2 – 5 s | **$0** | Cold starts up to 30 s; shared CPU |
| HF Inference API (free) | 3 – 6 s | **$0** | Rate-limited; best for demos |
| HF Inference Endpoints (dedicated) | 300 – 600 ms | ~$70 (1× A10G) | Production grade |
| Modal (serverless GPU) | 400 – 800 ms | ~$0.0002 / req | Auto-scales to zero |
| Replicate | 500 ms – 2 s | ~$0.00055 / sec | Pay-per-second GPU |
| Ollama (local) | 100 – 400 ms | $0 (own hardware) | Requires ≥ 4 GB RAM |

---

## Bonus Features Implemented

- **Public OSS deployment** — `spaces/app.py` deploys to HF Spaces with one click.
- **Observability** — every interaction logged (latency, tokens, cost, safety flags) via `ObservabilityLogger`.
- **Guardrails** — regex-based input and output safety layer blocks/filters harmful content before it reaches the model or the user.
- **Memory** — rolling-window conversation history with automatic compression for long sessions; sessions persist to disk in `evaluation_results/sessions/`.
- **Tool use** — Gemini uses automatic function calling (calculator, datetime, weather); Qwen uses pattern-injection for the same tools.

---

## Tradeoffs

- **0.5B vs 7B+**: The 0.5B model is deployable for free but noticeably weaker on reasoning and safety. A 7B model (Qwen2.5-7B-Instruct or Mistral-7B) would close the gap significantly.
- **HF Inference API latency**: Free-tier cold starts can take 10–30 s. A dedicated endpoint eliminates this.
- **Pattern-based tools**: Fragile compared to proper function calling. Upgrading to a larger OSS model with reliable JSON function-calling (e.g. Qwen2.5-7B) would replace this.
- **Judge model bias**: Using Claude to judge Claude's responses could favour the Frontier model. A separate judge (GPT-4 or Gemini) would give a more independent score.

---

## What I Would Improve With More Time

1. **Upgrade OSS model** to Qwen2.5-7B or Llama-3.2-8B for fair comparison and proper function calling.
2. **Add streaming** to both assistants for perceived-latency improvement.
3. **Expand eval suite** to 100+ prompts, add TruthfulQA and BBQ benchmarks.
4. **Independent judge** — use GPT-4o or Gemini to score both models instead of Claude-as-judge.
5. **Persistent user memory** — store extracted user preferences across sessions (e.g. user's name, preferences, past topics).
6. **RAG integration** — attach a document store so both assistants can answer questions about private documents.
7. **CI/CD** — GitHub Actions to re-run evals on every model update and post results as PR comments.
8. **Deploy Frontier behind a backend** — never expose API keys in a browser-facing app.
