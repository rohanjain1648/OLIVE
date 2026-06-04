---
title: Qwen2.5 Personal Assistant
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "4.31.0"
app_file: app.py
pinned: false
license: apache-2.0
---

# Qwen2.5-0.5B-Instruct Personal Assistant

Open-source AI personal assistant deployed on Hugging Face Spaces.

## Features
- Multi-turn conversation with short-term memory
- Basic tool use: calculator, date/time, weather
- Input + output safety guardrails

## Setup (local)
```bash
pip install -r requirements.txt
HF_TOKEN=hf_... python app.py
```

## Secrets
Add `HF_TOKEN` as a Space secret for authenticated Inference API access.
