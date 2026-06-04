"""
Standalone evaluation runner.
Usage:
    python run_evaluation.py
Environment variables required:
    GOOGLE_API_KEY  — free at https://aistudio.google.com/apikey
    HF_TOKEN        — for Qwen2.5 (optional on free tier, but recommended)
"""
import json
import os
import sys


def main():
    print("=" * 60)
    print("  AI Assistant Evaluation Runner")
    print("=" * 60)

    google_key = os.getenv("GOOGLE_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    if not google_key:
        print("\nERROR: GOOGLE_API_KEY environment variable not set.")
        print("Get a free key at:  https://aistudio.google.com/apikey")
        print("Then run:  set GOOGLE_API_KEY=AIza...")
        sys.exit(1)

    print("\nInitialising assistants…")
    from src.assistants.oss_assistant import OSSAssistant
    from src.assistants.frontier_assistant import FrontierAssistant
    from src.evaluation.evaluator import Evaluator

    oss = OSSAssistant(hf_token=hf_token)
    frontier = FrontierAssistant(api_key=google_key)
    evaluator = Evaluator(judge_api_key=google_key)

    print("Running evaluation across 30 prompts (10 factual / 10 adversarial / 10 bias)…")
    print("Each prompt is tested on both models then scored by Gemini 2.0 Flash.\n")

    results = evaluator.run_full_evaluation(oss, frontier)

    os.makedirs("evaluation_results", exist_ok=True)
    out_path = "evaluation_results/evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved → {out_path}\n")
    _print_summary(results)


def _print_summary(results: dict):
    print("=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    for model_key in ("oss", "frontier"):
        label = "OSS (Qwen2.5-0.5B-Instruct)" if model_key == "oss" else "Frontier (Gemini 2.0 Flash)"
        r = results[model_key]
        print(f"\n{label}")
        print(f"  Avg Accuracy        : {r['avg_accuracy']:.2f} / 5")
        print(f"  Avg Safety          : {r['avg_safety']:.2f} / 5")
        print(f"  Avg Helpfulness     : {r['avg_helpfulness']:.2f} / 5")
        print(f"  Avg Bias Score      : {r['avg_bias_score']:.2f} / 5")
        print(f"  Hallucination Rate  : {r['hallucination_rate']*100:.0f}%")
        print(f"  Jailbreak Resistance: {r['jailbreak_resistance']*100:.0f}%")
        print(f"  Avg Latency         : {r['avg_latency_ms']:.0f} ms")
    print("\nDone. Open the Streamlit app to see charts:  streamlit run streamlit_app.py")


if __name__ == "__main__":
    main()
