import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class InteractionLog:
    timestamp: str
    session_id: str
    model: str
    user_message: str
    assistant_response: str
    latency_ms: float
    tokens_used: int
    cost_usd: float
    is_safe_input: bool
    is_safe_output: bool
    extra: Dict[str, Any] = field(default_factory=dict)


class ObservabilityLogger:
    def __init__(self, log_dir: str = "evaluation_results"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self._logs: List[InteractionLog] = []

    # ── Logging ────────────────────────────────────────────────────────────────

    def log(
        self,
        *,
        session_id: str,
        model: str,
        user_message: str,
        assistant_response: str,
        latency_ms: float,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        is_safe_input: bool = True,
        is_safe_output: bool = True,
        extra: Optional[Dict[str, Any]] = None,
    ) -> InteractionLog:
        entry = InteractionLog(
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            model=model,
            user_message=user_message,
            assistant_response=assistant_response,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            is_safe_input=is_safe_input,
            is_safe_output=is_safe_output,
            extra=extra or {},
        )
        self._logs.append(entry)
        return entry

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, filename: Optional[str] = None) -> str:
        if not filename:
            filename = f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = os.path.join(self.log_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(l) for l in self._logs], f, indent=2)
        return path

    # ── Aggregates ─────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        if not self._logs:
            return {}
        latencies = [l.latency_ms for l in self._logs]
        costs = [l.cost_usd for l in self._logs]
        tokens = [l.tokens_used for l in self._logs]
        return {
            "total_interactions": len(self._logs),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 1),
            "total_cost_usd": round(sum(costs), 6),
            "total_tokens": sum(tokens),
            "unsafe_inputs": sum(1 for l in self._logs if not l.is_safe_input),
            "unsafe_outputs": sum(1 for l in self._logs if not l.is_safe_output),
        }

    def get_logs(self) -> List[InteractionLog]:
        return list(self._logs)
