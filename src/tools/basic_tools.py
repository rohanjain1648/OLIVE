"""
Basic tools shared by both assistants.
- Frontier (Groq/Llama): injected via pattern detection into augmented prompt
- OSS (Qwen): pattern-injected into the prompt before calling the model
"""
import math
import re
from datetime import datetime
from typing import Any, Dict


def calculator(expression: str) -> dict:
    """Evaluate a mathematical expression safely.

    Supports arithmetic (+, -, *, /), parentheses, sqrt, sin, cos, tan,
    log, log10, floor, ceil, abs, round, min, max, pow, pi, e.

    Args:
        expression: The math expression to evaluate, e.g. '2+2', 'sqrt(16)', 'sin(pi/2)'

    Returns:
        A dict with 'result' (numeric answer), 'expression' (cleaned input), 'success' (bool).
    """
    safe_env: Dict[str, Any] = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
        "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "log10": math.log10,
        "floor": math.floor, "ceil": math.ceil,
    }
    clean = re.sub(r"[^0-9+\-*/().%, a-zA-Z]", "", expression)
    try:
        result = eval(clean, safe_env)  # noqa: S307
        return {"result": result, "expression": clean, "success": True}
    except Exception as exc:
        return {"error": str(exc), "expression": clean, "success": False}


def get_current_datetime() -> dict:
    """Get the current local date and time.

    Returns:
        A dict with 'date' (YYYY-MM-DD), 'time' (HH:MM:SS), 'datetime' (ISO), 'day_of_week'.
    """
    now = datetime.now()
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "timezone": "Local",
    }


def get_weather(location: str) -> dict:
    """Get simulated current weather for a city or location.

    Args:
        location: City name or location string, e.g. 'London', 'New York'

    Returns:
        A dict with temperature, condition, humidity, wind, and a disclaimer note.
    """
    return {
        "location": location,
        "temperature": "22°C / 72°F",
        "condition": "Partly cloudy",
        "humidity": "65%",
        "wind": "15 km/h NW",
        "note": "Simulated weather data — replace with a real weather API for production.",
    }


# ── Shared handler map (used by OSS pattern-injection) ────────────────────────
TOOL_HANDLERS: Dict[str, Any] = {
    "calculator": calculator,
    "get_current_datetime": get_current_datetime,
    "get_weather": get_weather,
}
