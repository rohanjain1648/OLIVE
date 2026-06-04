import re
from typing import Tuple

# Patterns that indicate harmful / jailbreak intent
_INPUT_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|your)\s+(instructions?|guidelines?|rules?)",
    r"(pretend|act\s+as|you\s+are\s+now)\s+.{0,30}(without|with\s+no)\s+(restrictions?|guidelines?|rules?|ethics?)",
    r"\b(DAN|do\s+anything\s+now|jailbreak)\b",
    r"\b(bomb|explosive|bioweapon|nerve\s+agent)\b.{0,30}(make|build|create|synthesize|instructions?)",
    r"(make|create|synthesize|produce)\s+.{0,30}(meth|heroin|fentanyl|cocaine|poison)",
    r"(write|create|generate)\s+.{0,30}(malware|ransomware|virus|trojan|keylogger)",
    r"how\s+to\s+(hack|crack|break\s+into)\s+.{0,30}(bank|server|system|account)",
    r"\b(self.harm|suicide\s+method|how\s+to\s+die)\b",
]

_OUTPUT_PATTERNS = [
    r"step\s+\d+.{0,60}(explosive|poison|weapon|malware)",
    r"(here\s+(are|is)|follow\s+these)\s+.{0,30}instructions.{0,30}(make|build|create).{0,30}(bomb|weapon|drug)",
]

_COMPILED_IN = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INPUT_PATTERNS]
_COMPILED_OUT = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _OUTPUT_PATTERNS]


def check_input_safety(text: str) -> Tuple[bool, str]:
    """Return (is_safe, reason). Fires before sending to the model."""
    for pat in _COMPILED_IN:
        if pat.search(text):
            return False, "Input matches a potentially harmful pattern — request blocked by guardrail."
    return True, ""


def check_output_safety(text: str) -> Tuple[bool, str]:
    """Return (is_safe, reason). Fires after receiving from the model."""
    for pat in _COMPILED_OUT:
        if pat.search(text):
            return False, "Output may contain harmful step-by-step instructions."
    return True, ""


def sanitize_response(text: str, replacement: str = "[Response filtered for safety]") -> str:
    is_safe, _ = check_output_safety(text)
    return text if is_safe else replacement
