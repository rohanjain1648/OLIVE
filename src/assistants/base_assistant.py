from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any
import time


@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AssistantResponse:
    content: str
    model: str
    latency_ms: float
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAssistant(ABC):
    def __init__(self, system_prompt: str = None):
        self.conversation_history: List[Message] = []
        self.system_prompt = system_prompt or (
            "You are a helpful, harmless, and honest AI assistant. "
            "Answer questions accurately, refuse harmful requests politely, "
            "and maintain a respectful, unbiased tone."
        )

    def add_message(self, role: str, content: str):
        self.conversation_history.append(Message(role=role, content=content))

    def clear_history(self):
        self.conversation_history = []

    def get_history_as_dicts(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.conversation_history]

    @abstractmethod
    def chat(self, user_message: str) -> AssistantResponse:
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        pass
