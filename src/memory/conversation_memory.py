import json
import os
from typing import List, Dict


class ConversationMemory:
    """
    Wraps conversation history with optional summarisation of older turns
    so the context window stays manageable across long sessions.
    """

    MAX_MESSAGES = 20  # summarise when history exceeds this

    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = []
        self.summary: str = ""
        self.key_facts: List[str] = []

    # ── Core operations ────────────────────────────────────────────────────────

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.MAX_MESSAGES:
            self._compress()

    def clear(self):
        self.messages = []
        self.summary = ""
        self.key_facts = []

    def get_messages_for_api(self) -> List[Dict[str, str]]:
        """Return messages to send to the model, prepending the rolling summary if present."""
        if not self.summary:
            return self.messages
        # Inject summary as a synthetic assistant note before the live window
        return [{"role": "assistant", "content": f"[Conversation summary so far: {self.summary}]"}] + self.messages

    def add_key_fact(self, fact: str):
        if fact and fact not in self.key_facts:
            self.key_facts.append(fact)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, directory: str = "evaluation_results/sessions"):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.session_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": self.session_id,
                "messages": self.messages,
                "summary": self.summary,
                "key_facts": self.key_facts,
            }, f, indent=2)

    @classmethod
    def load(cls, session_id: str, directory: str = "evaluation_results/sessions") -> "ConversationMemory":
        mem = cls(session_id)
        path = os.path.join(directory, f"{session_id}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            mem.messages = data.get("messages", [])
            mem.summary = data.get("summary", "")
            mem.key_facts = data.get("key_facts", [])
        return mem

    # ── Internal ───────────────────────────────────────────────────────────────

    def _compress(self):
        """Keep the most recent 10 messages; summarise the rest."""
        older = self.messages[:-10]
        self.messages = self.messages[-10:]
        topics = [m["content"][:60] for m in older if m["role"] == "user"]
        new_summary = "; ".join(topics[:5])
        if self.summary:
            self.summary = f"{self.summary} | {new_summary}"
        else:
            self.summary = new_summary
