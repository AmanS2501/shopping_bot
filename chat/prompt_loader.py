import json
from pathlib import Path
from typing import Dict, Any

def _escape_braces(s: str) -> str:
    # Prevent str.format from treating JSON braces as placeholders
    return s.replace("{", "{{").replace("}", "}}")

class PromptConfig:
    def __init__(self, data: Dict[str, Any]):
        self.data = data

    @classmethod
    def load(cls, path: str) -> "PromptConfig":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def render_system(self) -> str:
        lab = self.data.get("labels", {})
        lines = [
            ln.format(assistant_name=lab.get("assistant_name", "Assistant"))
            for ln in self.data["prompt_shell"]["system"]
        ]
        return "\n".join(lines)

    def render_developer(self) -> str:
        rules = _escape_braces(json.dumps(self.data.get("rules", {}), ensure_ascii=False))
        personality = _escape_braces(json.dumps(self.data.get("personality", {}), ensure_ascii=False))
        lines = [
            ln.format(rules=rules, personality=personality)
            for ln in self.data["prompt_shell"]["developer"]
        ]
        return "\n".join(lines)

    def render_user(self, question: str, context: str) -> str:
        # Escape braces in context to avoid accidental formatting tokens from documents
        safe_context = _escape_braces(context)
        return self.data["prompt_shell"]["user_template"].format(
            question=question, context=safe_context
        )
