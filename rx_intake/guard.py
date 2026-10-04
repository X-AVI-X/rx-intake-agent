"""Prompt-injection screening for incoming documents.

This is defence in depth, not the main control. The main controls are
structural: the document is passed as data inside <document> tags, the model's
only output is a fixed schema, and it has no tools that can change anything.
So even a successful injection can at worst produce wrong field values, which
the rules engine and human review are there to catch. This screen just makes
sure suspicious documents never get auto-accepted.
"""

from __future__ import annotations

import re

from .schemas import Issue, Severity

_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|rules)",
    r"disregard (the |all )?(previous|prior|above|system)",
    r"\bsystem prompt\b",
    r"you are now\b",
    r"\bact as\b",
    r"\b(note|message|instructions?) (to|for) (the )?(ai|model|assistant|llm|bot)\b",
    r"(approve|accept) (this|automatically|without review)",
    r"skip (the )?(review|checks?|validation)",
    r"\bset (status|severity)\b",
    r"</?(system|instructions?|document)>",
]
_REGEX = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def screen(text: str) -> list[Issue]:
    hits = sorted({m.group(0).lower() for m in _REGEX.finditer(text)})
    if not hits:
        return []
    return [Issue(code="possible_prompt_injection", severity=Severity.BLOCK,
                  message=f"Instruction-like text in document: {', '.join(hits)}")]
