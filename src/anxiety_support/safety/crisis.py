"""Crisis handling: a high-precision keyword net and the crisis response text.

This is defence in depth. Crisis disclosures are caught by (1) this regex net,
(2) Llama Guard category S11, and (3) the NeMo dialog rail `user express crisis`.
A crisis is never just "blocked": the user always gets a supportive reply with
helplines.
"""

from __future__ import annotations

import re

_PATTERNS = [
    r"\b(kill|killing|hang|hanging|hurt|hurting|harm|harming|cut|cutting)\s+my\s*self\b",
    r"\bkill\s+myself\b",
    r"\bsuicid(e|al)\b",
    r"\bend(ing)?\s+(my|it)\s+(life|all)\b",
    r"\b(want|wanna|going)\s+to\s+die\b",
    r"\bdon'?t\s+want\s+to\s+(live|be\s+alive|exist)\b",
    r"\b(no|not\s+any)\s+reason\s+to\s+live\b",
    r"\bbetter\s+off\s+(dead|without\s+me)\b",
    r"\boverdos(e|ing)\b",
    r"\bself[-\s]?harm\b",
    r"\b(cut|cutting|burn|burning)\s+(my\s+)?(arms?|wrists?|legs?|thighs?|skin)\b",
]
_CRISIS_RE = re.compile("|".join(_PATTERNS), re.IGNORECASE)

RESOURCES = {
    "IN": [
        "Tele-MANAS (Govt. of India, 24x7, free): call 14416 or 1-800-891-4416",
        "Emergency services: 112",
    ],
    "US": [
        "988 Suicide & Crisis Lifeline: call or text 988",
        "Emergency services: 911",
    ],
    "INTL": ["Find a local helpline: https://findahelpline.com"],
}


def detect_crisis(text: str) -> bool:
    return bool(_CRISIS_RE.search(text))


def crisis_message(region: str = "IN") -> str:
    lines = RESOURCES.get(region.upper(), []) + RESOURCES["INTL"]
    resources = "\n".join(f"- {line}" for line in lines)
    return (
        "I'm really sorry you're feeling this much pain, and I'm glad you told me. "
        "Your safety matters most right now, and this is more than I can help with on my own. "
        "Please reach out to someone who can support you right away:\n"
        f"{resources}\n"
        "If you are in immediate danger, please call emergency services or go to the nearest hospital. "
        "If you can, let someone you trust know how you're feeling and stay with them. "
        "I'm still here to talk with you."
    )
