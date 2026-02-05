# src/robustness.py
"""
Robustness utilities

Creates "noisy" versions of token lists to simulate real-world text imperfections:
- casing: random case flips for alphabetic tokens
- punct: randomly drop or duplicate punctuation tokens
- typo: simple character-level typos for longer alphabetic tokens
"""

from __future__ import annotations

import random
import string
from typing import List

_PUNCT = set(list(".,:;!?()[]{}<>/\\|@#$%^&*+-=_~`'\""))


def perturb_tokens(tokens: List[str], mode: str, seed: int) -> List[str]:
    r = random.Random(int(seed))
    mode = str(mode).lower().strip()
    out = list(tokens)

    if mode == "casing":
        for i, t in enumerate(out):
            if t.isalpha() and r.random() < 0.30:
                out[i] = t.upper() if t.islower() else t.lower()
        return out

    if mode == "punct":
        new = []
        for t in out:
            if t in _PUNCT:
                if r.random() < 0.25:
                    continue
                new.append(t)
                if r.random() < 0.10:
                    new.append(t)
            else:
                new.append(t)
        return new

    if mode == "typo":

        def typo_word(w: str) -> str:
            if len(w) < 5:
                return w
            p = r.random()
            if p < 0.33:
                j = r.randint(0, len(w) - 2)
                s = list(w)
                s[j], s[j + 1] = s[j + 1], s[j]
                return "".join(s)
            if p < 0.66:
                j = r.randint(1, len(w) - 2)
                return w[:j] + w[j + 1 :]
            j = r.randint(1, len(w) - 2)
            return w[:j] + r.choice(string.ascii_lowercase) + w[j + 1 :]

        for i, t in enumerate(out):
            if t.isalpha() and r.random() < 0.15:
                out[i] = typo_word(t.lower())
        return out

    return out
