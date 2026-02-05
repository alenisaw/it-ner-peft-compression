# src/prompting.py
"""
Prompting baseline for IT-domain NER

Goal:
- Convert token-level NER into an LLM prompting task
- Use strict JSON outputs to reduce parsing ambiguity
- Evaluate with seqeval-compatible BIO tags on a small subset (20-50 samples)

Provider:
- ollama (recommended): local LLM via HTTP API
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests


ENTITY_TYPES = ["ERROR", "HARDWARE", "OS", "SOFTWARE", "VERSION"]


@dataclass(frozen=True)
class PromptTemplates:
    """Three prompt variants required by the assignment."""

    @staticmethod
    def v1_minimal() -> str:
        return (
            "You are an information extraction system.\n"
            "Task: IT-domain NER over a tokenized sentence.\n"
            f"Entity types: {', '.join(ENTITY_TYPES)}.\n"
            "Return ONLY valid JSON with this schema:\n"
            "{\n"
            '  "entities": [\n'
            '    {"type": "SOFTWARE", "start": 3, "end": 5, "text": "Visual Studio Code"}\n'
            "  ]\n"
            "}\n"
            "Notes: start/end are token indices, end is exclusive.\n"
            "If there are no entities, return {\"entities\": []}.\n"
        )

    @staticmethod
    def v2_role_strict() -> str:
        return (
            "Role: You label IT support tickets for NER.\n"
            f"Allowed entity types: {', '.join(ENTITY_TYPES)}.\n"
            "Rules:\n"
            "- Output MUST be valid JSON, nothing else\n"
            "- Use token indices (start inclusive, end exclusive)\n"
            "- Do not guess: if uncertain, omit the entity\n"
            "Schema: {\"entities\": [{\"type\": \"...\", \"start\": int, \"end\": int, \"text\": \"...\"}]}\n"
            "Return {\"entities\": []} when empty.\n"
        )

    @staticmethod
    def v3_fewshot() -> str:
        # Few-shot examples use tokens, to mirror the real input
        ex1_tokens = ["User", "cannot", "connect", "to", "VPN", "on", "Windows", "11"]
        ex1_out = {
            "entities": [
                {"type": "SOFTWARE", "start": 4, "end": 5, "text": "VPN"},
                {"type": "OS", "start": 6, "end": 8, "text": "Windows 11"},
            ]
        }
        ex2_tokens = ["Docker", "build", "fails", "with", "Error", "code", "1", "on", "Ubuntu", "22.04"]
        ex2_out = {
            "entities": [
                {"type": "SOFTWARE", "start": 0, "end": 1, "text": "Docker"},
                {"type": "ERROR", "start": 4, "end": 5, "text": "Error"},
                {"type": "VERSION", "start": 6, "end": 7, "text": "1"},
                {"type": "OS", "start": 8, "end": 10, "text": "Ubuntu 22.04"},
            ]
        }
        return (
            "You extract entities from token lists (NER).\n"
            f"Entity types: {', '.join(ENTITY_TYPES)}.\n"
            "Return ONLY valid JSON: {\"entities\": [{\"type\": \"...\", \"start\": int, \"end\": int, \"text\": \"...\"}]}\n"
            "Examples:\n"
            f"Tokens: {json.dumps(ex1_tokens)}\n"
            f"Output: {json.dumps(ex1_out)}\n\n"
            f"Tokens: {json.dumps(ex2_tokens)}\n"
            f"Output: {json.dumps(ex2_out)}\n\n"
            "Now do the same for the next input.\n"
        )


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def _normalize_entities(obj: Dict[str, Any], tokens: List[str]) -> Tuple[List[Dict[str, Any]], bool]:
    ents = obj.get("entities", [])
    if not isinstance(ents, list):
        return [], False

    ok = True
    out: List[Dict[str, Any]] = []
    for e in ents:
        if not isinstance(e, dict):
            ok = False
            continue
        t = str(e.get("type", "")).upper().strip()
        if t not in ENTITY_TYPES:
            ok = False
            continue
        try:
            s = int(e.get("start"))
            en = int(e.get("end"))
        except Exception:
            ok = False
            continue
        if s < 0 or en <= s or en > len(tokens):
            ok = False
            continue
        text_span = " ".join(tokens[s:en])
        out.append({"type": t, "start": s, "end": en, "text": str(e.get("text", text_span))})
    return out, ok


def spans_to_bio(tokens: List[str], spans: List[Dict[str, Any]]) -> List[str]:
    labs = ["O"] * len(tokens)
    for sp in spans:
        t = sp["type"]
        s = int(sp["start"])
        e = int(sp["end"])
        if s < 0 or e > len(tokens) or s >= e:
            continue
        if any(labs[i] != "O" for i in range(s, e)):
            continue
        labs[s] = f"B-{t}"
        for i in range(s + 1, e):
            labs[i] = f"I-{t}"
    return labs


def _call_ollama_generate(host: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(temperature)},
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return str(data.get("response", ""))


def predict_tokens(
    tokens: List[str],
    model: str,
    template: str,
    provider: str = "ollama",
    ollama_host: str = "http://localhost:11434",
    temperature: float = 0.0,
    max_retries: int = 2,
) -> Dict[str, Any]:
    provider = str(provider).lower().strip()

    base_prompt = template + "\n" + f"Tokens: {json.dumps(tokens, ensure_ascii=False)}\n" + "Output:"

    last_raw = ""
    for attempt in range(int(max_retries) + 1):
        prompt = base_prompt
        if attempt > 0:
            prompt = base_prompt + "\nRemember: return ONLY valid JSON. No markdown, no extra text."

        if provider == "ollama":
            raw = _call_ollama_generate(ollama_host, model, prompt, temperature=temperature)
        else:
            raise RuntimeError("Only provider='ollama' is supported in this repo version.")

        last_raw = raw
        obj = _extract_first_json(raw)
        if obj is None:
            continue
        spans, ok = _normalize_entities(obj, tokens)
        if spans and ok:
            return {"ok": True, "spans": spans, "raw": raw}
        if obj.get("entities") == []:
            return {"ok": True, "spans": [], "raw": raw}
        continue

    return {"ok": False, "spans": [], "raw": last_raw}


def eval_prompting_subset(
    samples: List[Dict[str, Any]],
    model: str,
    template: str,
    id2label: Dict[int, str],
    provider: str = "ollama",
    ollama_host: str = "http://localhost:11434",
    temperature: float = 0.0,
    seed: int = 13,
) -> Dict[str, Any]:
    import evaluate

    seqeval = evaluate.load("seqeval")

    preds_all: List[List[str]] = []
    refs_all: List[List[str]] = []
    invalid = 0

    for ex in samples:
        tokens = list(ex["tokens"])
        gold_ids = list(ex["ner_tags"])
        refs = [id2label[int(i)] for i in gold_ids]

        out = predict_tokens(
            tokens=tokens,
            model=model,
            template=template,
            provider=provider,
            ollama_host=ollama_host,
            temperature=temperature,
        )
        if not out["ok"]:
            invalid += 1
        pred = spans_to_bio(tokens, out["spans"])
        preds_all.append(pred)
        refs_all.append(refs)

    metrics = seqeval.compute(predictions=preds_all, references=refs_all, zero_division=0)
    overall = {
        "precision": float(metrics.get("overall_precision", 0.0)),
        "recall": float(metrics.get("overall_recall", 0.0)),
        "f1": float(metrics.get("overall_f1", 0.0)),
        "accuracy": float(metrics.get("overall_accuracy", 0.0)),
        "invalid_rate": float(invalid) / float(max(1, len(samples))),
        "n": int(len(samples)),
        "model": str(model),
    }
    return {"overall": overall, "raw": metrics}
