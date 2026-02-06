import html
import json
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

from src.paths import ProjectPaths, read_json

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


ENTITY_COLORS = {
    "SOFTWARE": "#38BDF8",
    "OS": "#22C55E",
    "HARDWARE": "#F59E0B",
    "ERROR": "#F43F5E",
    "VERSION": "#A78BFA",
}


APP_CSS = """
<style>
:root {
  --bg-0: #05070f;
  --bg-1: #0a0f1f;
  --bg-2: #0b1224;
  --glass: rgba(15, 23, 42, 0.75);
  --glass-strong: rgba(15, 23, 42, 0.85);
  --surface-light: #0f172a;
  --surface-light-2: #111827;
  --ink-dark: #0b1220;
  --ink-light: #f8fafc;
  --muted-light: #cbd5e1;
  --stroke: rgba(148, 163, 184, 0.35);
  --stroke-light: rgba(148, 163, 184, 0.45);
}

html, body, [class*="css"] {
  font-family: "Space Grotesk", "IBM Plex Sans", "Segoe UI", sans-serif;
  color: var(--ink-light);
  font-weight: 500;
}

.stApp {
  background:
    radial-gradient(1200px 620px at 6% -12%, rgba(56,189,248,0.30) 0%, rgba(56,189,248,0) 62%),
    radial-gradient(1040px 600px at 112% 6%, rgba(167,139,250,0.34) 0%, rgba(167,139,250,0) 62%),
    radial-gradient(820px 480px at 50% 120%, rgba(244,63,94,0.22) 0%, rgba(244,63,94,0) 58%),
    linear-gradient(180deg, var(--bg-0) 0%, var(--bg-2) 100%);
}

.main .block-container {
  max-width: 1400px;
  padding-left: 2.5rem;
  padding-right: 2.5rem;
}

@media (max-width: 900px) {
  .main .block-container {
    padding-left: 1.25rem;
    padding-right: 1.25rem;
  }
}

.hero {
  background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(15,23,42,0.6) 100%);
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 18px 22px;
  box-shadow: 0 18px 42px rgba(15, 23, 42, 0.18);
  margin-bottom: 16px;
  animation: fadeIn 0.6s ease-out;
  color: var(--ink-light);
  backdrop-filter: blur(14px);
}

.hero .kicker {
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 1.6px;
  color: var(--muted-light);
  margin-bottom: 6px;
}

.hero h1 {
  font-size: 34px;
  margin: 0 0 6px 0;
  letter-spacing: 0.5px;
  font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
  color: var(--ink-light);
  font-weight: 700;
}

.hero p {
  margin: 0;
  color: var(--muted-light);
}

.section-title {
  margin: 18px 0 10px;
  font-size: 18px;
  font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
  color: var(--ink-light);
  letter-spacing: 0.4px;
  font-weight: 700;
}

.section-sub {
  margin-top: -4px;
  margin-bottom: 12px;
  color: var(--muted-light);
  font-size: 13px;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 10px 0 6px;
}

.legend span {
  border-radius: 999px;
  padding: 4px 10px;
  background: var(--surface-light-2);
  border: 1px solid var(--stroke-light);
  font-size: 12px;
  color: var(--ink-light);
  box-shadow: 0 10px 22px rgba(15, 23, 42, 0.22);
  font-weight: 600;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}

.overview-card {
  background: var(--glass-strong);
  border: 1px solid var(--stroke);
  border-radius: 14px;
  padding: 12px 14px;
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.22);
  backdrop-filter: blur(14px);
}

.overview-card .title {
  font-size: 14px;
  color: var(--ink-light);
  margin-bottom: 4px;
  font-weight: 700;
}

.overview-card .sub {
  font-size: 12px;
  color: var(--muted-light);
  margin-bottom: 10px;
}

.overview-card .row {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  color: var(--ink-light);
  margin-bottom: 4px;
}

.overview-card .row span {
  color: var(--muted-light);
}

.run-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(15,23,42,0.6) 100%);
  border: 1px solid var(--stroke);
  border-radius: 14px;
  padding: 12px 16px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
  margin: 8px 0 10px;
  color: var(--ink-light);
  backdrop-filter: blur(12px);
}

.run-title {
  font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
  font-size: 20px;
  letter-spacing: 0.3px;
  font-weight: 700;
}

.run-sub {
  color: var(--muted-light);
  font-size: 12px;
  margin-top: 2px;
}

.run-code {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 8px;
  border: 1px solid var(--stroke);
  background: rgba(15, 23, 42, 0.7);
  color: var(--ink-light);
  font-family: "IBM Plex Mono", "Consolas", monospace;
}

.run-divider {
  height: 1px;
  margin: 22px 0;
  background: linear-gradient(90deg, rgba(56,189,248,0), rgba(56,189,248,0.8), rgba(167,139,250,0.8), rgba(56,189,248,0));
  border-radius: 999px;
}

.highlight-box {
  background: var(--glass-strong);
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 16px;
  line-height: 1.7;
  font-size: 18px;
  box-shadow: 0 16px 34px rgba(15, 23, 42, 0.24);
  color: var(--ink-light);
}

.ent {
  position: relative;
  padding: 0 3px;
  border-bottom: 3px solid var(--ent-color);
  background: linear-gradient(180deg, rgba(0,0,0,0) 60%, rgba(255,255,255,0.12) 100%);
  border-radius: 4px;
}

.ent-label {
  font-size: 11px;
  color: var(--ink-light);
  background: rgba(15, 23, 42, 0.7);
  border: 1px solid var(--ent-color);
  border-radius: 6px;
  padding: 1px 6px;
  margin-left: 6px;
}

.tokens {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 8px 0 2px;
}

.tok {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border: 1px solid var(--stroke);
  border-radius: 10px;
  padding: 2px 8px;
  background: var(--glass-strong);
  font-size: 13px;
  color: var(--ink-light);
  font-weight: 600;
}

.tok.ent {
  border-color: var(--tok-color);
  box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.08);
}

.tok-label {
  font-size: 10px;
  color: var(--ink-light);
  background: rgba(15, 23, 42, 0.7);
  border: 1px solid var(--tok-color);
  border-radius: 6px;
  padding: 1px 6px;
}

div.stButton > button {
  background: linear-gradient(120deg, #0ea5e9, #6366f1);
  color: #fff;
  border: none;
  padding: 0.55rem 1.2rem;
  border-radius: 10px;
  box-shadow: 0 10px 24px rgba(14, 165, 233, 0.35);
}

div.stButton > button:hover {
  filter: brightness(1.05);
}

div[data-baseweb="tag"] {
  background: var(--glass-strong);
  border: 1px solid var(--stroke);
}

div[data-baseweb="tag"] span {
  color: var(--ink-light);
  font-weight: 600;
}

section[data-testid="stSidebar"] > div {
  background: linear-gradient(180deg, var(--bg-1) 0%, #0b1220 100%);
}

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown {
  color: var(--ink-light);
}

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.10);
  color: var(--ink-light);
  border: 1px solid var(--stroke);
}

section[data-testid="stSidebar"] svg {
  fill: var(--ink-light);
}

div[data-testid="stMetric"] {
  background: var(--glass);
  border: 1px solid var(--stroke);
  border-radius: 12px;
  padding: 10px 12px;
  color: var(--ink-light);
  backdrop-filter: blur(12px);
}

div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div {
  color: var(--ink-light);
}

h3 {
  color: var(--ink-light);
}

div[data-testid="stTextArea"] textarea {
  background: rgba(15, 23, 42, 0.8);
  color: var(--ink-light);
  border: 1px solid var(--stroke);
  box-shadow: 0 12px 26px rgba(15, 23, 42, 0.35);
  backdrop-filter: blur(12px);
  font-weight: 600;
}

div[data-testid="stTextArea"] textarea::placeholder {
  color: var(--muted-light);
}

div[data-testid="stDataFrame"] {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 6px;
}

div[data-testid="stDataFrame"] * {
  color: #0b1220;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
"""


def _is_lora_adapter_dir(run_dir: Path) -> bool:
    return (
        (run_dir / "adapter_config.json").exists()
        or (run_dir / "adapter_model.safetensors").exists()
        or (run_dir / "adapter_model.bin").exists()
    )


def _load_label_names(paths: ProjectPaths) -> list[str]:
    meta_path = paths.data / "processed_meta.json"
    meta = read_json(meta_path)
    names = meta.get("new_label_names")
    if not names or not isinstance(names, list):
        raise RuntimeError("processed_meta.json does not contain new_label_names.")
    return [str(x) for x in names]


def _load_model_and_tokenizer(paths: ProjectPaths, run_dir: Path, device: str):
    label_names = _load_label_names(paths)
    label2id = {n: i for i, n in enumerate(label_names)}
    id2label = {i: n for i, n in enumerate(label_names)}

    tokenizer = AutoTokenizer.from_pretrained(str(run_dir), use_fast=True)

    resolved_path = run_dir / "config_resolved.json"
    resolved = read_json(resolved_path) if resolved_path.exists() else {}
    base_model = str(resolved.get("base_model") or "bert-base-cased")

    if _is_lora_adapter_dir(run_dir):
        if PeftModel is None:
            raise RuntimeError("This run looks like a PEFT/LoRA adapter, but `peft` is not installed.")

        cfg = AutoConfig.from_pretrained(base_model)
        cfg.num_labels = int(len(label_names))
        cfg.id2label = {int(k): v for k, v in id2label.items()}
        cfg.label2id = {k: int(v) for k, v in label2id.items()}

        base = AutoModelForTokenClassification.from_pretrained(base_model, config=cfg)
        model = PeftModel.from_pretrained(base, str(run_dir))
    else:
        model = AutoModelForTokenClassification.from_pretrained(str(run_dir))
        if not model.config.id2label or len(model.config.id2label) != len(label_names):
            model.config.id2label = {int(k): v for k, v in id2label.items()}
            model.config.label2id = {k: int(v) for k, v in label2id.items()}

    model.eval()
    model.to(device)
    return model, tokenizer, id2label


def _build_spans(text: str, offsets: list[tuple[int, int]], labels: list[str], mask: list[int]):
    spans: list[dict[str, int | str]] = []
    current = None
    for (start, end), lab, keep in zip(offsets, labels, mask):
        if not keep or start == end:
            continue
        if lab == "O":
            if current:
                spans.append(current)
                current = None
            continue

        ent = lab[2:] if lab.startswith(("B-", "I-")) else lab
        if lab.startswith("B-") or current is None or current["label"] != ent or start > current["end"]:
            if current:
                spans.append(current)
            current = {"label": ent, "start": int(start), "end": int(end)}
        else:
            current["end"] = int(end)

    if current:
        spans.append(current)
    return spans


def _build_token_rows(
    tokens: list[str],
    offsets: list[tuple[int, int]],
    labels: list[str],
    mask: list[int],
    special_tokens: set[str],
):
    rows = []
    for tok, (start, end), lab, keep in zip(tokens, offsets, labels, mask):
        if not keep:
            continue
        if start == end and tok in special_tokens:
            continue
        rows.append({"token": tok, "label": lab, "start": int(start), "end": int(end)})
    return rows


def _render_highlighted(text: str, spans: list[dict], colors: dict[str, str]) -> str:
    spans_sorted = sorted(spans, key=lambda s: (s["start"], s["end"]))
    out = []
    last = 0
    for sp in spans_sorted:
        start = int(sp["start"])
        end = int(sp["end"])
        if start < last or end <= start:
            continue
        out.append(html.escape(text[last:start]))
        color = colors.get(str(sp["label"]), "#CBD5E1")
        segment = html.escape(text[start:end])
        out.append(
            f'<span class="ent" style="--ent-color:{color}">{segment}'
            f'<span class="ent-label">{html.escape(str(sp["label"]))}</span></span>'
        )
        last = end
    out.append(html.escape(text[last:]))
    return "".join(out)


def _render_tokens(rows: list[dict], colors: dict[str, str]) -> str:
    parts = ['<div class="tokens">']
    for row in rows:
        tok = html.escape(str(row["token"]))
        label = str(row["label"])
        ent = label[2:] if label.startswith(("B-", "I-")) else label
        color = colors.get(ent, "#CBD5E1")
        cls = "tok ent" if ent != "O" else "tok"
        label_html = f'<span class="tok-label" style="--tok-color:{color}">{html.escape(label)}</span>'
        parts.append(
            f'<span class="{cls}" style="--tok-color:{color}">{tok}{label_html}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


@st.cache_resource(show_spinner=False)
def _load_cached(run: str, device: str):
    paths = ProjectPaths.from_root(".")
    run_dir = paths.models / run
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    return _load_model_and_tokenizer(paths, run_dir, device=device)


def _load_metrics(run_dir: Path):
    for name in ("metrics.json", "metrics_full.json"):
        path = run_dir / name
        if not path.exists():
            continue
        data = read_json(path)
        if name == "metrics_full.json" and isinstance(data, dict):
            data = data.get("overall", data)
        if isinstance(data, dict):
            return data
    return None


def _load_per_entity(run_dir: Path) -> dict | None:
    path = run_dir / "per_entity.json"
    if not path.exists():
        return None
    data = read_json(path)
    return data if isinstance(data, dict) else None


def _pretty_run_name(run: str) -> str:
    mapping = {
        "bert_full": "BERT - Full Fine-Tune",
        "bert_lora": "BERT - LoRA (PEFT)",
        "distil_full": "DistilBERT - Full Fine-Tune",
    }
    if run in mapping:
        return mapping[run]
    return run.replace("_", " ").strip().title()


def _fmt_metric(metrics: dict, key: str) -> str:
    val = metrics.get(key) if metrics else None
    if val is None:
        return "n/a"
    try:
        return f"{float(val):.3f}"
    except Exception:
        return "n/a"


def _predict(text: str, model, tokenizer, id2label: dict[int, str], max_length: int, device: str):
    if not text.strip():
        return [], []
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=int(max_length),
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    input_ids = enc["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    mask = enc.get("attention_mask", torch.ones_like(enc["input_ids"]))[0].tolist()
    inputs = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        logits = model(**inputs).logits

    pred_ids = logits.argmax(dim=-1)[0].tolist()
    labels = [id2label.get(int(i), "O") for i in pred_ids]
    spans = _build_spans(text, offsets, labels, mask)
    rows = _build_token_rows(tokens, offsets, labels, mask, set(tokenizer.all_special_tokens))
    return spans, rows


def _build_entity_frame(per_entity_by_run: dict[str, dict], runs: list[str]) -> pd.DataFrame:
    entity_set = set()
    for data in per_entity_by_run.values():
        if not isinstance(data, dict):
            continue
        entity_set.update(data.keys())

    ordered = [e for e in ENTITY_COLORS.keys() if e in entity_set]
    ordered += sorted([e for e in entity_set if e not in ordered])

    rows = {}
    for run in runs:
        per_ent = per_entity_by_run.get(run) or {}
        rows[run] = {e: float(per_ent.get(e, {}).get("f1", float("nan"))) for e in ordered}

    df = pd.DataFrame(rows, index=ordered)
    return df


def _heat_text_color(val) -> str:
    try:
        if pd.isna(val):
            return "color: #0b1220"
        return "color: #0b1220" if float(val) < 0.65 else "color: #f8fafc"
    except Exception:
        return "color: #0b1220"


def _style_heatmap(df: pd.DataFrame):
    styler = df.style.format("{:.3f}", na_rep="-")
    styler = styler.background_gradient(cmap="YlGnBu", vmin=0.0, vmax=1.0)
    styler = styler.applymap(_heat_text_color)
    return styler


def _render_overview_cards(
    runs: list[str],
    metrics_by_run: dict[str, dict],
    latency_by_run: dict[str, float],
) -> str:
    cards = []
    for run in runs:
        metrics = metrics_by_run.get(run) or {}
        latency_ms = latency_by_run.get(run)
        latency = f"{latency_ms:.1f} ms" if latency_ms is not None else "n/a"
        cards.append(
            "\n".join(
                [
                    '<div class="overview-card">',
                    f'<div class="title">{html.escape(_pretty_run_name(run))}</div>',
                    f'<div class="sub">{html.escape(run)}</div>',
                    f'<div class="row"><span>F1</span><strong>{_fmt_metric(metrics, "f1")}</strong></div>',
                    f'<div class="row"><span>Precision</span><strong>{_fmt_metric(metrics, "precision")}</strong></div>',
                    f'<div class="row"><span>Recall</span><strong>{_fmt_metric(metrics, "recall")}</strong></div>',
                    f'<div class="row"><span>Latency</span><strong>{latency}</strong></div>',
                    "</div>",
                ]
            )
        )
    return f'<div class="overview-grid">{"".join(cards)}</div>'


def _compare_token_rows(
    rows_a: list[dict],
    rows_b: list[dict],
    max_items: int = 200,
) -> tuple[list[dict], int]:
    n = min(len(rows_a), len(rows_b))
    mismatches = []
    for i in range(n):
        if rows_a[i]["label"] != rows_b[i]["label"]:
            mismatches.append(
                {
                    "idx": i,
                    "token_a": rows_a[i]["token"],
                    "label_a": rows_a[i]["label"],
                    "token_b": rows_b[i]["token"],
                    "label_b": rows_b[i]["label"],
                }
            )
        if len(mismatches) >= int(max_items):
            break
    return mismatches, n


def _make_entities_rows(text: str, spans: list[dict]) -> list[dict]:
    rows = []
    for s in spans:
        rows.append(
            {
                "text": text[s["start"] : s["end"]],
                "label": s["label"],
                "start": int(s["start"]),
                "end": int(s["end"]),
            }
        )
    return rows


def main() -> None:
    st.set_page_config(page_title="Entity Prism - IT NER Showcase", page_icon=":brain:", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.markdown(
        """
<div class="hero">
  <div class="kicker">IT NER / Model Showcase</div>
  <h1>Entity Prism</h1>
  <p>Paste a ticket, run multiple models, and compare entity extraction with a visual dashboard.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    paths = ProjectPaths.from_root(".")
    runs = [d.name for d in paths.models.iterdir() if d.is_dir() and d.name != "onnx"]
    runs = sorted(runs)

    if not runs:
        st.error("No model runs found in models/. Train or add a run first.")
        st.stop()

    samples = {
        "VPN + OS": "After the Windows 11 update, VPN stopped working and Docker returns Error code 1.",
        "Hardware fault": "Server RAM failure on Ubuntu 22.04 after BIOS update, error 0x3f.",
        "Dev stack": "Kubernetes on AWS fails with timeout; Helm 3.12.1 and Docker are involved.",
    }

    if "input_text" not in st.session_state:
        st.session_state["input_text"] = ""
    if "results" not in st.session_state:
        st.session_state["results"] = {}
    if "last_runs" not in st.session_state:
        st.session_state["last_runs"] = []
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "Overview"

    with st.sidebar:
        st.markdown("### Models")
        selected = st.multiselect("Models to run", runs, default=runs)
        device_choice = st.selectbox("Device for inference", ["auto", "cpu", "cuda"], index=0)
        max_length = st.slider("Max input length (tokens)", min_value=64, max_value=256, value=128, step=16)
        show_tokens = st.checkbox("Show token stream", value=True)
        show_tables = st.checkbox("Show tables & exports", value=True)
        st.markdown("### Samples")
        sample_key = st.selectbox("Quick sample", list(samples.keys()), index=0)
        col_a, col_b = st.columns(2)
        if col_a.button("Insert sample"):
            st.session_state["input_text"] = samples[sample_key]
        if col_b.button("Clear input"):
            st.session_state["input_text"] = ""

    if not selected:
        st.warning("Select at least one run to compare.")
        st.stop()

    if device_choice == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_choice
        if device == "cuda" and not torch.cuda.is_available():
            st.warning("CUDA not available, falling back to CPU.")
            device = "cpu"

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    text = st.text_area(
        "Ticket / incident text",
        key="input_text",
        height=160,
        placeholder="Example: After the Windows 11 update, VPN stopped working and Docker returns Error code 1.",
    )
    st.caption("Tip: short, IT-focused sentences yield cleaner entity boundaries.")
    run_btn = st.button("Run NER across selected models", type="primary")

    legend_items = "".join(
        [
            f'<span style="border-color:{c}; color:{c}; box-shadow: inset 0 0 0 1px {c};">{k}</span>'
            for k, c in ENTITY_COLORS.items()
        ]
    )
    st.markdown(f'<div class="legend">{legend_items}</div>', unsafe_allow_html=True)

    metrics_by_run = {run: _load_metrics(paths.models / run) for run in selected}
    per_entity_by_run = {run: _load_per_entity(paths.models / run) for run in selected}

    if run_btn:
        if not text.strip():
            st.warning("Type some text first.")
        else:
            results = {}
            for run in selected:
                try:
                    with st.spinner(f"Loading {run}..."):
                        model, tokenizer, id2label = _load_cached(run, device=device)
                    with st.spinner("Running NER..."):
                        t0 = time.perf_counter()
                        spans, token_rows = _predict(
                            text, model, tokenizer, id2label, max_length=max_length, device=device
                        )
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                except Exception as e:
                    st.error(f"Failed on {run}: {e}")
                    continue
                results[run] = {
                    "spans": spans,
                    "token_rows": token_rows,
                    "latency_ms": latency_ms,
                }

            st.session_state["results"] = results
            st.session_state["last_runs"] = list(selected)

    results = st.session_state.get("results", {})
    latency_by_run = {run: results.get(run, {}).get("latency_ms") for run in selected}

    st.markdown('<div class="section-title">Run Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Snapshot of overall metrics and latency for each run.</div>', unsafe_allow_html=True)
    st.markdown(
        _render_overview_cards(selected, metrics_by_run, latency_by_run),
        unsafe_allow_html=True,
    )
    st.caption("Latency updates after a Run NER pass.")

    st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Highlighted entities, tables, token stream, and exports.</div>', unsafe_allow_html=True)
    if not results:
        st.info("Run the models to see highlighted entities.")
    else:
        for i, run in enumerate(selected):
            if run not in results:
                continue
            pretty = _pretty_run_name(run)
            st.markdown(
                f'''
<div class="run-header">
  <div>
    <div class="run-title">{html.escape(pretty)}</div>
    <div class="run-sub">run: <span class="run-code">{html.escape(run)}</span></div>
  </div>
</div>
''',
                unsafe_allow_html=True,
            )

            metrics = metrics_by_run.get(run)
            if metrics:
                cols = st.columns(5)
                cols[0].metric("Overall Precision", _fmt_metric(metrics, "precision"))
                cols[1].metric("Overall Recall", _fmt_metric(metrics, "recall"))
                cols[2].metric("Overall F1", _fmt_metric(metrics, "f1"))
                cols[3].metric("Token Accuracy", _fmt_metric(metrics, "accuracy"))
                latency = results[run].get("latency_ms")
                cols[4].metric("Latency", f"{latency:.1f} ms" if latency else "n/a")

            spans = results[run]["spans"]
            token_rows = results[run]["token_rows"]

            html_out = _render_highlighted(text, spans, ENTITY_COLORS)
            st.markdown(f'<div class="highlight-box">{html_out}</div>', unsafe_allow_html=True)
            st.caption(f"Entities: {len(spans)} | Tokens: {len(token_rows)}")

            if show_tables:
                if spans:
                    rows = _make_entities_rows(text, spans)
                    st.markdown("#### Extracted entities")
                    st.dataframe(rows, hide_index=True, use_container_width=True)

                    col_a, col_b = st.columns(2)
                    col_a.download_button(
                        "Download JSON",
                        data=json.dumps(rows, indent=2),
                        file_name=f"{run}_entities.json",
                        mime="application/json",
                    )
                    df = pd.DataFrame(rows)
                    col_b.download_button(
                        "Download CSV",
                        data=df.to_csv(index=False),
                        file_name=f"{run}_entities.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No entities detected.")

            if show_tokens:
                st.markdown("#### Token stream")
                st.markdown(_render_tokens(token_rows, ENTITY_COLORS), unsafe_allow_html=True)

            if i < len(selected) - 1:
                st.markdown('<div class="run-divider"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model Diff</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Token-level mismatches between two runs.</div>', unsafe_allow_html=True)
    if not results or len([r for r in selected if r in results]) < 2:
        st.info("Run at least two models to compare token labels.")
    else:
        col_a, col_b = st.columns(2)
        run_a = col_a.selectbox("Run A", selected, index=0, key="cmp_a")
        run_b = col_b.selectbox("Run B", selected, index=1, key="cmp_b")
        if run_a == run_b:
            st.warning("Pick two different runs.")
        else:
            rows_a = results.get(run_a, {}).get("token_rows", [])
            rows_b = results.get(run_b, {}).get("token_rows", [])
            mismatches, n = _compare_token_rows(rows_a, rows_b)
            st.metric("Token mismatches", len(mismatches))
            st.caption(f"Compared tokens: {n}")
            if mismatches:
                st.dataframe(mismatches, hide_index=True, use_container_width=True)
            else:
                st.success("No label mismatches for aligned tokens.")

    st.markdown('<div class="section-title">Per-entity quality</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">F1 heatmap across entity types and runs.</div>', unsafe_allow_html=True)
    df = _build_entity_frame(per_entity_by_run, selected)
    if df.empty:
        st.info("No per-entity metrics found. Run evaluation to generate per_entity.json.")
    else:
        st.dataframe(_style_heatmap(df), use_container_width=True)


if __name__ == "__main__":
    main()
