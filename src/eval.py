# src/eval.py
"""
Evaluation for IT-domain NER.

Fixes:
- Correctly loads LoRA runs saved as PEFT adapters (adapter_config.json) by:
  1) building base model with correct num_labels/id2label/label2id
  2) attaching adapter via PeftModel.from_pretrained(...)
- Keeps normal full-FT runs working (model saved directly in run_dir)

Notebook-friendly:
- call `eval_run(run=..., config_path=..., root=...)`

Saves under `models/<run>/`:
- metrics.json (overall P/R/F1/accuracy)
- per_entity.json (per-entity metrics from seqeval)
- errors.json (entity confusions + worst samples by mismatch count)
- eval.log
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import evaluate
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
)

from src.paths import ProjectPaths, ds_key, read_json, read_yaml, setup_logger, write_json

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def _load_processed_dataset(paths: ProjectPaths, dataset_name: str) -> DatasetDict:
    key = ds_key(dataset_name)
    ds_path = paths.processed_data / key
    if not ds_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {ds_path}. Run dataset preparation first.")
    return load_from_disk(str(ds_path))


def _load_label_names(paths: ProjectPaths) -> List[str]:
    meta_path = paths.data / "processed_meta.json"
    meta = read_json(meta_path)
    names = meta.get("new_label_names")
    if not names or not isinstance(names, list):
        raise RuntimeError("processed_meta.json does not contain new_label_names.")
    return [str(x) for x in names]


def _tokenize_and_align(examples, tokenizer, label_all_tokens: bool, max_length: int):
    tok = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=int(max_length),
    )
    aligned = []
    for i in range(len(examples["tokens"])):
        word_ids = tok.word_ids(batch_index=i)
        prev = None
        labs = []
        for wid in word_ids:
            if wid is None:
                labs.append(-100)
            elif wid != prev:
                labs.append(int(examples["ner_tags"][i][wid]))
            else:
                labs.append(int(examples["ner_tags"][i][wid]) if label_all_tokens else -100)
            prev = wid
        aligned.append(labs)
    tok["labels"] = aligned
    return tok


def _entity_of(label: str) -> str:
    if label == "O":
        return "O"
    if label.startswith(("B-", "I-")):
        return label[2:]
    return label


def _seqeval_metrics(
    pred_ids: np.ndarray,
    label_ids: np.ndarray,
    id2label: Dict[int, str],
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    seqeval = evaluate.load("seqeval")
    preds = np.argmax(pred_ids, axis=-1)

    true_predictions: List[List[str]] = []
    true_labels: List[List[str]] = []

    for pred_row, label_row in zip(preds, label_ids):
        sent_preds: List[str] = []
        sent_labels: List[str] = []
        for p, l in zip(pred_row, label_row):
            if int(l) == -100:
                continue
            sent_preds.append(id2label[int(p)])
            sent_labels.append(id2label[int(l)])
        true_predictions.append(sent_preds)
        true_labels.append(sent_labels)

    out = seqeval.compute(predictions=true_predictions, references=true_labels, zero_division=0)

    overall = {
        "precision": float(out.get("overall_precision", 0.0)),
        "recall": float(out.get("overall_recall", 0.0)),
        "f1": float(out.get("overall_f1", 0.0)),
        "accuracy": float(out.get("overall_accuracy", 0.0)),
    }

    per_entity: Dict[str, Dict[str, float]] = {}
    for k, v in out.items():
        if str(k).startswith("overall_"):
            continue
        if isinstance(v, dict) and all(m in v for m in ("precision", "recall", "f1")):
            per_entity[str(k)] = {
                "precision": float(v.get("precision", 0.0)),
                "recall": float(v.get("recall", 0.0)),
                "f1": float(v.get("f1", 0.0)),
                "number": float(v.get("number", 0.0)),
            }

    return overall, per_entity


def _confusions_and_examples(
    preds_logits: np.ndarray,
    labels: np.ndarray,
    id2label: Dict[int, str],
    max_examples: int,
    confusion_topk: int,
) -> Dict[str, Any]:
    preds = np.argmax(preds_logits, axis=-1)
    conf: Counter[tuple[str, str]] = Counter()
    example_rows: List[Dict[str, Any]] = []

    for i, (pr, lr) in enumerate(zip(preds, labels)):
        mism = 0
        row = {"idx": int(i), "true": [], "pred": []}

        for p, l in zip(pr, lr):
            if int(l) == -100:
                continue
            lt = id2label[int(l)]
            lp = id2label[int(p)]
            et = _entity_of(lt)
            ep = _entity_of(lp)
            if lt != lp:
                mism += 1
                conf[(et, ep)] += 1
            row["true"].append(lt)
            row["pred"].append(lp)

        if mism > 0:
            row["mismatches"] = int(mism)
            example_rows.append(row)

    example_rows.sort(key=lambda x: x.get("mismatches", 0), reverse=True)
    example_rows = example_rows[: int(max_examples)]

    conf_list = [
        {"true_entity": a, "pred_entity": b, "count": int(c)}
        for (a, b), c in conf.most_common(int(confusion_topk))
        if a != b
    ]

    return {"confusions_topk": conf_list, "examples_topk": example_rows}


def _is_lora_adapter_dir(run_dir: Path) -> bool:
    return (
        (run_dir / "adapter_config.json").exists()
        or (run_dir / "adapter_model.safetensors").exists()
        or (run_dir / "adapter_model.bin").exists()
    )


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
        model.eval()
        model.to(device)
        return model, tokenizer, id2label

    model = AutoModelForTokenClassification.from_pretrained(str(run_dir))
    model.eval()
    model.to(device)
    return model, tokenizer, id2label


def eval_run(
    run: str = "bert_lora",
    config_path: str = "configs/base.yaml",
    root: str | Path = ".",
) -> Dict[str, Any]:
    paths = ProjectPaths.from_root(root)
    paths.ensure()

    run_dir = paths.models / run
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir / "eval.log", "eval")

    cfg = read_yaml(Path(config_path))
    dataset_name = str(cfg["project"]["dataset"])
    max_length = int(cfg["data"]["max_length"])
    label_all_tokens = bool(cfg["data"]["label_all_tokens"])
    max_examples = int(cfg.get("eval", {}).get("max_error_examples", 50))
    confusion_topk = int(cfg.get("eval", {}).get("confusion_topk", 20))

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading tokenizer/model from %s", run_dir)
    model, tokenizer, id2label = _load_model_and_tokenizer(paths, run_dir, device=device)

    ds = _load_processed_dataset(paths, dataset_name)
    test_split = ds["test"]

    logger.info("Tokenizing test split: %d samples", len(test_split))
    tokenized_test = test_split.map(
        lambda ex: _tokenize_and_align(ex, tokenizer, label_all_tokens, max_length),
        batched=True,
        remove_columns=test_split.column_names,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    logger.info("Predict start")
    pred_out = trainer.predict(tokenized_test)
    logger.info("Predict done")

    overall, per_entity = _seqeval_metrics(pred_out.predictions, pred_out.label_ids, id2label)

    write_json(run_dir / "metrics.json", overall)
    write_json(run_dir / "per_entity.json", per_entity)

    errors = _confusions_and_examples(
        preds_logits=pred_out.predictions,
        labels=pred_out.label_ids,
        id2label=id2label,
        max_examples=max_examples,
        confusion_topk=confusion_topk,
    )
    write_json(run_dir / "errors.json", errors)

    resolved_path = run_dir / "config_resolved.json"
    resolved = read_json(resolved_path) if resolved_path.exists() else {}
    full = {"overall": overall, "per_entity": per_entity, "resolved": resolved}
    write_json(run_dir / "metrics_full.json", full)

    logger.info("Saved metrics.json, per_entity.json, metrics_full.json, errors.json")

    return {"run": run, "run_dir": str(run_dir), "overall": overall, "per_entity": per_entity}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="bert_lora")
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    args = ap.parse_args()
    out = eval_run(run=args.run, config_path=args.config, root=".")
    print(out["overall"])


if __name__ == "__main__":
    main()
