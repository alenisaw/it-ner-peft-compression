"""
Training for IT-domain NER (full fine-tuning vs LoRA PEFT vs DistilBERT).

Notebook-friendly:
- call `train_run(run=..., config_path=..., root=..., force_cpu=...)`
- saves all artifacts under `models/<run>/`

Runs:
- bert_full   : bert-base-cased full fine-tuning
- bert_lora   : bert-base-cased + LoRA (PEFT)
- distil_full : distilbert-base-cased full fine-tuning

Artifacts per run (models/<run>/):
- run.log
- config_resolved.json
- metrics_train.json
- trainer_log_history.json
- model + tokenizer
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import evaluate
import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.paths import ProjectPaths, ds_key, read_json, read_yaml, setup_logger, write_json

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    base_model: str
    use_lora: bool


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


def _pick_run(run: str, lora_enabled: bool) -> RunSpec:
    r = run.strip().lower()
    if r == "bert_full":
        return RunSpec("bert_full", "bert-base-cased", False)
    if r == "bert_lora":
        return RunSpec("bert_lora", "bert-base-cased", bool(lora_enabled))
    if r == "distil_full":
        return RunSpec("distil_full", "distilbert-base-cased", False)
    raise ValueError("Unknown run. Choose: bert_full | bert_lora | distil_full")


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


def _make_model_tokenizer(
    base_model: str,
    num_labels: int,
    id2label: Dict[int, str],
    label2id: Dict[str, int],
    use_lora: bool,
    lora_cfg: Dict[str, Any],
    gradient_checkpointing: bool,
    model_cfg: Dict[str, Any],
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    cfg = AutoConfig.from_pretrained(base_model)
    for k, v in model_cfg.items():
        if v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.num_labels = int(num_labels)
    cfg.id2label = dict(id2label)
    cfg.label2id = dict(label2id)
    model = AutoModelForTokenClassification.from_pretrained(base_model, config=cfg)

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if use_lora:
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise RuntimeError("peft is not available. Install `peft` to use LoRA.")

        target_modules = lora_cfg.get("target_modules", None)
        if target_modules is None:
            target_modules = ["query", "value"]
        else:
            target_modules = list(target_modules)

        modules_to_save = lora_cfg.get("modules_to_save", ["classifier"])
        modules_to_save = list(modules_to_save)

        cfg = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            target_modules=target_modules,
            modules_to_save=modules_to_save,
        )
        model = get_peft_model(model, cfg)

        if hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    return model, tokenizer


def _subset_if_needed(split, max_samples: Optional[int]):
    if split is None:
        return None
    if max_samples is None:
        return split
    n = int(max_samples)
    if n <= 0:
        return split
    return split.select(range(min(n, len(split))))


def _is_oom(e: BaseException) -> bool:
    s = str(e).lower()
    return "out of memory" in s or "cuda oom" in s


def _set_tf32_enabled(enabled: bool) -> None:
    if not enabled:
        return
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            mm = torch.backends.cuda.matmul
            if hasattr(mm, "fp32_precision"):
                mm.fp32_precision = "tf32"
    except Exception:
        pass


def _backup_existing_run_dir(run_dir: Path, logger) -> None:
    if not run_dir.exists():
        return
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = run_dir.parent / f"{run_dir.name}__bak__{stamp}"
    try:
        run_dir.rename(bak)
        logger.info("Existing run dir moved to: %s", bak)
    except Exception as e:
        logger.warning("Could not backup existing run dir (%s). Will reuse/overwrite. Error: %s", run_dir, e)


def _make_compute_metrics(id2label: Dict[int, str]):
    seqeval = evaluate.load("seqeval")

    def compute(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=-1)
        label_ids = eval_pred.label_ids

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
        return {
            "precision": float(out.get("overall_precision", 0.0)),
            "recall": float(out.get("overall_recall", 0.0)),
            "f1": float(out.get("overall_f1", 0.0)),
            "accuracy": float(out.get("overall_accuracy", 0.0)),
        }

    return compute


def _metric_settings(cfg_train: Dict[str, Any], has_eval: bool) -> Tuple[str, bool]:
    if not has_eval:
        return "eval_loss", False
    metric = str(cfg_train.get("metric_for_best_model", "eval_f1")).strip()
    if not metric.startswith("eval_"):
        metric = f"eval_{metric}"
    if "loss" in metric:
        greater = False
    else:
        greater = bool(cfg_train.get("greater_is_better", True))
    return metric, greater


def _make_training_args(
    run_dir: Path,
    cfg_train: Dict[str, Any],
    fp16: bool,
    logging_steps: int,
    save_total_limit: int,
    num_workers: int,
    seed: int,
    metric_for_best_model: str,
    greater_is_better: bool,
) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__).parameters

    kwargs = dict(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=True,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(logging_steps),
        learning_rate=float(cfg_train["lr"]),
        weight_decay=float(cfg_train["weight_decay"]),
        warmup_ratio=float(cfg_train["warmup_ratio"]),
        per_device_train_batch_size=int(cfg_train["batch_size"]),
        per_device_eval_batch_size=int(cfg_train["eval_batch_size"]),
        gradient_accumulation_steps=int(cfg_train["grad_accum"]),
        num_train_epochs=float(cfg_train["epochs"]),
        fp16=bool(fp16),
        label_smoothing_factor=float(cfg_train["label_smoothing"]),
        lr_scheduler_type=str(cfg_train["lr_scheduler"]),
        max_grad_norm=float(cfg_train["max_grad_norm"]),
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model=str(metric_for_best_model),
        greater_is_better=bool(greater_is_better),
        save_total_limit=int(save_total_limit),
        dataloader_num_workers=int(num_workers),
        optim="adamw_torch",
        seed=int(seed),
        data_seed=int(seed),
    )

    if "group_by_length" in sig:
        kwargs["group_by_length"] = bool(cfg_train.get("group_by_length", False))

    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def _train_once(
    run_dir: Path,
    model,
    tokenizer,
    train_ds,
    eval_ds,
    cfg_train: Dict[str, Any],
    fp16: bool,
    logging_steps: int,
    save_total_limit: int,
    num_workers: int,
    seed: int,
    early_stopping_patience: int,
    early_stopping_threshold: float,
    compute_metrics,
    metric_for_best_model: str,
    greater_is_better: bool,
    logger,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    collator = DataCollatorForTokenClassification(tokenizer)
    training_args = _make_training_args(
        run_dir,
        cfg_train,
        fp16,
        logging_steps,
        save_total_limit,
        num_workers,
        seed,
        metric_for_best_model,
        greater_is_better,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    tr_sig = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in tr_sig:
        trainer_kwargs["tokenizer"] = tokenizer
    if compute_metrics is not None:
        trainer_kwargs["compute_metrics"] = compute_metrics

    callbacks = []
    if eval_ds is not None and int(early_stopping_patience) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(early_stopping_patience),
                early_stopping_threshold=float(early_stopping_threshold),
            )
        )
    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer = Trainer(**trainer_kwargs)

    logger.info("Training start")
    train_res = trainer.train()
    train_metrics = dict(getattr(train_res, "metrics", {}) or {})
    logger.info("Training done")

    eval_metrics: Dict[str, Any] = {}
    if eval_ds is not None:
        logger.info("Eval start")
        eval_metrics = trainer.evaluate(eval_dataset=eval_ds)
        logger.info("Eval done")

    logger.info("Saving model to %s", run_dir)
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    log_history = list(getattr(trainer.state, "log_history", []) or [])
    return train_metrics, eval_metrics, log_history


def train_run(
    run: str = "bert_lora",
    config_path: str = "configs/base.yaml",
    root: str | Path = ".",
    force_cpu: bool = False,
) -> Dict[str, Any]:
    paths = ProjectPaths.from_root(root)
    paths.ensure()

    cfg = read_yaml(Path(config_path)) or {}

    project_cfg = cfg.get("project", {}) if isinstance(cfg.get("project", {}), dict) else {}

    dataset_name = (
        project_cfg.get("dataset")
        or cfg.get("dataset")
        or cfg.get("dataset_name")
        or (cfg.get("data", {}) or {}).get("dataset")
        or (cfg.get("data_prep", {}) or {}).get("dataset")
        or (cfg.get("data_prep", {}) or {}).get("dataset_name")
        or "mrm8488/stackoverflow-ner"
    )
    seed = int(
        project_cfg.get("seed")
        or cfg.get("seed")
        or (cfg.get("train", {}) or {}).get("seed")
        or (cfg.get("data", {}) or {}).get("seed")
        or (cfg.get("data_prep", {}) or {}).get("seed")
        or 13
    )

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    max_length = int(data_cfg.get("max_length", data_cfg.get("max_len", cfg.get("max_length", 128))))
    label_all_tokens = bool(data_cfg.get("label_all_tokens", cfg.get("label_all_tokens", False)))

    tcfg = cfg.get("train", {}) if isinstance(cfg.get("train", {}), dict) else {}
    tcfg = dict(tcfg)

    tcfg.setdefault("epochs", 3)
    tcfg.setdefault("lr", 2e-5)
    tcfg.setdefault("weight_decay", 0.01)
    tcfg.setdefault("warmup_ratio", 0.06)
    tcfg.setdefault("batch_size", 8)
    tcfg.setdefault("eval_batch_size", 16)
    tcfg.setdefault("grad_accum", 1)
    tcfg.setdefault("fp16", True)
    tcfg.setdefault("tf32", True)
    tcfg.setdefault("gradient_checkpointing", True)
    tcfg.setdefault("logging_steps", 50)
    tcfg.setdefault("save_total_limit", 2)
    tcfg.setdefault("dataloader_num_workers", 2)
    tcfg.setdefault("label_smoothing", 0.0)
    tcfg.setdefault("lr_scheduler", "linear")
    tcfg.setdefault("max_grad_norm", 1.0)
    tcfg.setdefault("early_stopping_patience", 0)
    tcfg.setdefault("early_stopping_threshold", 0.0)
    tcfg.setdefault("metric_for_best_model", "eval_f1")
    tcfg.setdefault("greater_is_better", True)
    tcfg.setdefault("group_by_length", False)

    max_train_samples = tcfg.get("max_train_samples", None)
    max_eval_samples = tcfg.get("max_eval_samples", None)

    lcfg = cfg.get("lora", {}) if isinstance(cfg.get("lora", {}), dict) else {}
    lcfg = dict(lcfg)
    mcfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    mcfg = dict(mcfg)

    run_spec = _pick_run(run, bool(lcfg.get("enabled", True)))

    if run_spec.use_lora and (lcfg.get("lr") is not None):
        tcfg["lr"] = float(lcfg["lr"])

    run_dir = paths.models / run_spec.run_name
    logger = setup_logger(run_dir / "run.log", "train")

    _backup_existing_run_dir(run_dir, logger)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    use_cuda = torch.cuda.is_available() and (not force_cpu)
    fp16 = bool(tcfg.get("fp16", True)) and use_cuda
    tf32 = bool(tcfg.get("tf32", True)) and use_cuda
    grad_ckpt = bool(tcfg.get("gradient_checkpointing", True))
    logging_steps = int(tcfg.get("logging_steps", 50))
    save_total_limit = int(tcfg.get("save_total_limit", 2))
    num_workers = int(tcfg.get("dataloader_num_workers", 2))

    logger.info("Run: %s", run_spec.run_name)
    logger.info("Base model: %s", run_spec.base_model)
    logger.info("Use LoRA: %s", run_spec.use_lora)
    logger.info("Dataset: %s", dataset_name)
    logger.info("Seed: %s", seed)
    logger.info("CUDA: %s | fp16: %s | tf32: %s | grad_ckpt: %s", use_cuda, fp16, tf32, grad_ckpt)
    logger.info("LR (effective): %s", tcfg.get("lr"))

    if tf32:
        _set_tf32_enabled(True)

    ds = _load_processed_dataset(paths, str(dataset_name))
    label_names = _load_label_names(paths)
    label2id = {n: i for i, n in enumerate(label_names)}
    id2label = {i: n for i, n in enumerate(label_names)}

    train_split = _subset_if_needed(ds["train"], max_train_samples)
    val_split = ds.get("validation", None) or ds.get("test", None)
    val_split = _subset_if_needed(val_split, max_eval_samples)

    model, tokenizer = _make_model_tokenizer(
        base_model=run_spec.base_model,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        use_lora=run_spec.use_lora,
        lora_cfg=lcfg,
        gradient_checkpointing=grad_ckpt,
        model_cfg=mcfg,
    )

    logger.info("Tokenizing train split: %d samples", len(train_split))
    tokenized_train = train_split.map(
        lambda ex: _tokenize_and_align(ex, tokenizer, label_all_tokens, max_length),
        batched=True,
        remove_columns=train_split.column_names,
    )

    tokenized_val = None
    if val_split is not None:
        logger.info("Tokenizing eval split: %d samples", len(val_split))
        tokenized_val = val_split.map(
            lambda ex: _tokenize_and_align(ex, tokenizer, label_all_tokens, max_length),
            batched=True,
            remove_columns=val_split.column_names,
        )

    compute_metrics = None
    if tokenized_val is not None:
        compute_metrics = _make_compute_metrics(id2label)

    metric_for_best_model, greater_is_better = _metric_settings(tcfg, tokenized_val is not None)

    base_bs = int(tcfg["batch_size"])
    base_acc = int(tcfg["grad_accum"])
    effective = max(1, base_bs * base_acc)
    candidates = [base_bs, max(1, base_bs // 2), max(1, base_bs // 4)]

    resolved = {
        "run": run_spec.run_name,
        "base_model": run_spec.base_model,
        "use_lora": run_spec.use_lora,
        "dataset": str(dataset_name),
        "seed": int(seed),
        "max_length": int(max_length),
        "label_all_tokens": bool(label_all_tokens),
        "num_labels": int(len(label_names)),
        "label_names": label_names,
        "fp16": bool(fp16),
        "tf32": bool(tf32),
        "gradient_checkpointing": bool(grad_ckpt),
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "effective_batch": int(effective),
        "lr_effective": float(tcfg["lr"]),
        "label_smoothing": float(tcfg["label_smoothing"]),
        "lr_scheduler": str(tcfg["lr_scheduler"]),
        "max_grad_norm": float(tcfg["max_grad_norm"]),
        "metric_for_best_model": str(metric_for_best_model),
        "greater_is_better": bool(greater_is_better),
        "early_stopping_patience": int(tcfg.get("early_stopping_patience", 0)),
        "early_stopping_threshold": float(tcfg.get("early_stopping_threshold", 0.0)),
        "group_by_length": bool(tcfg.get("group_by_length", False)),
        "lora": lcfg if run_spec.use_lora else {"enabled": False},
        "model_cfg": mcfg,
    }

    last_err: Optional[BaseException] = None
    train_metrics: Dict[str, Any] = {}
    eval_metrics: Dict[str, Any] = {}
    log_history: List[Dict[str, Any]] = []

    for bs in candidates:
        acc = max(1, int((effective + bs - 1) // bs))
        tcfg_try = dict(tcfg)
        tcfg_try["batch_size"] = int(bs)
        tcfg_try["grad_accum"] = int(acc)

        logger.info("Trying batch_size=%d grad_accum=%d (effective=%d)", bs, acc, bs * acc)

        try:
            train_metrics, eval_metrics, log_history = _train_once(
                run_dir=run_dir,
                model=model,
                tokenizer=tokenizer,
                train_ds=tokenized_train,
                eval_ds=tokenized_val,
                cfg_train=tcfg_try,
                fp16=fp16,
                logging_steps=logging_steps,
                save_total_limit=save_total_limit,
                num_workers=num_workers,
                seed=seed,
                early_stopping_patience=int(tcfg.get("early_stopping_patience", 0)),
                early_stopping_threshold=float(tcfg.get("early_stopping_threshold", 0.0)),
                compute_metrics=compute_metrics,
                metric_for_best_model=metric_for_best_model,
                greater_is_better=greater_is_better,
                logger=logger,
            )
            resolved.update(
                {
                    "batch_size": int(bs),
                    "grad_accum": int(acc),
                    "epochs": float(tcfg_try["epochs"]),
                    "lr": float(tcfg_try["lr"]),
                    "weight_decay": float(tcfg_try["weight_decay"]),
                    "warmup_ratio": float(tcfg_try["warmup_ratio"]),
                    "eval_batch_size": int(tcfg_try["eval_batch_size"]),
                    "dataloader_num_workers": int(num_workers),
                }
            )
            logger.info("Training succeeded with batch_size=%d grad_accum=%d", bs, acc)
            last_err = None
            break
        except RuntimeError as e:
            last_err = e
            if use_cuda and _is_oom(e):
                logger.warning("OOM detected. Retrying with smaller batch size.")
                torch.cuda.empty_cache()
                continue
            logger.exception("Training failed")
            raise

    if last_err is not None and not train_metrics:
        raise RuntimeError(f"Training failed after OOM retries: {last_err}") from last_err

    write_json(run_dir / "config_resolved.json", resolved)
    write_json(run_dir / "metrics_train.json", {"train": train_metrics, "eval": eval_metrics})
    write_json(run_dir / "trainer_log_history.json", {"log_history": log_history})

    logger.info("Done")
    return {
        "run": run_spec.run_name,
        "run_dir": str(run_dir),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "resolved": resolved,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="bert_lora", choices=["bert_full", "bert_lora", "distil_full"])
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()
    train_run(run=args.run, config_path=args.config, root=".", force_cpu=bool(args.force_cpu))


if __name__ == "__main__":
    main()
