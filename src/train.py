# src/train.py
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

import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
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
):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=int(num_labels),
        id2label=id2label,
        label2id=label2id,
    )

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if use_lora:
        if get_peft_model is None or LoraConfig is None or TaskType is None:
            raise RuntimeError("peft is not available. Install `peft` to use LoRA.")

        target_modules = lora_cfg.get("target_modules", None)
        if target_modules is None:
            target_modules = ["query", "value"]

        modules_to_save = lora_cfg.get("modules_to_save", ["classifier"])

        cfg = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            bias=str(lora_cfg.get("bias", "none")),
            target_modules=list(target_modules),
            modules_to_save=list(modules_to_save),
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

    # New torch API (preferred) – avoids deprecation warnings from old flags.
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            mm = torch.backends.cuda.matmul
            if hasattr(mm, "fp32_precision"):
                mm.fp32_precision = "tf32"
    except Exception:
        pass


def _backup_existing_run_dir(run_dir: Path, logger) -> None:
    """
    Prevent losing previous results when re-running the same `run_name`.
    Moves existing `models/<run>/` to `models/<run>__bak__YYYYmmdd_HHMMSS`.
    """
    if not run_dir.exists():
        return
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = run_dir.parent / f"{run_dir.name}__bak__{stamp}"
    try:
        run_dir.rename(bak)
        logger.info("Existing run dir moved to: %s", bak)
    except Exception as e:
        logger.warning("Could not backup existing run dir (%s). Will reuse/overwrite. Error: %s", run_dir, e)


def _make_training_args(
    run_dir: Path,
    cfg_train: Dict[str, Any],
    fp16: bool,
    logging_steps: int,
    save_total_limit: int,
    num_workers: int,
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
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=int(save_total_limit),
        dataloader_num_workers=int(num_workers),
        optim="adamw_torch",
    )

    # transformers compatibility: evaluation_strategy -> eval_strategy
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
    logger,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    collator = DataCollatorForTokenClassification(tokenizer)
    training_args = _make_training_args(run_dir, cfg_train, fp16, logging_steps, save_total_limit, num_workers)

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    # transformers compatibility: some versions deprecate/rename tokenizer arg
    tr_sig = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in tr_sig:
        trainer_kwargs["tokenizer"] = tokenizer

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

    cfg = read_yaml(Path(config_path))
    dataset_name = str(cfg["project"]["dataset"])
    seed = int(cfg["project"]["seed"])

    max_length = int(cfg["data"]["max_length"])
    label_all_tokens = bool(cfg["data"]["label_all_tokens"])

    tcfg = dict(cfg["train"])
    max_train_samples = tcfg.get("max_train_samples", None)
    max_eval_samples = tcfg.get("max_eval_samples", None)

    lcfg = dict(cfg.get("lora", {}))
    run_spec = _pick_run(run, bool(lcfg.get("enabled", True)))

    run_dir = paths.models / run_spec.run_name
    logger = setup_logger(run_dir / "run.log", "train")

    # IMPORTANT: backup before overwrite, so your previous 3-epoch run doesn't disappear
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
    logger.info("CUDA: %s | fp16: %s | tf32: %s | grad_ckpt: %s", use_cuda, fp16, tf32, grad_ckpt)

    if tf32:
        _set_tf32_enabled(True)

    ds = _load_processed_dataset(paths, dataset_name)
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

    base_bs = int(tcfg["batch_size"])
    base_acc = int(tcfg["grad_accum"])
    effective = base_bs * base_acc
    candidates = [base_bs, max(1, base_bs // 2), max(1, base_bs // 4)]

    resolved = {
        "run": run_spec.run_name,
        "base_model": run_spec.base_model,
        "use_lora": run_spec.use_lora,
        "dataset": dataset_name,
        "seed": seed,
        "max_length": max_length,
        "label_all_tokens": label_all_tokens,
        "num_labels": len(label_names),
        "label_names": label_names,
        "fp16": fp16,
        "tf32": tf32,
        "gradient_checkpointing": grad_ckpt,
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "effective_batch": effective,
        "lora": lcfg if run_spec.use_lora else {"enabled": False},
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
