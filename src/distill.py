# src/distill.py
"""
Knowledge distillation for IT-domain NER (token classification).

Implements:
- CE loss to gold labels
- KD loss via KL(student || teacher) on logits with temperature

Outputs under `models/distil_distill/`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import inspect
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
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
    from peft import PeftModel
except ImportError:
    PeftModel = None


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


def _subset_if_needed(split, max_samples: Optional[int]):
    if split is None:
        return None
    if max_samples is None:
        return split
    n = int(max_samples)
    if n <= 0:
        return split
    return split.select(range(min(n, len(split))))


def _make_training_args(
    run_dir: Path,
    cfg: Dict[str, Any],
    fp16: bool,
    logging_steps: int,
    save_total_limit: int,
    num_workers: int,
) -> TrainingArguments:
    sig = inspect.signature(TrainingArguments.__init__).parameters

    kwargs: Dict[str, Any] = dict(
        output_dir=str(run_dir / "checkpoints"),
        overwrite_output_dir=True,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(logging_steps),
        learning_rate=float(cfg["lr"]),
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["eval_batch_size"]),
        num_train_epochs=float(cfg["epochs"]),
        fp16=bool(fp16),
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=int(save_total_limit),
        dataloader_num_workers=int(num_workers),
        optim="adamw_torch",
    )

    if "evaluation_strategy" in sig:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


class DistillTrainer(Trainer):
    def __init__(
        self,
        *args,
        teacher=None,
        temperature: float = 2.0,
        alpha_ce: float = 0.5,
        alpha_kd: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = float(temperature)
        self.alpha_ce = float(alpha_ce)
        self.alpha_kd = float(alpha_kd)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits_s = outputs.logits

        loss_ce = None
        if labels is not None:
            loss_ce = F.cross_entropy(
                logits_s.view(-1, logits_s.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        loss_kd = None
        if self.teacher is not None:
            with torch.no_grad():
                out_t = self.teacher(**{k: v for k, v in inputs.items() if k != "labels"})
                logits_t = out_t.logits

            T = self.temperature
            log_p_s = F.log_softmax(logits_s / T, dim=-1)
            p_t = F.softmax(logits_t / T, dim=-1)

            if labels is not None:
                mask = (labels != -100).unsqueeze(-1)
                log_p_s = log_p_s * mask
                p_t = p_t * mask
                denom = mask.sum().clamp_min(1.0)
            else:
                denom = float(logits_s.shape[0] * logits_s.shape[1])

            loss_kd = (F.kl_div(log_p_s, p_t, reduction="sum") / denom) * (T * T)

        if loss_ce is None and loss_kd is None:
            loss = outputs.loss if getattr(outputs, "loss", None) is not None else None
        elif loss_ce is None:
            loss = loss_kd
        elif loss_kd is None:
            loss = loss_ce
        else:
            loss = self.alpha_ce * loss_ce + self.alpha_kd * loss_kd

        return (loss, outputs) if return_outputs else loss


def _load_teacher(paths: ProjectPaths, teacher_run: str, logger):
    tdir = paths.models / teacher_run
    if not tdir.exists():
        raise FileNotFoundError(f"Teacher run dir not found: {tdir}")

    adapter_cfg = tdir / "adapter_config.json"
    if adapter_cfg.exists() and PeftModel is not None:
        base = read_json(tdir / "config.json")
        base_name = base.get("_name_or_path") or base.get("model_type") or None
        if base_name is None:
            raise RuntimeError("Could not infer base model name for PEFT teacher.")
        logger.info("Loading PEFT teacher: base=%s adapter=%s", base_name, tdir)
        teacher = AutoModelForTokenClassification.from_pretrained(base_name)
        teacher = PeftModel.from_pretrained(teacher, str(tdir))
        return teacher

    logger.info("Loading teacher from %s", tdir)
    return AutoModelForTokenClassification.from_pretrained(str(tdir))


def distill_run(
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

    dcfg = dict(cfg.get("distill", {}))
    if not bool(dcfg.get("enabled", True)):
        raise RuntimeError("distill.enabled is False")

    teacher_run = str(dcfg.get("teacher_run", "bert_full"))
    student_base = str(dcfg.get("student_base_model", "distilbert-base-cased"))
    temperature = float(dcfg.get("temperature", 2.0))
    alpha_ce = float(dcfg.get("alpha_ce", 0.5))
    alpha_kd = float(dcfg.get("alpha_kd", 0.5))

    set_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    use_cuda = torch.cuda.is_available() and (not force_cpu)
    fp16 = bool(cfg.get("train", {}).get("fp16", True)) and use_cuda
    tf32 = bool(cfg.get("train", {}).get("tf32", True)) and use_cuda
    logging_steps = int(cfg.get("train", {}).get("logging_steps", 50))
    save_total_limit = int(cfg.get("train", {}).get("save_total_limit", 2))
    num_workers = int(cfg.get("train", {}).get("dataloader_num_workers", 2))

    run_dir = paths.models / "distil_distill"
    logger = setup_logger(run_dir / "run.log", "distill")
    _backup_existing_run_dir(run_dir, logger)
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset: %s", dataset_name)
    logger.info("Teacher run: %s", teacher_run)
    logger.info("Student base: %s", student_base)
    logger.info("CUDA: %s | fp16: %s | tf32: %s", use_cuda, fp16, tf32)
    logger.info("KD: T=%.2f alpha_ce=%.2f alpha_kd=%.2f", temperature, alpha_ce, alpha_kd)

    if tf32:
        _set_tf32_enabled(True)

    ds = _load_processed_dataset(paths, dataset_name)
    label_names = _load_label_names(paths)
    label2id = {n: i for i, n in enumerate(label_names)}
    id2label = {i: n for i, n in enumerate(label_names)}

    train_split = _subset_if_needed(ds["train"], dcfg.get("max_train_samples", None))
    val_split = ds.get("validation", None) or ds.get("test", None)
    val_split = _subset_if_needed(val_split, dcfg.get("max_eval_samples", None))

    tokenizer = AutoTokenizer.from_pretrained(student_base, use_fast=True)
    student = AutoModelForTokenClassification.from_pretrained(
        student_base,
        num_labels=int(len(label_names)),
        id2label=id2label,
        label2id=label2id,
    )

    teacher = _load_teacher(paths, teacher_run, logger)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

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

    training_args = _make_training_args(
        run_dir=run_dir,
        cfg={
            "lr": float(dcfg.get("lr", 3e-5)),
            "batch_size": int(dcfg.get("batch_size", 16)),
            "eval_batch_size": int(dcfg.get("eval_batch_size", 32)),
            "epochs": float(dcfg.get("epochs", 5)),
        },
        fp16=fp16,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        num_workers=num_workers,
    )

    collator = DataCollatorForTokenClassification(tokenizer)

    trainer_kwargs = dict(
        model=student,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=collator,
    )
    tr_sig = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in tr_sig:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = DistillTrainer(
        **trainer_kwargs,
        teacher=teacher,
        temperature=temperature,
        alpha_ce=alpha_ce,
        alpha_kd=alpha_kd,
    )

    logger.info("Training start")
    train_res = trainer.train()
    train_metrics = dict(getattr(train_res, "metrics", {}) or {})
    logger.info("Training done")

    eval_metrics: Dict[str, Any] = {}
    if tokenized_val is not None:
        logger.info("Eval start")
        eval_metrics = trainer.evaluate(eval_dataset=tokenized_val)
        logger.info("Eval done")

    logger.info("Saving student model to %s", run_dir)
    trainer.save_model(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    resolved = {
        "dataset": dataset_name,
        "seed": seed,
        "teacher_run": teacher_run,
        "student_base_model": student_base,
        "temperature": temperature,
        "alpha_ce": alpha_ce,
        "alpha_kd": alpha_kd,
        "epochs": float(dcfg.get("epochs", 5)),
        "lr": float(dcfg.get("lr", 3e-5)),
        "batch_size": int(dcfg.get("batch_size", 16)),
        "eval_batch_size": int(dcfg.get("eval_batch_size", 32)),
        "fp16": fp16,
        "tf32": tf32,
    }

    write_json(run_dir / "config_resolved.json", resolved)
    write_json(run_dir / "metrics_train.json", {"train": train_metrics, "eval": eval_metrics})
    write_json(run_dir / "trainer_log_history.json", {"log_history": list(getattr(trainer.state, "log_history", []) or [])})

    logger.info("Done")
    return {"run": "distil_distill", "run_dir": str(run_dir), "train_metrics": train_metrics, "eval_metrics": eval_metrics, "resolved": resolved}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()
    distill_run(config_path=args.config, root=".", force_cpu=bool(args.force_cpu))


if __name__ == "__main__":
    main()
