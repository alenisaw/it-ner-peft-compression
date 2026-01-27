# src/data.py
"""
IT-domain NER dataset prep:
- Gold dataset: HF NER dataset (e.g., mrm8488/stackoverflow-ner) -> mapped to 5-entity BIO schema
- Weak corpus (optional): large IT text (e.g., GitHub issues) -> pseudo-labeling via regex/gazetteers -> mixed into train
- Oversampling (optional): increases entity density in training
Outputs:
- data/raw/<key>/, data/processed/<key>/
- data/processed_meta.json, data/dataset_meta.json, data/prepare.log
"""

from __future__ import annotations
import os
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from src.paths import ProjectPaths, ds_key, setup_logger, write_json

Tag = Union[int, str]


def load_raw_dataset(dataset_name: str) -> DatasetDict:
    ds = load_dataset(dataset_name)
    if not isinstance(ds, DatasetDict):
        raise TypeError("Expected DatasetDict with train/validation/test splits.")
    return ds


def save_dataset(ds: DatasetDict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))


def load_saved_dataset(dir_path: Path) -> DatasetDict:
    return load_from_disk(str(dir_path))


def _detect_tag_mode(ds_raw: DatasetDict) -> str:
    ex = ds_raw["train"][0]
    tags = ex.get("ner_tags")
    if not tags:
        return "unknown"
    t0 = tags[0]
    if isinstance(t0, int):
        return "int"
    if isinstance(t0, str):
        return "str"
    return "unknown"


def _infer_old_label_names(ds_raw: DatasetDict) -> Optional[List[str]]:
    ner_feat = ds_raw["train"].features.get("ner_tags", None)
    if ner_feat is None:
        raise RuntimeError("Dataset does not have 'ner_tags' feature.")
    feat = getattr(ner_feat, "feature", None)
    if feat is None:
        feat = ner_feat
    names = getattr(feat, "names", None)
    if names is None:
        return None
    return [str(x) for x in names]


def _new_label_vocab() -> List[str]:
    return [
        "O",
        "B-ERROR", "B-HARDWARE", "B-OS", "B-SOFTWARE", "B-VERSION",
        "I-ERROR", "I-HARDWARE", "I-OS", "I-SOFTWARE", "I-VERSION",
    ]


def build_entity_map_from_labels(labels: List[str]) -> Dict[str, str]:
    software_kw = {
        "application", "app", "software", "tool", "library", "framework", "package",
        "language", "programminglanguage", "ide", "editor", "browser",
        "database", "db", "sql", "nosql",
        "service", "cloud", "aws", "azure", "gcp",
        "kubernetes", "docker", "container",
        "api", "sdk", "platform", "runtime",
        "server", "client",
        "protocol", "http", "https", "tcp", "udp", "ssh", "ssl", "tls",
        "repo", "repository", "git", "github",
    }
    os_kw = {"operatingsystem", "os", "windows", "linux", "ubuntu", "debian", "macos", "android", "ios"}
    hardware_kw = {"hardware", "device", "router", "switch", "cpu", "gpu", "disk", "ssd", "hdd", "ram", "laptop"}
    error_kw = {"error", "exception", "bug", "crash", "failure", "timeout", "stacktrace", "traceback"}
    version_kw = {"version", "release", "update", "patch", "build", "kb", "rc"}

    def norm(s: str) -> str:
        return s.strip().lower().replace("_", "").replace("-", "").replace(" ", "")

    def choose_entity(base: str) -> str:
        b = norm(base)
        if any(k in b for k in os_kw):
            return "OS"
        if any(k in b for k in error_kw):
            return "ERROR"
        if any(k in b for k in version_kw):
            return "VERSION"
        if any(k in b for k in hardware_kw):
            return "HARDWARE"
        if any(k in b for k in software_kw):
            return "SOFTWARE"
        return "O"

    mapping: Dict[str, str] = {}
    for lab in labels:
        lab = str(lab)
        if lab == "O":
            mapping[lab] = "O"
            continue
        if lab.startswith(("B-", "I-")) and len(lab) > 2:
            pref = lab[:2]
            base = lab[2:]
            ent = choose_entity(base)
            mapping[lab] = "O" if ent == "O" else f"{pref}{ent}"
        else:
            ent = choose_entity(lab)
            mapping[lab] = "O" if ent == "O" else f"B-{ent}"
    return mapping


def remap_ner_tags(
    ds_raw: DatasetDict,
    mapping: Dict[str, str],
    old_label_names: Optional[List[str]],
    tag_mode: str,
) -> Tuple[DatasetDict, List[str], Dict[str, Any]]:
    new_label_names = _new_label_vocab()
    new_label2id = {n: i for i, n in enumerate(new_label_names)}
    coverage: Dict[str, Any] = {
        "mode": tag_mode,
        "mapped_count": 0,
        "total_count": 0,
        "coverage_ratio": 0.0,
        "mapped_label_names": [],
        "unmapped_label_names": [],
    }

    if tag_mode == "int":
        if not old_label_names:
            raise RuntimeError("int ner_tags but missing label names. Use force_redownload=True.")
        old_id2name = {i: n for i, n in enumerate(old_label_names)}

        def _map_example(ex):
            out = []
            for t in ex["ner_tags"]:
                old = old_id2name.get(int(t), "O")
                mapped = mapping.get(old, "O")
                out.append(new_label2id.get(mapped, 0))
            ex["ner_tags"] = out
            return ex

        ds2 = ds_raw.map(_map_example)

        uniq = old_label_names[:]
        mapped = [n for n in uniq if mapping.get(n, "O") != "O"]
        unmapped = [n for n in uniq if mapping.get(n, "O") == "O" and n != "O"]
        coverage["mapped_label_names"] = mapped
        coverage["unmapped_label_names"] = unmapped
        coverage["mapped_count"] = int(len(mapped))
        coverage["total_count"] = int(len(uniq))
        coverage["coverage_ratio"] = float(len(mapped) / max(1, len(uniq)))
        return ds2, new_label_names, coverage

    if tag_mode == "str":
        uniq = set()
        N = min(50000, len(ds_raw["train"]))
        for ex in ds_raw["train"].select(range(N)):
            for lab in ex["ner_tags"]:
                uniq.add(str(lab))
        uniq_list = sorted(uniq)

        def _map_example(ex):
            out = []
            for lab in ex["ner_tags"]:
                mapped = mapping.get(str(lab), "O")
                out.append(new_label2id.get(mapped, 0))
            ex["ner_tags"] = out
            return ex

        ds2 = ds_raw.map(_map_example)

        mapped = [n for n in uniq_list if mapping.get(n, "O") != "O"]
        unmapped = [n for n in uniq_list if mapping.get(n, "O") == "O" and n != "O"]
        coverage["mapped_label_names"] = mapped
        coverage["unmapped_label_names"] = unmapped
        coverage["mapped_count"] = int(len(mapped))
        coverage["total_count"] = int(len(uniq_list))
        coverage["coverage_ratio"] = float(len(mapped) / max(1, len(uniq_list)))
        return ds2, new_label_names, coverage

    raise RuntimeError(f"Unsupported tag_mode={tag_mode}")


def _length_stats(ds_split) -> Dict[str, float]:
    lens = [len(x) for x in ds_split["tokens"]]
    if not lens:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.mean(lens)),
        "median": float(statistics.median(lens)),
        "min": float(min(lens)),
        "max": float(max(lens)),
    }


def _label_counts(ds_split, label_names: List[str]) -> Dict[str, int]:
    counts = {n: 0 for n in label_names}
    for seq in ds_split["ner_tags"]:
        for t in seq:
            counts[label_names[int(t)]] += 1
    return counts


def _bio_issues(ds_split, label_names: List[str]) -> Dict[str, int]:
    issues = 0
    total = 0
    for seq in ds_split["ner_tags"]:
        prev_type = "O"
        for t in seq:
            total += 1
            lab = label_names[int(t)]
            if lab.startswith("I-"):
                cur_type = lab[2:]
                if prev_type != cur_type:
                    issues += 1
                prev_type = cur_type
            elif lab.startswith("B-"):
                prev_type = lab[2:]
            else:
                prev_type = "O"
    return {"i_without_prev_same_entity": int(issues), "total_tags": int(total)}


def _has_any_entity(example) -> bool:
    return any(int(t) != 0 for t in example["ner_tags"])


def _has_entity_type(example, label_names: List[str], entity: str) -> bool:
    for t in example["ner_tags"]:
        if label_names[int(t)].endswith(entity):
            return True
    return False


def oversample_train(
    ds_proc: DatasetDict,
    label_names: List[str],
    factor: float = 1.0,
    rare_boost: float = 0.0,
    seed: int = 13,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    train = ds_proc["train"]
    if factor <= 1.0 and rare_boost <= 0.0:
        return ds_proc, {"enabled": False}

    train_ent = train.filter(_has_any_entity)
    if len(train_ent) == 0:
        return ds_proc, {"enabled": False, "reason": "no entity examples found"}

    extra_n = int(max(0, (factor - 1.0) * len(train)))
    extra = train_ent.shuffle(seed=seed).select(range(min(extra_n, len(train_ent))))
    merged = concatenate_datasets([train, extra]) if extra_n > 0 else train

    rare_meta = {"enabled": False}
    if rare_boost > 0.0:
        counts = _label_counts(train, label_names)
        ent_types = ["SOFTWARE", "OS", "HARDWARE", "ERROR", "VERSION"]
        ent_token_counts = {e: 0 for e in ent_types}
        for lab, c in counts.items():
            for e in ent_types:
                if lab.endswith(e):
                    ent_token_counts[e] += int(c)
        rare_sorted = sorted(ent_types, key=lambda e: ent_token_counts[e])
        rare = rare_sorted[:2]
        pools = []
        for e in rare:
            pool = train.filter(lambda ex, e=e: _has_entity_type(ex, label_names, e))
            if len(pool) > 0:
                pools.append(pool)
        if pools:
            pool_all = concatenate_datasets(pools)
            rare_n = int(rare_boost * len(train) * 0.2)
            rare_extra = pool_all.shuffle(seed=seed + 1).select(range(min(rare_n, len(pool_all))))
            merged = concatenate_datasets([merged, rare_extra])
            rare_meta = {"enabled": True, "rare_entities": rare, "rare_extra": int(len(rare_extra))}
        else:
            rare_meta = {"enabled": False, "reason": "no rare pools"}

    ds_new = DatasetDict({**ds_proc, "train": merged})
    meta = {
        "enabled": True,
        "factor": float(factor),
        "rare_boost": float(rare_boost),
        "train_before": int(len(train)),
        "train_entity_pool": int(len(train_ent)),
        "extra_added": int(len(merged) - len(train)),
        "train_after": int(len(merged)),
        "rare_meta": rare_meta,
    }
    return ds_new, meta


def _pick_text_fields(example: Dict[str, Any]) -> str:
    candidates = ["text", "body", "content", "comment", "title", "message", "description"]
    parts: List[str] = []
    for k in candidates:
        v = example.get(k, None)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    if parts:
        return "\n".join(parts)
    for k, v in example.items():
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _simple_tokenize(text: str, max_len: int = 128) -> List[str]:
    toks = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return toks[:max_len]


def _weak_pseudo_label(tokens: List[str]) -> List[str]:
    """
    Weak labeling (pseudo-labeling) for IT-domain NER using regex + gazetteers.

    Notes:
    - This is intentionally conservative: it aims for high precision, not recall.
    - Supports multi-token phrases (e.g., "Windows 11", "Visual Studio Code", "Google Chrome")
      by tagging them as a BIO span.
    """
    os_terms = {
        "windows", "win10", "win11", "ubuntu", "debian", "linux", "macos", "android", "ios",
    }
    hw_terms = {
        "router", "switch", "server", "laptop", "cpu", "gpu", "ssd", "hdd", "ram", "keyboard", "mouse",
    }
    sw_terms = {
        "docker", "kubernetes", "chrome", "firefox", "nginx", "apache", "postgres", "mysql", "mongodb",
        "redis", "node", "nodejs", "python", "java", "dotnet", ".net", "git", "github", "gitlab",
        "jira", "slack", "vpn", "vscode", "powershell",
    }
    err_terms = {"error", "exception", "crash", "failed", "failure", "timeout", "traceback", "stacktrace", "bug"}

    # multi-token phrases (lowercased) -> entity
    phrase_map = {
        ("windows", "11"): "OS",
        ("windows", "10"): "OS",
        ("visual", "studio", "code"): "SOFTWARE",
        ("vs", "code"): "SOFTWARE",
        ("google", "chrome"): "SOFTWARE",
        ("microsoft", "edge"): "SOFTWARE",
    }

    ver_re = re.compile(r"^(v)?\d+(\.\d+){1,4}([a-z0-9\-]+)?$", re.IGNORECASE)
    kb_re = re.compile(r"^KB\d+$", re.IGNORECASE)

    labels = ["O"] * len(tokens)

    def can_write(i: int) -> bool:
        return 0 <= i < len(labels) and labels[i] == "O"

    def mark_span(i0: int, i1: int, ent: str) -> None:
        """Mark [i0, i1) as BIO span, if all tokens are currently 'O'."""
        if i0 < 0 or i1 > len(tokens) or i0 >= i1:
            return
        if any(labels[i] != "O" for i in range(i0, i1)):
            return
        labels[i0] = f"B-{ent}"
        for i in range(i0 + 1, i1):
            labels[i] = f"I-{ent}"

    # 1) Multi-token phrases first (longer first to avoid partial overlaps)
    max_phrase_len = max((len(k) for k in phrase_map.keys()), default=0)
    toks_l = [t.lower() for t in tokens]
    for n in range(max_phrase_len, 1, -1):
        for i in range(0, len(tokens) - n + 1):
            key = tuple(toks_l[i : i + n])
            ent = phrase_map.get(key)
            if ent:
                mark_span(i, i + n, ent)

    # 2) Single-token heuristics
    for i, t in enumerate(tokens):
        tl = t.lower()

        # Skip if already labeled by a phrase
        if labels[i] != "O":
            continue

        # VERSION-like tokens
        if kb_re.match(t) or ver_re.match(tl) or tl in {"rc1", "rc2", "beta", "alpha"}:
            mark_span(i, i + 1, "VERSION")
            continue

        # OS / hardware / software / error
        if tl in os_terms:
            mark_span(i, i + 1, "OS")
            continue
        if tl in hw_terms:
            mark_span(i, i + 1, "HARDWARE")
            continue
        if tl in err_terms or tl.endswith("error") or tl.endswith("exception"):
            mark_span(i, i + 1, "ERROR")
            continue
        if tl in sw_terms:
            mark_span(i, i + 1, "SOFTWARE")
            continue

    # 3) Simple pattern: "<OS> <number>" if both are still O (e.g., "Windows 11")
    for i in range(len(tokens) - 1):
        if labels[i] != "O" or labels[i + 1] != "O":
            continue
        if toks_l[i] == "windows" and re.fullmatch(r"\d{1,2}", toks_l[i + 1]):
            mark_span(i, i + 2, "OS")

    return labels


def _bio_to_ids(bio: List[str], label2id: Dict[str, int]) -> List[int]:
    out: List[int] = []
    for lab in bio:
        out.append(int(label2id.get(lab, 0)))
    return out


def build_weak_dataset(
    weak_dataset_name: str,
    sample_n: int,
    max_len: int,
    seed: int,
) -> Dataset:
    env_token = os.getenv("HF_TOKEN", "").strip()
    token_arg = env_token if env_token else True

    try:
        ds = load_dataset(
            weak_dataset_name,
            split="train",
            streaming=True,
            token=token_arg,
        )
    except Exception as e:
        msg = str(e).lower()
        if ("gated" in msg) or ("must be authenticated" in msg) or ("403" in msg):
            raise RuntimeError(
                f"Cannot access gated dataset '{weak_dataset_name}'. "
                f"Login with `huggingface-cli login` or set HF_TOKEN env variable."
            ) from e
        raise

    items_tokens: List[List[str]] = []
    items_tags: List[List[int]] = []

    label_names = _new_label_vocab()
    label2id = {n: i for i, n in enumerate(label_names)}

    for ex in ds.shuffle(seed=seed).take(sample_n):
        text = _pick_text_fields(ex)
        if not text:
            continue
        toks = _simple_tokenize(text, max_len=max_len)
        if len(toks) < 5:
            continue
        bio = _weak_pseudo_label(toks)
        tag_ids = _bio_to_ids(bio, label2id)
        if all(t == 0 for t in tag_ids):
            continue
        items_tokens.append(toks)
        items_tags.append(tag_ids)

    return Dataset.from_dict({"tokens": items_tokens, "ner_tags": items_tags})


def mix_gold_and_weak(
    gold_train: Dataset,
    weak_ds: Dataset,
    weak_ratio: float,
    seed: int,
) -> Tuple[Dataset, Dict[str, Any]]:
    if weak_ratio <= 0.0 or len(weak_ds) == 0:
        return gold_train, {"enabled": False}

    target_weak = int(max(1, weak_ratio * len(gold_train)))
    use_n = min(target_weak, len(weak_ds))
    weak_sel = weak_ds.shuffle(seed=seed).select(range(use_n))
    mixed = concatenate_datasets([gold_train, weak_sel])
    meta = {"enabled": True, "weak_ratio": float(weak_ratio), "weak_used": int(use_n), "gold_train": int(len(gold_train)), "train_after_mix": int(len(mixed))}
    return mixed, meta


def prepare_dataset(
    root: str | Path = ".",
    dataset_name: str = "mrm8488/stackoverflow-ner",
    force_redownload: bool = False,
    oversample_factor: float = 1.0,
    rare_boost: float = 0.0,
    seed: int = 13,
    weak_dataset_name: Optional[str] = None,
    weak_sample_n: int = 50000,
    weak_max_len: int = 128,
    weak_ratio: float = 0.5,
) -> Dict[str, Any]:
    paths = ProjectPaths.from_root(root)
    paths.ensure()

    logger = setup_logger(paths.data / "prepare.log", "data_prep")
    logger.info("Starting dataset preparation")
    logger.info("Dataset: %s", dataset_name)

    key = ds_key(dataset_name)
    raw_path = paths.raw_data / key
    processed_path = paths.processed_data / key

    loaded_from_cache = False
    if raw_path.exists() and not force_redownload:
        logger.info("Loading cached raw dataset: %s", raw_path)
        ds_raw = load_saved_dataset(raw_path)
        loaded_from_cache = True
    else:
        logger.info("Downloading dataset from HuggingFace")
        ds_raw = load_raw_dataset(dataset_name)
        logger.info("Saving raw dataset to: %s", raw_path)
        save_dataset(ds_raw, raw_path)

    tag_mode = _detect_tag_mode(ds_raw)
    old_label_names = _infer_old_label_names(ds_raw)

    if tag_mode == "int" and old_label_names is None and loaded_from_cache:
        logger.warning("Cached dataset has int tags but missing ClassLabel names -> refreshing cache once")
        ds_raw = load_raw_dataset(dataset_name)
        old_label_names = _infer_old_label_names(ds_raw)
        tag_mode = _detect_tag_mode(ds_raw)
        logger.info("Overwriting raw cache at: %s", raw_path)
        save_dataset(ds_raw, raw_path)

    if tag_mode not in {"int", "str"}:
        raise RuntimeError(f"Unsupported ner_tags type (tag_mode={tag_mode}).")

    if tag_mode == "int":
        if not old_label_names:
            raise RuntimeError("int ner_tags but still no label names. Use force_redownload=True.")
        labels_for_map = old_label_names
    else:
        uniq = set()
        N = min(50000, len(ds_raw["train"]))
        for ex in ds_raw["train"].select(range(N)):
            for lab in ex["ner_tags"]:
                uniq.add(str(lab))
        labels_for_map = sorted(uniq)

    mapping = build_entity_map_from_labels(labels_for_map)

    write_json(
        paths.data / "dataset_meta.json",
        {
            "dataset": dataset_name,
            "tag_mode": tag_mode,
            "original_num_labels": (len(old_label_names) if old_label_names else None),
            "original_label_names": old_label_names,
            "labels_for_map_count": int(len(labels_for_map)),
        },
    )

    logger.info("Remapping labels to 5-entity schema (mode=%s)", tag_mode)
    ds_proc, new_label_names, coverage = remap_ner_tags(ds_raw, mapping, old_label_names, tag_mode)

    weak_meta = {"enabled": False}
    if weak_dataset_name:
        logger.info("Building weak dataset: %s (sample_n=%d)", weak_dataset_name, int(weak_sample_n))
        weak_ds = build_weak_dataset(
            weak_dataset_name=weak_dataset_name,
            sample_n=int(weak_sample_n),
            max_len=int(weak_max_len),
            seed=int(seed),
        )
        mixed_train, weak_meta = mix_gold_and_weak(
            gold_train=ds_proc["train"],
            weak_ds=weak_ds,
            weak_ratio=float(weak_ratio),
            seed=int(seed),
        )
        ds_proc = DatasetDict({**ds_proc, "train": mixed_train})

    ds_proc, oversample_meta = oversample_train(
        ds_proc=ds_proc,
        label_names=new_label_names,
        factor=float(oversample_factor),
        rare_boost=float(rare_boost),
        seed=int(seed),
    )

    eda = {
        "splits": {k: int(len(v)) for k, v in ds_proc.items()},
        "train_token_length": _length_stats(ds_proc["train"]),
        "train_label_counts": _label_counts(ds_proc["train"], new_label_names),
        "bio_warnings_train": _bio_issues(ds_proc["train"], new_label_names),
    }

    processed_meta = {
        "dataset": dataset_name,
        "tag_mode": tag_mode,
        "new_label_names": new_label_names,
        "new_num_labels": int(len(new_label_names)),
        "mapping_coverage": coverage,
        "weak_data": weak_meta,
        "oversampling": oversample_meta,
        "eda": eda,
        "domain_adaptation": {
            "target_domain": "IT support / software engineering text",
            "task": "domain-specific NER with compact schema (5 entities)",
        },
    }

    write_json(paths.data / "processed_meta.json", processed_meta)

    logger.info("Saving processed dataset to: %s", processed_path)
    save_dataset(ds_proc, processed_path)
    logger.info("Done")

    ratio = float(coverage.get("coverage_ratio", 0.0))
    if ratio < 0.5:
        logger.warning("Low mapping coverage (ratio=%.3f). Consider expanding keyword rules.", ratio)

    return processed_meta


def main() -> None:
    prepare_dataset(root=".")
    print("Dataset prepared.")


if __name__ == "__main__":
    main()
