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


def _get_feature_label_names(ds_raw: DatasetDict, key: str) -> Optional[List[str]]:
    feats = ds_raw["train"].features
    if key not in feats:
        return None
    ner_feat = feats.get(key)
    feat = getattr(ner_feat, "feature", None)
    if feat is None:
        feat = ner_feat
    names = getattr(feat, "names", None)
    if not names:
        return None
    return [str(x) for x in names]


def _infer_old_label_names(ds_raw: DatasetDict) -> Optional[List[str]]:
    names = _get_feature_label_names(ds_raw, "ner_tags")
    if names:
        return names
    for k in ("tags", "labels", "label_ids"):
        names = _get_feature_label_names(ds_raw, k)
        if names:
            return names
    return None


def _infer_old_label_names_fallback(
    ds_raw: DatasetDict,
    tag_mode: str,
    max_scan: int = 2000,
) -> Optional[List[str]]:
    if tag_mode == "str":
        uniq = set()
        n = min(max_scan, len(ds_raw["train"]))
        for i in range(n):
            tags = ds_raw["train"][i].get("ner_tags") or []
            for t in tags:
                if isinstance(t, str) and t.strip():
                    uniq.add(t.strip())
        if uniq:
            return sorted(uniq)

    if tag_mode == "int":
        max_id = None
        n = min(max_scan, len(ds_raw["train"]))
        for i in range(n):
            tags = ds_raw["train"][i].get("ner_tags") or []
            for t in tags:
                if isinstance(t, int):
                    max_id = t if max_id is None else max(max_id, t)
        if max_id is not None:
            return [f"LABEL_{i}" for i in range(int(max_id) + 1)]

    return None


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
    }
    os_kw = {"operatingsystem", "os"}
    hardware_kw = {"hardware", "device", "router", "switch", "server", "laptop", "pc", "cpu", "gpu"}
    version_kw = {"version", "release", "build"}
    error_kw = {"error", "issue", "bug", "exception", "failure", "crash"}

    out: Dict[str, str] = {}
    for lab in labels:
        ll = lab.lower().replace("-", "").replace("_", "").replace(" ", "")
        if ll in {"o", "outside"}:
            out[lab] = "O"
            continue
        if any(k in ll for k in software_kw):
            out[lab] = "SOFTWARE"
            continue
        if any(k in ll for k in os_kw):
            out[lab] = "OS"
            continue
        if any(k in ll for k in hardware_kw):
            out[lab] = "HARDWARE"
            continue
        if any(k in ll for k in version_kw):
            out[lab] = "VERSION"
            continue
        if any(k in ll for k in error_kw):
            out[lab] = "ERROR"
            continue
        if "product" in ll or "technology" in ll:
            out[lab] = "SOFTWARE"
            continue
        out[lab] = "O"
    return out


def _bio_to_ids(tags: List[str], label2id: Dict[str, int]) -> List[int]:
    return [int(label2id.get(t, 0)) for t in tags]


def _map_one_example(
    example: Dict[str, Any],
    old_label_names: Optional[List[str]],
    old_to_entity: Dict[str, str],
    new_label2id: Dict[str, int],
) -> Dict[str, Any]:
    tokens = example["tokens"]
    tags = example["ner_tags"]

    out_tags: List[str] = []
    prev_ent = "O"
    for t in tags:
        if isinstance(t, str):
            old_name = t
        else:
            if not old_label_names:
                old_name = f"LABEL_{int(t)}"
            else:
                idx = int(t)
                if idx < 0 or idx >= len(old_label_names):
                    old_name = f"LABEL_{idx}"
                else:
                    old_name = old_label_names[idx]

        ent = old_to_entity.get(old_name, "O")
        if ent == "O":
            out_tags.append("O")
            prev_ent = "O"
            continue
        if prev_ent != ent:
            out_tags.append(f"B-{ent}")
        else:
            out_tags.append(f"I-{ent}")
        prev_ent = ent

    tag_ids = _bio_to_ids(out_tags, new_label2id)
    return {"tokens": tokens, "ner_tags": tag_ids}


def map_dataset_to_new_schema(
    ds_raw: DatasetDict,
    old_label_names: Optional[List[str]],
    tag_mode: str,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    new_labels = _new_label_vocab()
    new_label2id = {n: i for i, n in enumerate(new_labels)}
    meta: Dict[str, Any] = {"new_label_names": new_labels}

    label_source = "features"
    if old_label_names is None:
        old_label_names = _infer_old_label_names_fallback(ds_raw, tag_mode=tag_mode)
        label_source = "fallback"

    if old_label_names is None:
        raise RuntimeError("Cannot infer original label names for mapping.")

    old_to_entity = build_entity_map_from_labels(old_label_names)
    meta["old_label_names"] = old_label_names
    meta["old_to_entity_map"] = old_to_entity
    meta["old_label_source"] = label_source
    meta["tag_mode"] = tag_mode

    def fn(ex):
        return _map_one_example(ex, old_label_names, old_to_entity, new_label2id)

    ds_proc = DatasetDict({k: ds_raw[k].map(fn) for k in ds_raw.keys()})
    return ds_proc, meta


def _label_counts(ds: Dataset, label_names: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {n: 0 for n in label_names}
    for ex in ds:
        for t in ex["ner_tags"]:
            counts[label_names[int(t)]] += 1
    return counts


def _dataset_stats(ds: DatasetDict, label_names: List[str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    for split in ds.keys():
        lengths = [len(x) for x in ds[split]["tokens"]]
        stats[split] = {
            "n": int(len(ds[split])),
            "len_min": int(min(lengths)) if lengths else 0,
            "len_max": int(max(lengths)) if lengths else 0,
            "len_mean": float(statistics.mean(lengths)) if lengths else 0.0,
            "len_med": float(statistics.median(lengths)) if lengths else 0.0,
            "label_counts": _label_counts(ds[split], label_names),
        }
    return stats


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
    os_terms = {
        "windows", "win10", "win11", "ubuntu", "debian", "linux", "macos", "android", "ios",
    }
    hw_terms = {
        "router", "switch", "server", "laptop", "cpu", "gpu", "ssd", "hdd", "ram", "keyboard", "mouse",
    }
    sw_terms = {
        "docker", "kubernetes", "helm",
        "chrome", "firefox", "edge",
        "nginx", "apache",
        "postgres", "postgresql", "mysql", "mariadb", "mongodb",
        "redis", "elasticsearch",
        "node", "nodejs", "npm", "yarn",
        "python", "pip", "conda",
        "java", "maven", "gradle",
        "dotnet", ".net", "nuget",
        "git", "github", "gitlab",
        "jira", "confluence", "slack", "teams",
        "vpn", "vscode", "visualstudio", "powershell", "bash", "zsh",
        "aws", "azure", "gcp",
    }
    err_terms = {"error", "exception", "crash", "failed", "failure", "timeout", "traceback", "stacktrace", "bug"}

    phrase_map = {
        ("windows", "11"): "OS",
        ("windows", "10"): "OS",
        ("windows", "7"): "OS",
        ("windows", "8.1"): "OS",
        ("windows", "server"): "OS",
        ("windows", "server", "2019"): "OS",
        ("windows", "server", "2022"): "OS",
        ("visual", "studio", "code"): "SOFTWARE",
        ("visual", "studio"): "SOFTWARE",
        ("vs", "code"): "SOFTWARE",
        ("google", "chrome"): "SOFTWARE",
        ("microsoft", "edge"): "SOFTWARE",
        ("microsoft", "teams"): "SOFTWARE",
        ("office", "365"): "SOFTWARE",
        ("microsoft", "office"): "SOFTWARE",
    }

    ver_re = re.compile(r"^(v)?\d+(\.\d+){1,4}([a-z0-9\-]+)?$", re.IGNORECASE)
    kb_re = re.compile(r"^KB\d+$", re.IGNORECASE)
    winver_re = re.compile(r"^\d{2}h\d$", re.IGNORECASE)
    build_re = re.compile(r"^\d{4,6}$")

    labels = ["O"] * len(tokens)

    def mark_span(i0: int, i1: int, ent: str) -> None:
        if i0 < 0 or i1 > len(tokens) or i0 >= i1:
            return
        if any(labels[i] != "O" for i in range(i0, i1)):
            return
        labels[i0] = f"B-{ent}"
        for i in range(i0 + 1, i1):
            labels[i] = f"I-{ent}"

    max_phrase_len = max((len(k) for k in phrase_map.keys()), default=0)
    toks_l = [t.lower() for t in tokens]
    for n in range(max_phrase_len, 1, -1):
        for i in range(0, len(tokens) - n + 1):
            key = tuple(toks_l[i : i + n])
            ent = phrase_map.get(key)
            if ent:
                mark_span(i, i + n, ent)

    for i, t in enumerate(tokens):
        tl = t.lower()

        if labels[i] != "O":
            continue

        prev = tokens[i - 1].lower() if i > 0 else ""

        if (
            kb_re.match(t)
            or ver_re.match(tl)
            or winver_re.match(tl)
            or tl in {"rc1", "rc2", "beta", "alpha"}
            or (build_re.match(t) and prev in {"build", "ver", "version"})
        ):
            mark_span(i, i + 1, "VERSION")
            continue

        stack_re = re.compile(r"^[a-z]{2,}(\d+(\.\d+)+)$", re.IGNORECASE)
        if stack_re.match(tl):
            mark_span(i, i + 1, "SOFTWARE")
            continue

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

    for i in range(len(tokens) - 1):
        if labels[i] != "O" or labels[i + 1] != "O":
            continue
        if toks_l[i] == "windows" and re.fullmatch(r"\d{1,2}", toks_l[i + 1]):
            mark_span(i, i + 2, "OS")

    return labels


def build_weak_dataset(
    weak_dataset_name: str,
    sample_n: int,
    max_len: int,
    min_len: int,
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
    seen: set[str] = set()

    label_names = _new_label_vocab()
    label2id = {n: i for i, n in enumerate(label_names)}

    for ex in ds.shuffle(seed=seed).take(sample_n):
        text = _pick_text_fields(ex)
        if not text:
            continue
        toks = _simple_tokenize(text, max_len=max_len)
        toks_key = " ".join([t.lower() for t in toks])
        if toks_key in seen:
            continue
        seen.add(toks_key)
        if len(toks) < int(min_len):
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
    meta = {
        "enabled": True,
        "weak_ratio": float(weak_ratio),
        "weak_used": int(use_n),
        "gold_train": int(len(gold_train)),
        "train_after_mix": int(len(mixed)),
    }
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
    weak_min_len: int = 8,
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

    if old_label_names is None:
        old_label_names = _infer_old_label_names_fallback(ds_raw, tag_mode=tag_mode)

    logger.info("Tag mode: %s", tag_mode)
    logger.info("Old labels: %s", str(old_label_names)[:200])

    ds_proc, meta = map_dataset_to_new_schema(ds_raw, old_label_names, tag_mode=tag_mode)
    label_names = meta["new_label_names"]

    ds_proc2, overs_meta = oversample_train(
        ds_proc,
        label_names,
        factor=float(oversample_factor),
        rare_boost=float(rare_boost),
        seed=int(seed),
    )
    meta["oversample"] = overs_meta

    weak_meta = {"enabled": False}
    if weak_dataset_name:
        logger.info("Building weak dataset: %s", weak_dataset_name)
        weak_ds = build_weak_dataset(
            weak_dataset_name,
            sample_n=int(weak_sample_n),
            max_len=int(weak_max_len),
            min_len=int(weak_min_len),
            seed=int(seed),
        )
        mixed, mix_meta = mix_gold_and_weak(
            ds_proc2["train"],
            weak_ds,
            weak_ratio=float(weak_ratio),
            seed=int(seed),
        )
        ds_proc2 = DatasetDict({**ds_proc2, "train": mixed})
        weak_meta = {
            "enabled": True,
            "weak_dataset": weak_dataset_name,
            "weak_sample_n": int(weak_sample_n),
            "weak_max_len": int(weak_max_len),
            "weak_min_len": int(weak_min_len),
            "mix": mix_meta,
        }
    meta["weak"] = weak_meta

    logger.info("Saving processed dataset to: %s", processed_path)
    save_dataset(ds_proc2, processed_path)

    stats = _dataset_stats(ds_proc2, label_names)
    meta["stats"] = stats
    meta["dataset_key"] = key

    write_json(paths.data / "processed_meta.json", {"new_label_names": label_names, "meta": meta})
    write_json(paths.data / "dataset_meta.json", meta)

    logger.info("Done. Splits: %s", {k: len(ds_proc2[k]) for k in ds_proc2.keys()})
    return meta
