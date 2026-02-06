# IT-Domain Named Entity Recognition with PEFT and Compression

This repository provides a reproducible, model-centric study of IT-domain NER.
It implements dataset relabeling, augmentation, model training, evaluation, error analysis,
and ONNX export/benchmarking for deployment-oriented trade-offs.
All results below are taken directly from the stored artifacts in `data/` and `models/`.

## Scope and Methodology

The pipeline covers:
- Data preparation with label remapping, oversampling, and weak supervision.
- Three model variants: full BERT fine-tuning, LoRA (PEFT) on BERT, and DistilBERT.
- Entity-level evaluation using seqeval on a held-out test set.
- Error analysis with confusion summaries.
- ONNX export and latency benchmarking for deployment assessment.

## Dataset and Labeling

Source dataset: `mrm8488/stackoverflow-ner`.

Label space was collapsed into a 5-entity BIO schema:
`ERROR`, `HARDWARE`, `OS`, `SOFTWARE`, `VERSION` (plus `O`).

Data augmentation:
- Oversampling factor: 1.5 with rare-entity boost 0.5.
- Training size before oversampling: 9,263 examples.
- After oversampling: 11,701 examples (including a rare-entity boost for ERROR and OS).
- Weakly labeled augmentation: 3,510 examples from `bigcode/the-stack-github-issues`
  (100k sampled, filtered, and mixed at a 0.30 ratio).
- Final training size: 15,211 examples.

Dataset statistics (after augmentation):

| Split | Samples | Mean tokens | Median tokens |
|-------|---------|-------------|---------------|
| Train | 15,211  | 40.18       | 18            |
| Val   | 2,936   | 14.75       | 13            |
| Test  | 3,108   | 14.65       | 13            |

## Models

| Run        | Backbone              | Adaptation |
|------------|-----------------------|------------|
| bert_full  | bert-base-cased       | Full fine-tuning |
| bert_lora  | bert-base-cased       | LoRA (PEFT) |
| distil_full| distilbert-base-cased | Full fine-tuning |

LoRA configuration (from `models/bert_lora/config_resolved.json`):
- r=16, alpha=32, dropout=0.05
- target modules: query, key, value, dense
- classifier head saved with adapters

Training configuration (common settings):
- Max length: 128
- Epochs: 6
- Batch size: 16
- Optimizer: AdamW
- Learning rate: 2e-5 (BERT/DistilBERT), 1e-4 (LoRA)
- Scheduler: cosine
- Label smoothing: 0.05
- Gradient checkpointing: enabled
- Mixed precision: fp16 with tf32 enabled
- Early stopping: patience 1

## Evaluation Results (Entity-Level, Test Set)

Overall metrics from `models/*/metrics.json`:

| Model       | Precision | Recall | F1    | Accuracy |
|-------------|-----------|--------|-------|----------|
| bert_full   | 0.748     | 0.733  | 0.740 | 0.988    |
| bert_lora   | 0.694     | 0.755  | 0.723 | 0.987    |
| distil_full | 0.754     | 0.683  | 0.717 | 0.987    |

Per-entity F1 (entity counts from test set):

| Entity   | Count | bert_full | bert_lora | distil_full |
|----------|-------|-----------|-----------|-------------|
| ERROR    | 18    | 0.188     | 0.065     | 0.069       |
| HARDWARE | 53    | 0.720     | 0.693     | 0.660       |
| OS       | 66    | 0.843     | 0.843     | 0.841       |
| SOFTWARE | 838   | 0.733     | 0.714     | 0.705       |
| VERSION  | 111   | 0.818     | 0.827     | 0.840       |

Key outcome:
- Full BERT yields the highest overall F1.
- LoRA preserves recall and remains close to full fine-tuning while updating fewer parameters.
- DistilBERT is competitive but shows lower recall at comparable precision.

## Model Size and ONNX Inference

Saved model sizes (safetensors):
- `models/bert_full/model.safetensors`: 410.97 MB
- `models/distil_full/model.safetensors`: 248.73 MB
- `models/bert_lora/adapter_model.safetensors`: 10.18 MB

ONNX export artifacts (from `models/onnx/*/bench.json`):
- BERT FP16 ONNX: 1.56 MB
- LoRA FP16 ONNX: 1.56 MB
- DistilBERT FP16 ONNX: 0.79 MB

CPU latency (ORT FP32, seq_len=128, avg over 200 runs):

| Model       | Avg latency (ms) |
|-------------|------------------|
| bert_full   | 43.32            |
| bert_lora   | 60.78            |
| distil_full | 30.37            |

Note: ONNX benchmarks used `CPUExecutionProvider` because GPU providers were not available
in the benchmark environment captured in the repository.

## Error Analysis (Summary)

Dominant confusions from `models/bert_full/errors.json`:
- SOFTWARE -> O (232 cases)
- O -> SOFTWARE (176 cases)
- ERROR -> O (33 cases)
- VERSION -> O (20 cases)

This indicates that boundary and omission errors are most frequent, especially for SOFTWARE
entities, while rare classes (ERROR) remain challenging.

## Code Structure

Key modules:
- `src/data.py`: dataset relabeling, oversampling, weak labeling, and statistics export
- `src/train.py`: training for BERT, LoRA, and DistilBERT
- `src/eval.py`: evaluation, per-entity metrics, and confusion summaries
- `src/export.py`: ONNX export and latency benchmarking
- `src/distill.py`: knowledge distillation trainer (student DistilBERT)
- `src/prompting.py`: LLM prompting baseline (Ollama-based)
- `src/robustness.py`: token-level perturbations for robustness checks

## Conclusions

The results show that parameter-efficient fine-tuning (LoRA) retains most of the
full fine-tuning quality while substantially reducing adapter size, and that
DistilBERT offers faster CPU inference with a modest drop in recall.
The provided artifacts include metrics, logs, and ONNX exports suitable for
reproducible evaluation and deployment-oriented analysis.

## Demo 

Run the interactive showcase:

```bash
streamlit run streamlit_app.py
```

Notes:
- The demo reads model runs from `models/` (excluding `models/onnx`).
- It uses `data/processed_meta.json` for label names; run dataset preparation if missing.
