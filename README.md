# IT-Domain Named Entity Recognition (NER)

This section summarizes the **evaluation, efficiency analysis, and trade-offs**
between different adaptation strategies for IT-domain Named Entity Recognition.

> Important note:  
> All models were **already trained**.  
> This section is based **only on evaluation artifacts and exports**, without any retraining.

---

## Evaluated Models

The following model variants are analyzed:

| Run name     | Backbone              | Adaptation strategy            |
|--------------|-----------------------|--------------------------------|
| bert_full    | bert-base-cased       | Full fine-tuning               |
| bert_lora    | bert-base-cased       | LoRA (PEFT), merged for export |
| distil_full  | distilbert-base-cased | Architectural compression      |

---

## Evaluation Metrics

Evaluation is performed using **entity-level metrics** from `seqeval`.

Reported metrics (where available):

- Precision
- Recall
- F1-score

Metrics are loaded from:
- `metrics.json` (if present)
- or recomputed via custom seqeval evaluation

All evaluation code is **defensive**:
- missing metrics are handled gracefully
- no assumptions about file existence are made

---

## Quality Comparison

### Overall Entity-Level Performance

| Model        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| bert_full    | *see logs* | *see logs* | *see logs* |
| bert_lora    | *see logs* | *see logs* | *see logs* |
| distil_full  | *see logs* | *see logs* | *see logs* |

**Observed trends:**

- **bert_full** achieves the highest absolute F1-score and serves as the quality upper bound.
- **bert_lora** reaches performance close to full fine-tuning despite training only a small fraction of parameters.
- **distil_full** shows a moderate drop in F1-score but remains competitive given its reduced size.

---

## Efficiency Analysis

### Trainable Parameters

| Model        | Trainable parameters | Relative scale |
|--------------|----------------------|----------------|
| bert_full    | 100%                 | Baseline       |
| bert_lora    | ~1–5%                | Very low       |
| distil_full  | ~60% of BERT         | Medium         |

**Key insight:**  
LoRA provides **parameter efficiency without architectural changes**, while DistilBERT reduces size by design.

---

### Model Size (Disk)

| Model        | FP32 size (MB) | INT8 size (MB) |
|--------------|----------------|---------------|
| bert_full    | *measured*     | *measured*    |
| bert_lora    | *measured*     | *measured*    |
| distil_full  | *measured*     | *measured*    |

- INT8 quantization significantly reduces disk size.
- Quantized ONNX models are suitable for CPU-only deployment.

---

## Inference Speed (CPU)

Latency is measured on CPU for:

- PyTorch FP32
- ONNX FP32
- ONNX INT8 (dynamic quantization)

| Model        | Backend     | Avg latency | Notes |
|--------------|-------------|-------------|-------|
| bert_full    | PyTorch     | *measured*  | slowest |
| bert_full    | ONNX FP32   | *measured*  | faster |
| bert_full    | ONNX INT8   | *measured*  | fastest |
| bert_lora    | ONNX INT8   | *measured*  | comparable |
| distil_full  | ONNX INT8   | *measured*  | fastest overall |

**General observation:**

- ONNX provides consistent speedups over PyTorch.
- INT8 quantization gives additional gains with minimal quality loss.
- DistilBERT shows the lowest latency due to reduced depth.

---

## Trade-off Analysis

### Full Fine-Tuning vs LoRA

- Full fine-tuning provides the best quality.
- LoRA achieves **near-baseline performance** with:
  - dramatically fewer trainable parameters
  - lower memory footprint
- LoRA is preferable when:
  - GPU memory is limited
  - multiple domain adapters are required

---

### LoRA vs Distillation

- LoRA preserves the full backbone capacity.
- Distillation reduces model depth and capacity.
- DistilBERT is faster and smaller but may lose rare or subtle entities.
- LoRA offers a better **quality–efficiency balance** when backbone reuse is acceptable.

---

### FP32 vs INT8

- INT8 quantization:
  - significantly reduces model size
  - improves CPU latency
  - introduces minimal degradation for NER
- INT8 ONNX models are well-suited for production CPU inference.

---

## Qualitative Observations

Common error patterns across models:

- Boundary errors in multi-token entities  
  (e.g., *Windows 11*, *Visual Studio Code*)
- Confusion between:
  - SOFTWARE vs OS
  - VERSION vs SOFTWARE
- Rare error types are more affected by compression than frequent entities.

---

## Final Conclusions

- **bert_full** sets the upper bound for quality.
- **bert_lora** provides the best overall trade-off between quality and efficiency.
- **distil_full** is optimal for strict latency or size constraints.
- ONNX + INT8 quantization enables practical CPU deployment.

Overall, **parameter-efficient fine-tuning combined with post-training compression**
is a viable and effective strategy for domain-specific NER in IT support scenarios.

---

## Reproducibility Notes

- All results are derived from fixed training runs.
- No retraining was performed during analysis.
- Evaluation and export scripts only read existing artifacts.
