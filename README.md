# IT-Domain Named Entity Recognition (NER): Practical Compression for Real-World Inference

This repository presents an analysis of **quality, efficiency, and deployment trade-offs**
between different adaptation strategies for **IT-domain Named Entity Recognition (NER)**.

All models compared here were trained beforehand.  
The focus is on **comparative evaluation and practical implications**, not on training procedures.

---

## Evaluated Models

| Run name     | Backbone              | Adaptation strategy            |
|--------------|-----------------------|--------------------------------|
| bert_full    | bert-base-cased       | Full fine-tuning               |
| bert_lora    | bert-base-cased       | LoRA (PEFT), merged weights    |
| distil_full  | distilbert-base-cased | Architectural compression      |

---

## Evaluation Metrics

Evaluation is performed using **entity-level NER metrics**:

- Precision  
- Recall  
- F1-score  

Metrics are computed at the entity level following standard sequence-labeling evaluation practice.

---

## Quality Comparison

### Overall Entity-Level Performance

| Model        | Precision | Recall | F1-score |
|--------------|-----------|--------|----------|
| bert_full    | 0.672     | 0.743  | **0.706** |
| bert_lora    | 0.637     | 0.776  | 0.699 |
| distil_full  | 0.454     | 0.297  | 0.359 |

**Observations:**

- **bert_full** achieves the highest F1-score and defines the quality upper bound.
- **bert_lora** closely matches full fine-tuning, trading slightly lower precision for higher recall.
- **distil_full** shows a substantial recall drop, leading to noticeably lower overall quality.

---

## Efficiency Analysis

### Trainable Parameters

| Model        | Trainable parameters | Relative scale |
|--------------|----------------------|----------------|
| bert_full    | 100%                 | High           |
| bert_lora    | ~1–5%                | Very low       |
| distil_full  | ~60% of BERT         | Medium         |

**Key insight:**  
LoRA achieves near-baseline quality while updating only a small fraction of parameters, whereas DistilBERT reduces capacity through architectural changes.

---

### Model Size

- INT8 quantization significantly reduces model size across all variants.
- Quantized models are suitable for **CPU-only deployment**.
- Size reduction is most impactful for full BERT, while also benefiting LoRA and DistilBERT models.

---

## Inference Speed (CPU)

Latency is evaluated under:

- PyTorch FP32  
- ONNX FP32  
- ONNX INT8  

**General trends:**

- ONNX inference consistently outperforms PyTorch execution.
- INT8 quantization provides additional latency improvements.
- **DistilBERT** achieves the lowest latency due to reduced depth.
- **LoRA-based BERT** remains competitive while retaining full backbone capacity.

---

## Trade-off Analysis

### Full Fine-Tuning vs LoRA

- Full fine-tuning yields the best absolute quality.
- LoRA achieves **near-baseline performance** with:
  - dramatically fewer trainable parameters
  - lower memory requirements
- LoRA is preferable when resources are constrained or multiple domain adaptations are needed.

---

### LoRA vs Distillation

- LoRA preserves the representational capacity of the full backbone.
- Distillation reduces size and depth at the cost of expressiveness.
- DistilBERT performs well under strict latency constraints but struggles with rare or complex entities.
- LoRA offers a more balanced quality–efficiency trade-off when backbone reuse is acceptable.

---

### FP32 vs INT8

- INT8 quantization:
  - substantially reduces model size
  - improves CPU inference speed
  - introduces only minor degradation for NER
- INT8 ONNX models are suitable for practical production scenarios.

---

## Qualitative Observations

Common error patterns across models include:

- Boundary errors in multi-token entities  
  (e.g., *Windows 11*, *Visual Studio Code*)
- Confusion between closely related entity types:
  - SOFTWARE vs OS
  - SOFTWARE vs VERSION
- Rare entities are more sensitive to compression than frequent ones.

---

## Final Conclusions

- **bert_full** defines the upper bound for quality.
- **bert_lora** provides the best balance between performance and efficiency.
- **distil_full** is appropriate for scenarios with strict latency or size constraints.
- Combining parameter-efficient adaptation with post-training quantization enables practical CPU deployment.

Overall, **parameter-efficient fine-tuning combined with lightweight compression**
is an effective strategy for domain-specific NER in IT support scenarios.
