# Domain-Specific NER for IT Support using PEFT and Model Compression

## Overview

This project studies how to efficiently adapt transformer-based models for  
**Named Entity Recognition (NER)** in the **IT support domain**.

Although large pretrained language models demonstrate strong performance on general-domain text, their effectiveness often decreases when applied to domain-specific data such as IT support tickets, bug reports, or developer discussions.  
Full fine-tuning can mitigate this issue, but it is computationally expensive and may be inefficient in resource-constrained settings.

The goal of this project is to investigate whether **parameter-efficient fine-tuning (PEFT)** and **model compression techniques** can provide an effective and practical alternative for IT-domain NER.

---

## Task Description

**Task type:** Sequence labeling (Named Entity Recognition)  
**Labeling scheme:** BIO tagging  

Given an IT-related text, the model assigns an entity label to each token.

### Example

```
User cannot connect to VPN on Windows 11 after update
O    O      O       O  B-SOFTWARE O B-OS     I-OS  O     O
```

---

## Entity Schema

The task focuses on a compact and domain-relevant set of **five IT-specific entity types**:

| Entity     | Description                           | Examples            |
|------------|---------------------------------------|---------------------|
| SOFTWARE   | Applications, services, tools          | VPN, Docker, Chrome |
| OS         | Operating systems                      | Windows 11, Ubuntu  |
| HARDWARE   | Physical devices or components         | router, server, GPU |
| ERROR      | Errors, crashes, failure messages      | timeout, crash      |
| VERSION    | Versions, updates, release identifiers | v2.1.3, KB503       |

All other tokens are labeled as `O`.

---

## Dataset

**Dataset:** `mrm8488/stackoverflow-ner` (HuggingFace)

- Source: StackOverflow and GitHub texts  
- Domain: Software engineering / IT  
- Format: Tokenized sentences with BIO-style NER tags  
- Size: ~22k labeled sentences  
- Splits: Predefined train / validation / test  

The dataset is treated as a **domain-specific corpus**.  
Domain adaptation is performed by fine-tuning general-purpose pretrained models on this IT-domain dataset.

---

## Models and Methods

The following modeling strategies are used in this project:

### BERT (Full Fine-Tuning)
- Model: `bert-base-cased`
- All model parameters are updated during training
- Used as a reference approach

### BERT with Parameter-Efficient Fine-Tuning (LoRA)
- Backbone parameters are frozen
- Low-rank adaptation (LoRA) is applied to attention layers
- Only a small subset of parameters is trained

### DistilBERT (Model Compression)
- Model: `distilbert-base-cased`
- A smaller and faster alternative to BERT
- Fully fine-tuned for the NER task

### Post-training Compression (ONNX)
- Export of trained models to ONNX format
- Dynamic INT8 quantization on CPU

---

## Evaluation

> **Status:** _To be completed_

Planned evaluation metrics include:

- Entity-level Precision, Recall, and F1-score (using `seqeval`)
- Per-entity performance analysis
- Number of trainable parameters
- Model size (MB)
- Training time
- CPU inference latency (PyTorch vs ONNX vs quantized ONNX)

Results and analysis will be added after completing the experimental runs.

---

## Experimental Workflow

1. Dataset loading and preprocessing  
2. Entity label mapping and BIO alignment  
3. Exploratory data analysis (EDA)  
4. Model training:
   - BERT (full fine-tuning)
   - BERT with LoRA
   - DistilBERT
5. Evaluation and qualitative error analysis  
6. Model export and post-training compression  
7. Comparative analysis of performance–efficiency trade-offs  

---

## Repository Structure

```
it-ner-peft-compression/
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── it_ner_peft_compression.ipynb
│       # End-to-end experiments and analysis
│
├── src/
│   ├── data.py        # Dataset loading, label mapping, token alignment
│   ├── train.py       # Training routines
│   ├── eval.py        # Evaluation and error analysis
│   └── export.py      # ONNX export and quantization
│
├── report/
│   ├── figures/       # Figures used in the report
│   ├── latex/
│   │   └── main.tex   # LaTeX source for the final report
│   └── report.pdf    # Final PDF (to be generated)
│
└── configs/
    └── base.yaml      # Training and model configuration
```

---

## Results

> **Status:** _Not available yet_

This section will summarize quantitative results and comparisons once experiments are completed.

---

## Discussion

> **Status:** _Not available yet_

This section will discuss observed trade-offs, limitations, and insights derived from the experiments.

---

## Reproducibility

All experiments are conducted on a single-GPU setup (RTX 3070 Ti, 8GB VRAM).  
Training configurations, dataset splits, and random seeds are fixed to ensure reproducibility.
