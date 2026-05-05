# ViT in the Low‑Data Regime – CIFAR‑10 (100 samples/class)

This repository contains experiments exploring how Vision Transformers (ViTs) behave under a strict low‑data setting and how modern Self‑Supervised Learning (SSL) and Knowledge Distillation (KD) techniques can bridge the performance gap.

**📄 Full report:** [vit_low_data_report.pdf](./report/vit_low_data_report.pdf)

## 🚀 Quick Overview

All experiments use **CIFAR‑10** with exactly **100 labelled samples per class** (1 000 training images total) and an independent validation set of the same size. The goal is to build a small, efficient ViT that performs competitively without large‑scale labelled datasets.

### 🔥 Key Results

| Model / Strategy | Test Acc | Params |
|------------------|----------|--------|
| Vanilla ViT from scratch | 36.22 % | 7.18 M |
| Pretrained ViT‑S/16 (teacher) | 92.11 % | 21.67 M |
| Custom SmallViT from scratch | 54.97 % | 7.96 M |
| MIM + diff. LR fine‑tuning | 45.66 % | 7.96 M |
| MIM + uniform LR fine‑tuning | 55.70 % | 7.96 M |
| **MIM + KD (BEST)** | **68.31 %** | **7.96 M** |

**→ +34.8 pp absolute improvement over the vanilla ViT baseline.**

## 🧠 Techniques

- **Custom SmallViT** – ConvStem, PEG, register tokens, SwiGLU FFN  
- **Masked Image Modelling** – self‑supervised pre‑training without labels  
- **Knowledge Distillation** – soft‑label supervision from an ImageNet‑pretrained ViT‑S/16 teacher  
- **Strong Augmentation** – RandAugment, Mixup, CutMix, label smoothing

models link: https://drive.google.com/drive/folders/1FkUDu2fhGz_AqWWp6CBW2wnJulfYemp3?usp=sharing
