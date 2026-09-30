# ViT in the Low-Data Regime: CIFAR-10 with 100 Samples per Class

This repository studies how Vision Transformers behave under a very low-data regime and evaluates whether self-supervised pretraining and knowledge distillation can recover much of the performance lost to scarce labels.

The project uses CIFAR-10 with exactly 100 labeled samples per class, creating a 1,000-image training subset. The goal is to benchmark a compact student ViT against a teacher model and test whether structured pretraining can help when supervised learning alone is not enough.

The full report is available at: [report/vit_low_data_report.pdf](report/vit_low_data_report.pdf)

## Overview

Low-data learning is a realistic challenge in domains where annotation is expensive, imbalanced, or unavailable. This project explores the same question for ViTs:

- Can a compact ViT learn effectively from only 1,000 labeled examples?
- How much does a strong teacher improve learning?
- Can masked image modeling (MIM) pretraining create a better initialization?
- Can knowledge distillation bridge the gap between a small student and an ImageNet-pretrained teacher?

## Key Results

| Model / Strategy | Test Accuracy | Params |
| --- | ---: | ---: |
| Vanilla ViT from scratch | 36.22% | 7.18 M |
| Pretrained ViT-S/16 teacher | 92.11% | 21.67 M |
| Custom SmallViT from scratch | 54.97% | 7.96 M |
| MIM + different LR fine-tuning | 45.66% | 7.96 M |
| MIM + uniform LR fine-tuning | 55.70% | 7.96 M |
| MIM + KD (best) | 68.31% | 7.96 M |

This represents a +34.8 percentage-point absolute gain over the vanilla baseline, showing that the combination of self-supervised pretraining and distillation substantially helps in low-data regimes.

## Core Ideas

### 1. Small, efficient student architecture
The student model is built in [models/small_vit.py](models/small_vit.py) and uses a compact transformer design with:

- ConvStem for early CNN-style downsampling
- Positional encoding with register tokens
- PEG (Positional Encoding Generator) to encourage spatial inductive bias
- SwiGLU-style feed-forward block
- LayerScale residual scaling

This design is intended to reduce the sample complexity of ViTs while keeping parameter count low.

### 2. Masked Image Modeling (MIM)
The MIM-style pretraining model is defined in [models/mim_model.py](models/mim_model.py). It masks a large fraction of the feature patches, reconstructs the missing stem features from the visible tokens, and teaches the student to model the latent structure of the image without annotations.

### 3. Knowledge Distillation
The project distills from a teacher network trained on ImageNet through timm. The custom loss utilities in [utils/losses.py](utils/losses.py) include:

- soft-target cross-entropy
- KL-divergence distillation loss
- feature alignment loss

This supports the final low-data learning stage where a small student learns from the teacher's soft predictions and feature space.

### 4. Strong data augmentation
The data pipeline in [utils/data.py](utils/data.py) constructs a CIFAR-10 subset with exactly 100 examples per class and applies augmentation such as:

- RandomResizedCrop
- RandomHorizontalFlip
- RandAugment
- mixup-style training support via custom loss utilities

## Repository Structure

```text
Vit-Low-data-study/
├── README.md
├── requirements.txt
├── models/
│   ├── kd_pretraining.py
│   ├── mim_model.py
│   └── small_vit.py
├── notebooks/
│   ├── 01_vanilla_vit_baseline.ipynb
│   ├── 02_teacher_vit_finetune.ipynb
│   ├── 03_mim_pretraining.ipynb
│   └── 04_full_pipeline.ipynb
├── report/
│   └── vit_low_data_report.pdf
├── results/
│   ├── custom_normal_vit_loss.png
│   ├── final_mim_kd_loss.png
│   ├── MIM_pretraining_loss.png
│   ├── MIM_strat1_loss.png
│   ├── MIM_strat2_loss.png
│   ├── teacher_loss.png
│   └── timm_vit_loss.png
├── utils/
│   ├── data.py
│   └── losses.py
└── .gitignore
```

## Setup

1. Clone the repository:

```bash
git clone <repo-url>
cd Vit-Low-data-study
```

2. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

The project uses:

- PyTorch
- TorchVision
- timm
- NumPy
- Matplotlib
- tqdm

## Data Pipeline

The dataset logic lives in [utils/data.py](utils/data.py). It:

- downloads CIFAR-10 automatically with `torchvision.datasets.CIFAR10`
- selects exactly `num_per_class` examples from each class
- builds a `Subset` for the low-data training regime
- applies training vs evaluation transforms separately

A typical training split is:

- 100 examples/class for training
- remaining validation/test examples held out for evaluation

This is a low-label setup rather than a full benchmark pipeline and is suitable for studying label scarcity.

## Training Workflow

The notebooks represent the main experimental pipeline:

### 1. Vanilla baseline
`notebooks/01_vanilla_vit_baseline.ipynb`

- trains a compact ViT from scratch
- establishes the bare low-data baseline
- captures loss curves and accuracy

### 2. Teacher fine-tuning
`notebooks/02_teacher_vit_finetune.ipynb`

- loads a pretrained ViT-S/16 from timm
- fine-tunes it on the low-data CIFAR-10 subset
- acts as the teacher for downstream distillation

### 3. MIM pretraining
`notebooks/03_mim_pretraining.ipynb`

- pretrains the student with masked reconstruction objectives
- uses the stem feature output as target
- learns a better initialization before supervised fine-tuning

### 4. Full learning pipeline
`notebooks/04_full_pipeline.ipynb`

- runs the end-to-end student pipeline
- combines MIM pretraining and knowledge distillation
- evaluates the final low-data performance

## Model Definitions

### SmallViT
Defined in [models/small_vit.py](models/small_vit.py). The architecture uses:

- ConvStem for compact spatial tokenization
- embeddings for patch tokens and register tokens
- transformer encoder blocks
- PEG-based spatial refinement
- global average pooling + classifier head

### MIMSmallViT
Defined in [models/mim_model.py](models/mim_model.py). This model extends the student by adding a masked encoder-decoder structure that reconstructs missing patch-level features.

## Running the Project

Since the project is organized around notebooks, the simplest workflow is:

```bash
jupyter notebook
```

Then open the notebook in order:

1. `01_vanilla_vit_baseline.ipynb`
2. `02_teacher_vit_finetune.ipynb`
3. `03_mim_pretraining.ipynb`
4. `04_full_pipeline.ipynb`

The notebooks contain the training logic and experiment tracking. If you want to adapt the code, the reusable components live in the model and utility modules.

## Outputs and Artifacts

The project stores training curves in [results](results):

- `custom_normal_vit_loss.png`
- `timm_vit_loss.png`
- `teacher_loss.png`
- `MIM_pretraining_loss.png`
- `MIM_strat1_loss.png`
- `MIM_strat2_loss.png`
- `final_mim_kd_loss.png`

These plots support the conclusions in the report and show the optimization trajectory during baseline, teacher, MIM, and final distillation stages.

## Notes and Caveats

- This project is built for research and experimentation, not a production training framework.
- Model weights are not included in the repository by default; the paper/report and notebooks are the primary assets.
- The timm teacher model requires internet access or a cached checkpoint the first time it is downloaded.
- The subset selection is deterministic based on a fixed random seed when configured, which helps reproduce low-data experiments.

## License

This project is intended for research use and is distributed as-is without a formal license file. If you reuse any part of the code or report, please cite the repository and report appropriately.

## Related Resources

- CIFAR-10 dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- timm library: https://github.com/huggingface/pytorch-image-models
- MIM / self-supervised vision methods: references in the report and notebooks

Model weights and related assets are also linked in the project notes and report materials.
