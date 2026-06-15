# MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)

## 📖 Overview

This repository contains the official implementation of our paper **MoRA**, which addresses the challenging problem of multimodal learning with missing modalities. Our method introduces a novel low-rank adaptation approach specifically designed for robust visual recognition when one or more modalities are absent during training or inference.

### 🔑 Key Features

- ✨ **Robust to Missing Modalities**: Handles missing text, image, or both modalities
- 🚀 **Efficient Fine-tuning**: Low-rank adaptation for parameter-efficient training
- 🎯 **State-of-the-art Performance**: Evaluated on MM-IMDb, Hateful Memes, and Food-101

## 📋 Table of Contents

- [Installation](#-installation)
- [Data Preparation](#-data-preparation)
- [Usage](#-usage)
  - [Training](#training)
  - [Testing](#testing)
- [Citation](#-citation)

## 🔧 Installation

### Prerequisites

- Python >= 3.12
- PyTorch >= 2.6.0
- CUDA compatible GPU (recommended)

### Setup Environment

We recommend using [`uv`](https://github.com/astral-sh/uv) for fast dependency management:

```bash
# Clone the repository
git clone https://github.com/Tree-Shu-Zhao/MoRA.git
cd MoRA

# Install dependencies using uv
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

## 📊 Data Preparation

All three datasets (MM-IMDb, Hateful Memes, Food-101), already preprocessed into the Arrow format the training code expects, are hosted on the Hugging Face Hub at [`TreezzZ/MoRA-data`](https://huggingface.co/datasets/TreezzZ/MoRA-data). The files drop directly into `datasets/` with no renaming, and the precomputed missing-modality masks used in the paper are included.

```bash
# from the repository root
huggingface-cli download TreezzZ/MoRA-data --repo-type dataset --local-dir datasets
```

## 🚀 Usage

### Training

Train on different datasets with various missing modality configurations:

```bash
# Train on Hateful Memes
python src/main.py experiment=mora_hatememes

# Train on MM-IMDb
python src/main.py experiment=mora_mmimdb

# Train on Food-101
python src/main.py experiment=mora_food101
```

### Testing

Evaluate a trained model checkpoint:

```bash
# Test on Hateful Memes
python src/main.py \
    experiment=mora_hatememes \
    test.TEST_ONLY=True \
    test.CHECKPOINT_PATH=/path/to/checkpoint.pth
```

## 📚 Citation

If you found our paper useful, please cite it:

```bibtex
@inproceedings{zhao2026mora,
  title={MoRA: Missing Modality Low-Rank Adaptation for Visual Recognition},
  author={Zhao, Shu and Ahuja, Nilesh and Yu, Tan and Shen, Tianyi and Narayanan, Vijaykrishnan},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
