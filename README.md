# AWDC-MVC

This repository contains the source code for **AWDC-MVC** (Adaptive Weighting-guided Dual-level Contrastive Learning for
Multi-view Clustering).

## Requirements

The code is implemented in Python using PyTorch. The required dependencies are listed in `requirements.txt`.

- Python 3.x
- PyTorch
- NumPy
- SciPy
- scikit-learn

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Data Preparation

Please place your dataset files in the `data/` directory. The code supports several datasets including:

- **Cifar100**: Expects `cifar100.mat`
- **Prokaryotic**: Expects `prokaryotic.mat`
- **CCV**: Expects `ccv.mat`

Ensure the file paths and formats correspond to the implementations in `dataloader.py`.

## Usage

You can train the model using the `train.py` script. Specify the dataset name using the `--dataset` argument.

### Training Examples

**Train on Cifar100:**
```bash
python train.py --dataset Cifar100
```

**Train on Prokaryotic:**
```bash
python train.py --dataset Prokaryotic
```

### Key Arguments

- `--dataset`: The dataset to use (e.g., `Cifar100`, `Prokaryotic`, `CCV`). Default is `Cifar100`.
- `--batch_size`: Input batch size for training (default: 256).
- `--learning_rate`: Learning rate (default: 0.0003).
- `--pre_epochs`: Number of epochs for pre-training (auto-encoder).
- `--con_epochs`: Number of epochs for contrastive learning / main training.
- `--feature_dim`: Dimension of the latent feature representation.

## Project Structure

- `train.py`: The main entry point for training the model.
- `network.py`: Defines the neural network architecture, including Encoders, Decoders, and the Fusion module.
- `dataloader.py`: Contains `Dataset` classes for loading and preprocessing different datasets.
- `loss.py`: Implements custom loss functions such as Contrastive Loss.
- `metric.py` / `metricfinal.py`: Utility functions for evaluating model performance (Clustering Accuracy, NMI, ARI, etc.).
- `TransformerViewFusion.py`: Implementation of the Transformer-based view fusion mechanism.
- `MLPClassifier.py`: Multi-Layer Perceptron classifier.

## Citation
```
@article{WANG2026115231,
title = {Adaptive weighting-guided dual-level contrastive learning for multi-view clustering},
journal = {Knowledge-Based Systems},
volume = {335},
pages = {115231},
year = {2026},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.115231},
url = {https://www.sciencedirect.com/science/article/pii/S0950705125022658},
author = {Shuangyang Wang and Lihua Zhou and Bing Kong},
keywords = {Multi-view clustering, Transformer, Contrastive learning, View fusion, Adaptive weighting},
abstract = {Multi-view clustering has demonstrated great promise by leveraging complementary and consistent information across views. However, two critical challenges hinder its applicability in real-world scenarios: (1) how to effectively integrate consistent and complementary information from diverse view spaces while reducing interference from low-quality or noisy views and (2) how to mitigate high-quality semantic labels or features being forced to align with low-quality ones derived from unreliable views in achieving maximal cross-view consistency. To address these challenges, we propose a novel framework, Adaptive Weighting-guided Dual-level Contrastive Learning for Multi-view Clustering (AWDC-MVC). Our framework first introduces a Transformer-based adaptive fusion module. This module leverages its multi-head self-attention mechanism to intrinsically filter noise and up-weight informative views. This process produces not only a high-quality global consensus representation but also, more crucially, the dynamic importance weights for each view. Subsequently, these adaptive weights are used to guide a dual-level contrastive learning: i) at the feature level, each view-consensus representation is aligned with the global consensus representation and ii) at the decision level, the semantic labels from individual views are aligned with their counterparts derived from the global consensus representation. This entire process forms a powerful synergistic loop, where the adaptive weights provide quality-aware guidance for the two alignment levels, while the dual-level alignment provides rich, hierarchical supervision to refine the fusion process. Extensive experiments on multiple benchmark datasets demonstrate the superiority and effectiveness of our proposed AWDC-MVC framework.}
}
```




