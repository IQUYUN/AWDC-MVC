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

If you use this code for your research, please cite our paper:

```
[Please insert your citation here]
```


