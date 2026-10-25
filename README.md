# Fashion-MNIST: MLP Baseline vs CNN

Train two models from scratch to classify the 10 Fashion-MNIST categories: an MLP baseline and a small CNN.

## Research question
Does a small CNN outperform an MLP baseline on the held-out Fashion-MNIST test set?

## Dataset and split
- Fashion-MNIST has 70,000 grayscale 28×28 images across 10 classes.
- Source: https://github.com/zalandoresearch/fashion-mnist
- Paper: Xiao, Rasul & Vollgraf (2017), https://arxiv.org/abs/1708.07747
- The original 60,000-image training partition is split deterministically into 50,000 train and 10,000 validation images using seed 42. The official 10,000-image test split is used only in `evaluate.py`, after training/model selection. Do not tune hyperparameters on test results.
- Dataset files are downloaded automatically and must not be committed to GitHub.

## Training choices
- Models built from scratch; no pretrained weights.
- Loss: cross entropy; optimizer: Adam.
- Learning rate: 0.001; batch size: 128; maximum epochs: 8.
- Regularization: dropout 0.3, weight decay 0.0001, early stopping with patience 3.
- Fixed random seed: 42. Document any changes to these settings in the report.

## Setup and reproduction
Python 3.10+ recommended.

```bash
pip install -r requirements.txt
python train.py --epochs 8
python evaluate.py
```

`train.py` downloads data into `./data/`, trains both models, and writes checkpoints and histories to `./outputs/`. `evaluate.py` reports test accuracy and macro-F1, and saves a CNN confusion matrix and predictions to `./evaluation/`.

## Report checklist
Include dataset source/size/split, architecture diagram or layer description, training choices, test result table, learning curves, at least one concrete failure case from test predictions, limitations and next steps. Use only results from your own run; do not fabricate metrics.

## References
- Xiao, H., Rasul, K., & Vollgraf, F. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* https://arxiv.org/abs/1708.07747
- Fashion-MNIST repository: https://github.com/zalandoresearch/fashion-mnist
- PyTorch docs: https://pytorch.org/docs/stable/index.html
