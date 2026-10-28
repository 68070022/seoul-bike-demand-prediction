# Seoul Bike Rental Demand Prediction with an MLP

Predict hourly bike rental demand using the public [UCI Seoul Bike Sharing Demand dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand).

## Research question
Can a small MLP improve demand prediction over a Linear Regression baseline?

## Method
- Target: `Rented Bike Count`
- Model built and trained here: MLP (128 and 64 hidden units, ReLU, Dropout)
- Baseline: Linear Regression
- Loss / optimizer: MSE / Adam
- Regularization: Dropout, weight decay, early stopping
- Split: chronological 70% train / 15% validation / 15% test
- Metrics: MAE, RMSE, R²
- Random seed: 42

The preprocessing transformer is fitted on training data only. The test set is held out for final evaluation.

## Dataset citation
S. H. Lee, “Seoul Bike Sharing Demand,” UCI Machine Learning Repository, 2020. https://doi.org/10.24432/C5F62R

The scripts download the dataset automatically; the dataset is not committed to this repository.

## Setup
Python 3.10+ recommended.
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python train.py
python evaluate.py
```
`train.py` downloads and prepares data, fits preprocessing on train only, trains the MLP with validation-based early stopping, trains the baseline, and saves artifacts locally. `evaluate.py` reports held-out test metrics and saves failure cases and plots.

## Files
- `model.py`: MLP architecture
- `train.py`: data preparation, chronological split, training loop, baseline, saved artifacts
- `evaluate.py`: held-out evaluation, metrics, failure cases, plots
- `requirements.txt`: dependencies

## Reproducibility and honest reporting
The scripts set seed 42 and use a chronological split. Report the metrics from your own run; results may differ slightly by hardware/library version. Lower MAE/RMSE are better; higher R² is better. It is valid if the MLP does not outperform the baseline on every metric. Discuss learning curves and largest-error examples in the report.
