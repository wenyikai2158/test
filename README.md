# test

A repository for storing practice code.

## Contents

- `pathfinding_lab.py`: A* algorithm and its improvements, including
  - weight A*
  - different heuristics functions
  - 2 kinds of tricks
- `minimum snap.ipynb`: polynomial trajectory optimization notes.
- `transformer-weather-prediction/`: deep learning course project — Transformer-based
  multi-step weather forecasting (see below).

## Transformer Weather Forecasting

A deep learning course project (Group 15) that forecasts hourly temperature from the
**Jena Climate 2009–2016** dataset using Transformer-based sequence models, and
benchmarks them against LSTM and ARIMA baselines.

### Task setup

- **Dataset**: Jena Climate (`jena_climate_2009_2016.csv`), recorded at a 10-minute
  interval. Samples are subsampled hourly and the most recent 10,000 hours are used.
- **Input features** (11): `p (mbar)`, `T (degC)`, `Tpot (K)`, `Tdew (degC)`, `rh (%)`,
  `VPmax (mbar)`, `VPact (mbar)`, `VPdef (mbar)`, `sh (g/kg)`, `H2OC (mmol/mol)`,
  `rho (g/m**3)`.
- **Target**: `T (degC)`.
- **Windowing**: a 96-hour history window is used to predict the next 24 hours
  (multi-step forecasting). Inputs and targets are standardized with `StandardScaler`.
- **Split**: 70% train / 10% validation / 20% test, in chronological order.

### Files

| File | Description |
| --- | --- |
| `DeepLearning.py` | Single-step baseline. A Transformer encoder with sinusoidal positional encoding predicts the temperature one hour ahead (48-hour input window). Includes training with AdamW + cosine annealing, early stopping, gradient clipping, test-set MAE in degrees Celsius, and an actual-vs-predicted plot. |
| `ARIMA,LSTM,Transformer.py` | The main multi-step comparison. Trains the Transformer to output the whole future 24-hour curve directly (`[batch, 24]`), then benchmarks it against an LSTM baseline and a traditional ARIMA(2,1,0) rolling-forecast model, and draws an MAE bar chart for all three. |
| `improved_models.py` | A model zoo of Transformer variants: sparse attention (local / strided / random patterns), local + global hybrid attention with a learnable fusion weight, a CNN-Transformer hybrid, a graph-attention Transformer, and a combined "improved" model with relative positional encoding. Exposes a `create_improved_model(model_type, input_dim, **kwargs)` factory. |
| `hyperparameter_tuning.py` | Systematic hyperparameter search over learning rate, `d_model`, number of attention heads, encoder depth, and dropout. Each configuration is trained and evaluated with MSE / MAE / RMSE, results are written to JSON, and a six-panel comparison figure is saved. |

### Model architecture

The core model (`MultiStepWeatherTransformer` / `WeatherTransformer`) is a
Transformer **encoder-only** network:

1. Linear input projection from the 11 features to `d_model` (128).
2. Sinusoidal positional encoding.
3. `num_layers` (3) `TransformerEncoderLayer` blocks with `nhead` (8) heads,
   pre-layer normalization, and dropout 0.2.
4. The last time step's hidden state is decoded by a small MLP head into either
   1 value (single-step) or 24 values (multi-step).

Training uses AdamW (`lr=1e-3`, `weight_decay=1e-4`), cosine annealing to `1e-5`,
MSE loss, gradient norm clipping at 1.0, and early stopping on validation loss.

### Results

Predictions are de-standardized back to degrees Celsius before evaluation, so all
errors are reported in physical units. The head-to-head comparison on 24-hour
multi-step forecasting (Transformer vs. LSTM vs. ARIMA) is produced as a bar chart by
`ARIMA,LSTM,Transformer.py`.

### Running

```bash
pip install torch numpy pandas scikit-learn matplotlib statsmodels

# The dataset downloads itself on first run (~13 MB)
python DeepLearning.py                     # single-step baseline
python "ARIMA,LSTM,Transformer.py"         # multi-step + ARIMA/LSTM comparison
```

### Notes

- `hyperparameter_tuning.py` imports `TransformerModel` and `create_sequences` from a
  module named `main_complete`, which is **not included** in this folder; that script
  needs those helpers supplied before it can run.
- The dataset, model checkpoints (`*.pth`), and generated figures are not tracked in
  this repository.
- Documentation and comments inside the source files are in Chinese.

## Git workflow

Remote: https://github.com/wenyikai2158/test.git (HTTPS, credential helper = manager)

```bash
git pull --rebase origin main   # fetch remote changes first
git add -A
git commit -m "your message"
git push origin main
```
