# Trained model artifacts

This folder is filled by the export cell at the end of `notebook/RUL_CNC_v4_export.ipynb`.

Expected contents after export:

| File | Purpose |
|---|---|
| `preprocessing.json` | Training-fitted medians, IQR bounds, scaler parameters, feature list and model registry. **Required.** |
| `decision_tree.joblib`, `random_forest.joblib`, `gradient_boosting.joblib` | Tabular models |
| `cnn_lstm_seed42.keras`, `cnn_lstm_seed43.keras`, `cnn_lstm_seed44.keras` | CNN-LSTM, one file per seed (predictions are averaged) |
| `lstm_seed42.keras`, `lstm_seed43.keras`, `lstm_seed44.keras` | LSTM, one file per seed |

Copy the whole `outputs/models/` folder from Colab into this directory.
The `.joblib` files must be loaded with the same scikit-learn version used to create them, which is recorded in `preprocessing.json`.
