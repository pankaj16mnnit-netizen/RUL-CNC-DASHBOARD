# CNC Machining Centre — RUL Prediction Dashboard

Interactive Streamlit dashboard for the M.Tech AI/ML dissertation *Predictive Health Monitoring of Manufacturing Machinery: Remaining Useful Life (RUL) Estimation from Multi-Sensor Data*.

The dashboard has four tabs:

1. **Model comparison**: test-set R², RMSE, MAE and asymmetric score for Decision Tree, Random Forest, Gradient Boosting, LSTM and CNN-LSTM.
2. **Test predictions**: predicted against actual RUL for each held-out machine, with metrics recomputed from the exported predictions.
3. **Predict RUL (upload)**: upload sensor history as CSV and get RUL predictions from every available model, with per-machine trajectories and a downloadable CSV.
4. **About**: method and limitations.

## Repository layout

```
rul-cnc-dashboard/
├── app.py                  Streamlit dashboard
├── src/
│   ├── pipeline.py         Preprocessing and features (mirrors the notebook exactly)
│   └── inference.py        Loads exported models and predicts
├── models/                 Exported artifacts (from the notebook)
├── results/
│   ├── model_comparison.csv   Final comparison table from the executed run
│   └── test_predictions.csv   Per-row test predictions (copy from notebook outputs)
├── data/                   Sample upload and data notes
├── notebook/
│   └── RUL_CNC_v4_export.ipynb   Training notebook (Colab) with export cell
├── docs/
│   └── CNC_RUL_Final_Report.docx Final report
├── requirements.txt
└── .gitignore
```

## Step 1: Train and export (Google Colab)

1. Open `notebook/RUL_CNC_v4_export.ipynb` in Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Upload `cnc_machining_centre_rul_dataset (2).csv` when prompted.
4. Runtime → Run all. The default `FAST = True` takes about 30 minutes; set it to `False` for the full protocol.
5. Download the `outputs/` folder. Copy `outputs/models/` into this repository's `models/`, and copy `outputs/test_predictions.csv` and `outputs/model_comparison.csv` into `results/`.

## Step 2: Run locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501. Without TensorFlow it still shows the comparison and all tabular predictions. Install `tensorflow-cpu` to enable CNN-LSTM and LSTM predictions.

## Step 3: Push to GitHub

```bash
git init
git add .
git commit -m "CNC RUL dashboard: model comparison and RUL prediction from uploaded sensor data"
git branch -M main
git remote add origin https://github.com/<your-username>/rul-cnc-dashboard.git
git push -u origin main
```

## Step 4 (optional): Deploy

Streamlit Community Cloud can deploy directly from the GitHub repository. Select `app.py` as the entry point. Deployment without TensorFlow is lighter; remove `tensorflow-cpu` from `requirements.txt` if the deployment exceeds its size limit.

## How uploaded data is used

- The upload needs the same sensor columns as the training data. `unit_id` and `cycle` are optional: without them, the file is treated as one machine in row order.
- Rolling, trend and imputation steps use each machine's earlier rows only, so the prediction for a cycle depends only on the history up to that cycle.
- A single row can be scored, but rolling and trend features then carry no real history. The dashboard shows a warning in that case, and the prediction is indicative only.
- Medians, IQR bounds and scaler statistics come from the training machines (stored in `preprocessing.json`). Nothing is re-estimated from uploaded data.

## Limitations

- The dataset is synthetic; generalisation to real plant data is not demonstrated.
- The deep models were trained in FAST mode (8 tuning trials, 3 seeds).
- The CNN-LSTM margin over the tuned LSTM is within seed-to-seed variation.
