"""
CNC Machining Centre — Remaining Useful Life (RUL) dashboard.

Tabs
  1. Model comparison   : final test-set metrics for all five models
  2. Test predictions   : per-machine predicted vs actual RUL from the notebook export
  3. Predict RUL       : upload sensor history, get RUL from every available model
  4. About             : method and limitations

Run locally:  streamlit run app.py
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.inference import ModelBundle

ROOT = Path(__file__).resolve().parent
RESULTS_FILE = ROOT / "results" / "model_comparison.csv"
TEST_PRED_FILE = ROOT / "results" / "test_predictions.csv"
SAMPLE_FILE = ROOT / "data" / "sample_upload.csv"
RUL_CLIP = 130

st.set_page_config(page_title="CNC RUL Dashboard", page_icon="⚙️", layout="wide")


# ---------- cached loaders ----------
@st.cache_data
def load_results() -> pd.DataFrame | None:
    if RESULTS_FILE.exists():
        return pd.read_csv(RESULTS_FILE)
    return None


@st.cache_data
def load_test_predictions() -> pd.DataFrame | None:
    if TEST_PRED_FILE.exists():
        return pd.read_csv(TEST_PRED_FILE)
    return None


@st.cache_resource(show_spinner="Loading models...")
def load_bundle() -> tuple[ModelBundle | None, str | None]:
    try:
        bundle = ModelBundle()
    except FileNotFoundError as exc:
        return None, str(exc)
    bundle.load_deep()
    return bundle, None


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    d = p - y
    asym = float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))
    return {"R2": 1 - ss_res / ss_tot, "RMSE": float(np.sqrt(np.mean(err ** 2))),
            "MAE": float(np.mean(np.abs(err))), "AsymScore": asym}


def fig_to_st(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


# ---------- header ----------
st.title("Remaining Useful Life — CNC Machining Centre")
st.caption("Comparison of Decision Tree, Random Forest, Gradient Boosting, LSTM and CNN-LSTM models "
           "on a 100-machine multi-sensor run-to-failure dataset.")

tab_cmp, tab_test, tab_pred, tab_about = st.tabs(
    ["Model comparison", "Test predictions", "Predict RUL (upload)", "About"])

# ---------- 1. Model comparison ----------
with tab_cmp:
    res = load_results()
    if res is None:
        st.warning("results/model_comparison.csv not found.")
    else:
        st.subheader("Held-out test performance")
        st.caption("Deep models: mean ± std over seeds. Tabular models: single fit. "
                   "Test set: 20 held-out machines, 8,387 rows.")
        def fmt(v, sd, nd):
            if pd.isna(v):
                return "-"
            txt = f"{v:.{nd}f}"
            if sd is not None and not pd.isna(sd):
                txt += f" ± {sd:.{nd}f}"
            return txt
        show = pd.DataFrame({
            "Model": res["Model"],
            "R²": [fmt(r.R2, getattr(r, "R2_std", None), 3) for r in res.itertuples()],
            "RMSE (cycles)": [fmt(r.RMSE, getattr(r, "RMSE_std", None), 2) for r in res.itertuples()],
            "MAE (cycles)": [fmt(r.MAE, None, 2) for r in res.itertuples()],
            "Asymmetric score": [f"{r.AsymScore:,.1f}" for r in res.itertuples()],
        })
        st.dataframe(show, use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ordered = res.sort_values("R2", ascending=False)
            ax.bar(ordered["Model"], ordered["R2"], color="#3F7F93")
            ax.set_ylabel("R²"); ax.set_ylim(0.85, 1.0); ax.set_title("Test R² (higher is better)")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right"); fig.tight_layout()
            fig_to_st(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(6, 4))
            ordered = res.sort_values("RMSE")
            ax.bar(ordered["Model"], ordered["RMSE"], color="#B5523B")
            ax.set_ylabel("RMSE (cycles)"); ax.set_title("Test RMSE (lower is better)")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right"); fig.tight_layout()
            fig_to_st(fig)

        st.info("CNN-LSTM has the highest mean R² and lowest RMSE and asymmetric score. Its margin over the "
                "tuned LSTM is within seed-to-seed variation, so the two sequence models are statistically "
                "comparable on this dataset. The asymmetric score penalises late predictions more heavily.")

# ---------- 2. Test predictions ----------
with tab_test:
    tp = load_test_predictions()
    if tp is None:
        st.warning("results/test_predictions.csv not found. Copy it from the notebook's outputs folder.")
    else:
        pred_cols = [c for c in tp.columns if c.startswith("pred_")]
        names = {c: c.replace("pred_", "").replace("_", " ") for c in pred_cols}
        st.subheader("Predicted vs actual RUL on held-out machines")

        machines = sorted(tp["unit_id"].unique().tolist())
        unit = st.selectbox("Machine", machines, index=0, key="test_unit")
        sub = tp[tp["unit_id"] == unit].sort_values("cycle")

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(sub["cycle"], sub["RUL_clipped"], "k-", lw=2.5, label="Actual")
        for c in pred_cols:
            ax.plot(sub["cycle"], sub[c], "--", lw=1.2, label=names[c])
        ax.set_xlabel("Cycle"); ax.set_ylabel("RUL (clipped at 130)")
        ax.set_title(f"Machine #{unit}"); ax.legend(ncol=3, fontsize=8)
        fig.tight_layout(); fig_to_st(fig)

        st.subheader("Metrics recomputed from this file")
        st.caption("Recomputed here from the exported predictions, as a check against the comparison tab.")
        rows = []
        for c in pred_cols:
            m = regression_metrics(tp["RUL_clipped"].to_numpy(), tp[c].to_numpy())
            rows.append({"Model": names[c], **m})
        st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)

# ---------- 3. Predict RUL ----------
with tab_pred:
    st.subheader("Predict RUL from uploaded sensor data")
    st.markdown(
        "Upload a CSV with the same sensor columns as the training data. The file should contain `unit_id` "
        "and `cycle` so each machine's history can be built. A row on its own can be scored, but rolling and "
        "trend features need several cycles of history, so predictions from a short history are less reliable. "
        "Any `RUL` column in the upload is used only to compare predictions against actuals.")

    bundle, bundle_err = load_bundle()
    if bundle_err:
        st.error(bundle_err)
    if bundle is not None:
        for w in bundle.warnings:
            st.warning(w)
        st.write("**Models available:** " + ", ".join(bundle.available_models()))

        col_a, col_b = st.columns([3, 1])
        with col_a:
            uploaded = st.file_uploader("Sensor data (CSV)", type=["csv"])
        with col_b:
            use_sample = st.checkbox("Use bundled sample", value=False,
                                     help="Two machines: one with full history and one with 10 cycles.")

        raw = None
        if uploaded is not None:
            raw = pd.read_csv(uploaded)
        elif use_sample and SAMPLE_FILE.exists():
            raw = pd.read_csv(SAMPLE_FILE)

        if raw is None:
            st.info("Upload a file or tick 'Use bundled sample' to run predictions.")
        else:
            st.write(f"Loaded **{len(raw):,} rows** from {raw['unit_id'].nunique() if 'unit_id' in raw else 1} machine(s).")
            try:
                pred = bundle.predict(raw)
            except ValueError as exc:
                st.error(str(exc))
                pred = None

            if pred is not None:
                model_cols = bundle.available_models()
                for note in bundle.quality_notes(pred):
                    st.warning(note)

                mode = st.radio("Show", ["Latest cycle per machine", "Every cycle"], horizontal=True)
                if mode == "Latest cycle per machine":
                    view = pred.sort_values(["unit_id", "cycle"]).groupby("unit_id").tail(1)
                else:
                    view = pred

                st.subheader("Predicted RUL (cycles remaining)")
                shown = view[["unit_id", "cycle", "history_rows"] + model_cols + (["RUL_actual"] if "RUL_actual" in view else [])]
                st.dataframe(shown.round(1), use_container_width=True, hide_index=True)

                if "RUL_actual" in pred.columns and mode == "Latest cycle per machine":
                    st.caption("Actual RUL is present in the upload, so the latest-cycle error is shown below.")
                    err = pd.DataFrame({m: view[m] - view["RUL_actual"] for m in model_cols})
                    err.insert(0, "unit_id", view["unit_id"].values)
                    st.dataframe(err.round(1), use_container_width=True, hide_index=True)

                st.subheader("RUL trajectory for one machine")
                unit = st.selectbox("Machine", sorted(pred["unit_id"].unique().tolist()), key="pred_unit")
                sub = pred[pred["unit_id"] == unit].sort_values("cycle")
                fig, ax = plt.subplots(figsize=(10, 4.5))
                if "RUL_actual" in sub:
                    ax.plot(sub["cycle"], sub["RUL_actual"], "k-", lw=2.5, label="Actual")
                for m in model_cols:
                    ax.plot(sub["cycle"], sub[m], "--", lw=1.2, label=m)
                ax.set_xlabel("Cycle"); ax.set_ylabel("Predicted RUL (cycles)"); ax.legend(ncol=3, fontsize=8)
                ax.set_title(f"Machine #{unit}")
                fig.tight_layout(); fig_to_st(fig)

                csv_buf = StringIO()
                pred.to_csv(csv_buf, index=False)
                st.download_button("Download all predictions (CSV)", csv_buf.getvalue(),
                                   file_name="rul_predictions.csv", mime="text/csv")

# ---------- 4. About ----------
with tab_about:
    st.subheader("Method")
    st.markdown("""
- **Data:** 100 synthetic CNC Machining Centres, run to failure, 26 sensor and operational channels.
- **Split:** 80 training machines and 20 held-out test machines, split by machine, not by row.
- **Cleaning:** forward-fill within each machine, training medians, IQR caps fitted on training machines only.
- **Features:** 26 raw channels plus rolling mean and std, first difference and EWMA for 10 degradation sensors (76 total).
- **Target:** RUL clipped at 130 cycles during training.
- **Models:** Decision Tree, Random Forest, Gradient Boosting (tabular). LSTM and CNN-LSTM (30-cycle windows, averaged over seeds).
- **Metrics:** R², RMSE, MAE and a PHM08-style asymmetric score that penalises late predictions more.
""")
    st.subheader("Limitations")
    st.markdown("""
- The dataset is synthetic. Performance on real plant data has not been demonstrated.
- The deep models were tuned with 8 trials and evaluated over 3 seeds (FAST mode). Results under the full protocol may differ.
- The CNN-LSTM margin over the tuned LSTM is within seed variation.
- Predictions from short histories are less reliable, as described on the upload tab.
""")
