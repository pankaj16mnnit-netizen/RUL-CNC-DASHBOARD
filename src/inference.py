"""
Loads exported artifacts and produces RUL predictions from every available model.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.pipeline import history_lengths, prepare, windows_per_row

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


class ModelBundle:
    """Holds the preprocessing config and every model listed in it."""

    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = Path(model_dir)
        cfg_path = self.model_dir / "preprocessing.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} not found. Run the notebook export cell and copy outputs/models/ here.")
        self.cfg = json.loads(cfg_path.read_text())
        self.tabular: dict[str, object] = {}
        self.deep: dict[str, list] = {}
        self.warnings: list[str] = []
        self._load_tabular()

    @property
    def entries(self) -> list[dict]:
        return self.cfg["models"]

    def _load_tabular(self) -> None:
        for e in self.entries:
            if e["kind"] != "tabular":
                continue
            path = self.model_dir / e["files"][0]
            if path.exists():
                self.tabular[e["name"]] = joblib.load(path)
            else:
                self.warnings.append(f"Missing file for {e['name']}: {path.name}")

    def load_deep(self) -> None:
        """Load Keras models. Kept separate so the app still works without TensorFlow."""
        deep_entries = [e for e in self.entries if e["kind"] == "deep"]
        if not deep_entries:
            return
        try:
            import tensorflow as tf  # noqa: F401
            from tensorflow import keras
        except ImportError:
            self.warnings.append("TensorFlow is not installed: CNN-LSTM and LSTM predictions are unavailable.")
            return
        for e in deep_entries:
            models = []
            for fn in e["files"]:
                path = self.model_dir / fn
                if path.exists():
                    models.append(keras.models.load_model(path, compile=False))
            if models:
                self.deep[e["name"]] = models
            else:
                self.warnings.append(f"No weight files found for {e['name']}.")

    def available_models(self) -> list[str]:
        return list(self.tabular) + list(self.deep)

    def predict(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Predict RUL for every row of the upload, with one column per available model.

        Columns: unit_id, cycle, history_rows, <model name>..., and RUL if the upload has it.
        """
        df, Xs = prepare(raw, self.cfg)
        unit_ids = df["unit_id"].to_numpy()
        out = pd.DataFrame({
            "unit_id": unit_ids,
            "cycle": df["cycle"].to_numpy(),
            "history_rows": history_lengths(unit_ids),
        })
        if "RUL" in df.columns:
            out["RUL_actual"] = df["RUL"].to_numpy()

        for name, model in self.tabular.items():
            out[name] = np.asarray(model.predict(Xs)).ravel()

        for e in self.entries:
            if e["kind"] != "deep" or e["name"] not in self.deep:
                continue
            W = windows_per_row(Xs, unit_ids, e["window"])
            preds = [m.predict(W, batch_size=512, verbose=0).ravel() for m in self.deep[e["name"]]]
            out[e["name"]] = np.mean(preds, axis=0)
        return out

    def quality_notes(self, pred: pd.DataFrame) -> list[str]:
        """Plain-language warnings about how much history each prediction is based on."""
        notes = []
        deep_windows = {e["name"]: e["window"] for e in self.entries if e["kind"] == "deep"}
        last = pred.groupby("unit_id")["history_rows"].max()
        short_units = last[last < 15].index.tolist()
        if short_units:
            notes.append(f"Units {short_units} have fewer than 15 cycles of history. Rolling and trend "
                         "features are unreliable at this length, so predictions are indicative only.")
        for name, w in deep_windows.items():
            if name in pred.columns and (last < w).any():
                notes.append(f"{name} expects {w} cycles of history; units with less are zero-padded, "
                             "as in training, which lowers reliability.")
        return notes
