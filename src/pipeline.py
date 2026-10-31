"""
Preprocessing and feature engineering that mirror the training notebook exactly.

Every statistic used here (per-sensor medians, IQR bounds, scaler mean/scale, the
feature list) was fitted on the TRAINING machines inside the notebook and exported
to models/preprocessing.json. Nothing is re-estimated from uploaded data, so the
transform applied to new data is identical to the one the models were trained with.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLS = ["unit_id", "cycle", "timestamp", "shift", "machine_type"]
LABEL_COLS = ["RUL", "label_binary_30", "label_binary_15", "health_state", "failure_mode"]
TARGET = "RUL"


def infer_identifiers(raw: pd.DataFrame) -> pd.DataFrame:
    """Ensure unit_id and cycle exist. Single-machine files without ids are treated as one unit."""
    df = raw.copy()
    if "unit_id" not in df.columns:
        df["unit_id"] = 1
    if "cycle" not in df.columns:
        df["cycle"] = df.groupby("unit_id").cumcount() + 1
    return df


def check_columns(df: pd.DataFrame, sensor_cols: list[str]) -> list[str]:
    """Return the sensor columns that are missing from the upload."""
    return [c for c in sensor_cols if c not in df.columns]


def add_features(df: pd.DataFrame, key_sensors: list[str]) -> pd.DataFrame:
    """Causal per-machine features, identical to the notebook's add_features()."""
    g = df.groupby("unit_id")
    new = {}
    for c in key_sensors:
        new[f"{c}_mean5"] = g[c].transform(lambda s: s.rolling(5, min_periods=1).mean())
        new[f"{c}_std5"] = g[c].transform(lambda s: s.rolling(5, min_periods=2).std().fillna(0))
        new[f"{c}_mean15"] = g[c].transform(lambda s: s.rolling(15, min_periods=1).mean())
        new[f"{c}_diff1"] = g[c].diff().fillna(0)
        new[f"{c}_ewm10"] = g[c].transform(lambda s: s.ewm(span=10, adjust=False).mean())
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)


def prepare(raw: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Apply the training-time transforms to an uploaded frame.

    Returns (sorted frame with identifiers, scaled feature matrix aligned to its rows).
    """
    df = infer_identifiers(raw)
    df = df.drop_duplicates().sort_values(["unit_id", "cycle"]).reset_index(drop=True)

    S = cfg["sensor_cols"]
    missing = check_columns(df, S)
    if missing:
        raise ValueError(f"Upload is missing required sensor columns: {missing}")

    # Causal imputation within each machine, then training medians, then training IQR caps
    df[S] = df.groupby("unit_id")[S].transform(lambda s: s.ffill())
    med = pd.Series(cfg["median"])[S]
    lo = pd.Series(cfg["lower"])[S]
    hi = pd.Series(cfg["upper"])[S]
    df[S] = df[S].fillna(med).clip(lower=lo, upper=hi, axis=1)

    feat = add_features(df, cfg["key_sensors"])
    X = feat[cfg["features"]].to_numpy(dtype=np.float64)

    mean = np.asarray(cfg["scaler_mean"])
    scale = np.asarray(cfg["scaler_scale"])
    Xs = ((X - mean) / scale).astype(np.float32)
    return df, Xs


def windows_per_row(Xs: np.ndarray, unit_ids: np.ndarray, window: int) -> np.ndarray:
    """
    One causal window per row, ending at that row, zero-padded at the start of each
    machine's history. Same construction as the notebook's make_sequences().
    """
    out = np.zeros((len(Xs), window, Xs.shape[1]), dtype=np.float32)
    for uid in pd.unique(unit_ids):
        idx = np.where(unit_ids == uid)[0]
        F = Xs[idx]
        Fp = np.vstack([np.zeros((window - 1, F.shape[1]), dtype=np.float32), F])
        for k, i in enumerate(idx):
            out[i] = Fp[k:k + window]
    return out


def history_lengths(unit_ids: np.ndarray) -> np.ndarray:
    """Number of rows of history available for each row's machine up to and including that row."""
    s = pd.Series(unit_ids)
    return (s.groupby(s).cumcount() + 1).to_numpy()
