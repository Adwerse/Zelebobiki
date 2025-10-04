#!/usr/bin/env python3
"""
Shark appearance risk estimator built with scikit-learn.

Two-stage design:
1) Train regressors to predict environmental variables (e.g., SST & SSS).
2) Convert predicted env variables to a probability with a configurable formula.

- Robust to different sklearn versions (>=1.0).
- Handles missing values and basic numeric feature scaling automatically.
- Saves/loads fitted models via joblib.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# Keep imports stable across sklearn versions
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import joblib


# -------------------------- Config & formula --------------------------

@dataclass
class ModelConfig:
    feature_cols: List[str]
    target_sst: str = "sst"
    target_sss: str = "sss"
    random_state: int = 42
    n_estimators: int = 300
    max_depth: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def probability_formula(sst_c: float, sss_psu: float) -> float:
    """
    Editable formula: map predicted SST (°C) and SSS (PSU) to P(shark).
    Returns a number in [0, 1].

    Current baseline heuristic (purely illustrative):
    - Many shark sightings cluster in SST 18–27°C. Use a bell-shaped response around 23°C.
    - Slight salinity preference: near-oceanic salinity (~35 PSU) gets a mild boost.

    You can freely change this function without retraining models.
    """
    if np.isnan(sst_c) or np.isnan(sss_psu):
        return 0.0

    # SST preference: gaussian centered at 23°C with sigma ~4.5
    sst_center = 23.0
    sst_sigma = 4.5
    sst_score = math.exp(-0.5 * ((sst_c - sst_center) / sst_sigma) ** 2)

    # SSS preference: gaussian centered at 35 PSU with sigma ~2.5
    sss_center = 35.0
    sss_sigma = 2.5
    sss_score = math.exp(-0.5 * ((sss_psu - sss_center) / sss_sigma) ** 2)

    # Combine and squash to [0,1]
    raw = 0.65 * sst_score + 0.35 * sss_score
    # Optional logistic squashing
    prob = 1.0 / (1.0 + math.exp(-4.0 * (raw - 0.55)))
    # Clip to [0,1] for safety
    return max(0.0, min(1.0, prob))


# -------------------------- Core estimator --------------------------

class SharkRiskEstimator:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.sst_pipe: Optional[Pipeline] = None
        self.sss_pipe: Optional[Pipeline] = None

    def _make_regressor(self) -> Pipeline:
        numeric_features = self.config.feature_cols

        pre = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=[
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler(with_mean=False))  # with_mean=False to handle sparse safely
                ]), numeric_features),
            ],
            remainder="drop",
            sparse_threshold=0.3,
            n_jobs=None,  # parameter exists in newer sklearn; harmless if ignored
        )

        rf = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            random_state=self.config.random_state,
            n_jobs=-1,
        )

        pipe = Pipeline(steps=[("pre", pre), ("rf", rf)])
        return pipe

    def fit(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        required = set(self.config.feature_cols + [self.config.target_sst, self.config.target_sss])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Train SST
        df_sst = df[self.config.feature_cols + [self.config.target_sst]].dropna(subset=[self.config.target_sst])
        X_sst = df_sst[self.config.feature_cols]
        y_sst = df_sst[self.config.target_sst]
        X_tr_sst, X_te_sst, y_tr_sst, y_te_sst = train_test_split(X_sst, y_sst, test_size=0.2, random_state=self.config.random_state)
        self.sst_pipe = self._make_regressor()
        self.sst_pipe.fit(X_tr_sst, y_tr_sst)
        y_hat_sst = self.sst_pipe.predict(X_te_sst)
        report_sst = {
            "MAE": float(mean_absolute_error(y_te_sst, y_hat_sst)),
            "R2": float(r2_score(y_te_sst, y_hat_sst))
        }

        # Train SSS
        df_sss = df[self.config.feature_cols + [self.config.target_sss]].dropna(subset=[self.config.target_sss])
        X_sss = df_sss[self.config.feature_cols]
        y_sss = df_sss[self.config.target_sss]
        X_tr_sss, X_te_sss, y_tr_sss, y_te_sss = train_test_split(X_sss, y_sss, test_size=0.2, random_state=self.config.random_state)
        self.sss_pipe = self._make_regressor()
        self.sss_pipe.fit(X_tr_sss, y_tr_sss)
        y_hat_sss = self.sss_pipe.predict(X_te_sss)
        report_sss = {
            "MAE": float(mean_absolute_error(y_te_sss, y_hat_sss)),
            "R2": float(r2_score(y_te_sss, y_hat_sss))
        }

        return {"sst": report_sst, "sss": report_sss}

    def predict_env(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sst_pipe is None or self.sss_pipe is None:
            raise RuntimeError("Models are not fitted. Call fit() or load() first.")
        X = df[self.config.feature_cols].copy()
        sst_pred = self.sst_pipe.predict(X)
        sss_pred = self.sss_pipe.predict(X)
        out = df.copy()
        out["sst_pred"] = sst_pred
        out["sss_pred"] = sss_pred
        return out

    def risk_probability(self, sst_pred: np.ndarray, sss_pred: np.ndarray) -> np.ndarray:
        vec = np.vectorize(probability_formula, otypes=[float])
        return vec(sst_pred, sss_pred)

    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        out = self.predict_env(df)
        out["shark_prob"] = self.risk_probability(out["sst_pred"].values, out["sss_pred"].values)
        return out

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "sst_pipe": self.sst_pipe,
            "sss_pipe": self.sss_pipe
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "SharkRiskEstimator":
        payload = joblib.load(path)
        config = ModelConfig(**payload["config"])
        est = cls(config)
        est.sst_pipe = payload["sst_pipe"]
        est.sss_pipe = payload["sss_pipe"]
        return est


# -------------------------- CLI helpers --------------------------

def default_feature_cols(df: pd.DataFrame) -> List[str]:
    candidates = [c for c in ["decimalLatitude","decimalLongitude","day","month","year","depth","minimumDepthInMeters","maximumDepthInMeters","coordinateUncertaintyInMeters"] if c in df.columns]
    # make sure they are numeric
    good = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            good.append(c)
        else:
            with pd.option_context("mode.use_inf_as_na", True):
                df[c] = pd.to_numeric(df[c], errors="coerce")
                if pd.api.types.is_numeric_dtype(df[c]):
                    good.append(c)
    return good


def train_and_export(
    input_path: str,
    model_path: str,
    scored_csv_path: str,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    df = pd.read_csv(input_path, sep="\t")
    if feature_cols is None:
        feature_cols = default_feature_cols(df)

    cfg = ModelConfig(feature_cols=feature_cols)
    est = SharkRiskEstimator(cfg)
    reports = est.fit(df)

    # Save model
    est.save(model_path)

    # Score all rows that have features
    # For safety, fill missing numeric features with median before predicting
    use_cols = feature_cols.copy()
    for c in use_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df_pred = df.copy()
    # Let pipeline impute missing; only ensure columns exist and are numeric
    pred = est.predict_proba(df_pred[use_cols])
    out = df_pred.copy()
    out[["sst_pred","sss_pred","shark_prob"]] = pred[["sst_pred","sss_pred","shark_prob"]]
    out.to_csv(scored_csv_path, index=False)

    return {
        "feature_cols": feature_cols,
        "reports": reports,
        "n_scored": int(len(out)),
        "model_path": model_path,
        "scored_csv_path": scored_csv_path,
    }


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train shark risk estimator and score dataset.")
    p.add_argument("--input", required=True, help="Path to Occurrence.tsv or similar TSV file")
    p.add_argument("--model", required=True, help="Output path for the trained model (.joblib)")
    p.add_argument("--scored", required=True, help="Output CSV with predictions")
    p.add_argument("--features", nargs="*", default=None, help="Optional explicit feature column names")
    args = p.parse_args()

    result = train_and_export(args.input, args.model, args.scored, args.features)
    print(json.dumps(result, ensure_ascii=False, indent=2))
