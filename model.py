"""
model.py — TYPEGUARD
Dataset generation, model training, and similarity scoring.

Architecture:
  • Synthetic dataset generation per user profile
  • Random Forest classifier with probability output
  • Scaler + model persisted to disk
  • Similarity score = P(legitimate_user) × 100
  • Adaptive learning: retrain with new confirmed samples
"""

from __future__ import annotations
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report

from data_collection import simulate_typing_session, simulate_mouse_session
from feature_engineering import (
    BehavioralFeatures, extract_features, features_from_dict,
    FEATURE_NAMES, NUM_FEATURES
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# User Profile Definition
# ─────────────────────────────────────────────

class UserProfile:
    """
    Describes a user's typical behavioral statistics.
    Used to generate realistic synthetic training data.
    """
    def __init__(
        self,
        wpm_mean: float = 70,
        wpm_std: float = 8,
        hold_mean: float = 95,
        hold_std: float = 18,
        error_rate: float = 0.04,
        mouse_velocity_mean: float = 0.8,
        preferred_hours: Tuple[int, int] = (8, 22),
    ):
        self.wpm_mean = wpm_mean
        self.wpm_std = wpm_std
        self.hold_mean = hold_mean
        self.hold_std = hold_std
        self.error_rate = error_rate
        self.mouse_velocity_mean = mouse_velocity_mean
        self.preferred_hours = preferred_hours

# ─────────────────────────────────────────────
# Dataset Generator
# ─────────────────────────────────────────────

def generate_dataset(
    legitimate_profile: UserProfile,
    n_legit: int = 300,
    n_impostor: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a labelled dataset:
      label=1 → legitimate user
      label=0 → impostor (random behavioral profile)
    """
    rng = np.random.default_rng(seed)
    rows = []

    # ── Legitimate user samples ──────────────
    for _ in range(n_legit):
        wpm_m = float(rng.normal(legitimate_profile.wpm_mean, legitimate_profile.wpm_std * 0.3))
        ks = simulate_typing_session(
            wpm_mean=wpm_m,
            wpm_std=legitimate_profile.wpm_std,
            hold_mean_ms=legitimate_profile.hold_mean,
            hold_std_ms=legitimate_profile.hold_std,
            error_rate=legitimate_profile.error_rate,
            rng=rng,
        )
        ms = simulate_mouse_session(rng=rng)
        hour = int(rng.integers(*legitimate_profile.preferred_hours))
        dow = int(rng.integers(0, 7))
        feats = extract_features(ks, ms, hour=hour, day_of_week=dow)
        row = feats.to_dict()
        row['label'] = 1
        rows.append(row)

    # ── Impostor samples (diverse random profiles) ──
    for _ in range(n_impostor):
        imp_wpm = float(rng.uniform(30, 180))
        imp_hold = float(rng.uniform(50, 250))
        imp_err = float(rng.uniform(0, 0.20))
        ks = simulate_typing_session(
            wpm_mean=imp_wpm,
            wpm_std=imp_wpm * 0.15,
            hold_mean_ms=imp_hold,
            hold_std_ms=imp_hold * 0.2,
            error_rate=imp_err,
            rng=rng,
        )
        ms = simulate_mouse_session(num_moves=int(rng.integers(5, 60)), rng=rng)
        hour = int(rng.integers(0, 24))
        dow = int(rng.integers(0, 7))
        feats = extract_features(ks, ms, hour=hour, day_of_week=dow)
        row = feats.to_dict()
        row['label'] = 0
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

# ─────────────────────────────────────────────
# Model Trainer
# ─────────────────────────────────────────────

class BehavioralModel:
    """
    Wraps training, prediction, and persistence of the
    Random Forest behavioral classifier.
    """

    MODEL_VERSION = "1.0"

    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.is_trained: bool = False
        self.feature_importances_: Optional[np.ndarray] = None
        self._training_data: List[Dict] = []   # for adaptive learning

    # ── Training ──────────────────────────────

    def train(self, df: pd.DataFrame, cv: bool = True) -> Dict:
        """Train on a labelled DataFrame and return evaluation metrics."""
        X = df[FEATURE_NAMES].values.astype(np.float32)
        y = df['label'].values.astype(int)

        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        )
        calibrated_rf = CalibratedClassifierCV(rf, method='sigmoid', cv=3)
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', calibrated_rf),
        ])
        # Also train a plain RF for feature importances (CalibratedCV doesn't expose them)
        plain_rf = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1,
        )
        scaler_for_fi = StandardScaler()
        X_scaled = scaler_for_fi.fit_transform(X)
        plain_rf.fit(X_scaled, y)
        self.feature_importances_ = plain_rf.feature_importances_

        self.pipeline.fit(X, y)
        self.is_trained = True

        metrics = {}
        if cv:
            scores = cross_val_score(self.pipeline, X, y, cv=5, scoring='roc_auc')
            metrics['cv_auc_mean'] = float(scores.mean())
            metrics['cv_auc_std'] = float(scores.std())
        return metrics

    # ── Prediction ────────────────────────────

    def predict_proba(self, features: BehavioralFeatures) -> float:
        """Return probability of being the legitimate user (0.0 – 1.0)."""
        if not self.is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained yet.")
        X = features.to_array().reshape(1, -1)
        prob = self.pipeline.predict_proba(X)[0]
        # index 1 = legitimate class
        classes = self.pipeline.named_steps['clf'].classes_
        legit_idx = list(classes).index(1) if 1 in classes else 1
        return float(prob[legit_idx])

    def similarity_score(self, features: BehavioralFeatures) -> float:
        """Return match score as a percentage 0–100."""
        return round(self.predict_proba(features) * 100, 1)

    # ── Explainability ────────────────────────

    def explain(self, features: BehavioralFeatures) -> List[Dict]:
        """
        Return top-5 features driving the prediction (Explainable AI).
        Simple approach: multiply feature importance by normalised deviation.
        """
        if self.feature_importances_ is None:
            return []
        arr = features.to_array()
        scaler: StandardScaler = self.pipeline.named_steps['scaler']
        z_scores = np.abs(scaler.transform(arr.reshape(1, -1))[0])
        impact = self.feature_importances_ * z_scores
        top_idx = np.argsort(impact)[::-1][:5]
        return [
            {
                "feature": FEATURE_NAMES[i],
                "importance": round(float(self.feature_importances_[i]), 4),
                "z_score": round(float(z_scores[i]), 2),
                "impact": round(float(impact[i]), 4),
                "value": round(float(arr[i]), 3),
            }
            for i in top_idx
        ]

    # ── Adaptive Learning ─────────────────────

    def add_confirmed_sample(self, features: BehavioralFeatures, label: int):
        """Store a confirmed sample for the next retraining cycle."""
        row = features.to_dict()
        row['label'] = label
        self._training_data.append(row)

    def retrain_with_new_data(self, base_df: pd.DataFrame) -> Dict:
        """Merge stored confirmed samples with base data and retrain."""
        if not self._training_data:
            return {"status": "no_new_data"}
        new_df = pd.DataFrame(self._training_data)
        combined = pd.concat([base_df, new_df], ignore_index=True)
        metrics = self.train(combined, cv=False)
        self._training_data.clear()
        metrics['status'] = 'retrained'
        metrics['new_samples'] = len(new_df)
        return metrics

    # ── Persistence ───────────────────────────

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'feature_importances': self.feature_importances_,
                'version': self.MODEL_VERSION,
                'is_trained': self.is_trained,
            }, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.pipeline = data['pipeline']
        self.feature_importances_ = data['feature_importances']
        self.is_trained = data['is_trained']

# ─────────────────────────────────────────────
# Convenience: Build & train a fresh model
# ─────────────────────────────────────────────

def build_default_model(profile: Optional[UserProfile] = None) -> Tuple[BehavioralModel, pd.DataFrame]:
    """Generate synthetic data, train, and return model + dataset."""
    if profile is None:
        profile = UserProfile(wpm_mean=72, wpm_std=9, hold_mean=92, error_rate=0.035)
    df = generate_dataset(profile, n_legit=400, n_impostor=400)
    model = BehavioralModel()
    metrics = model.train(df)
    return model, df, metrics
