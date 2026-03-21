"""
ML Direction Agent — RF + KNN Ensemble for Per-Asset Direction Prediction
Inspired by: User's own Metal Futures project (RF+KNN ensemble)
             + LSTM project (bull_ratio, ATR filter, SMA trend filter)

For each asset: predict next-period direction using:
  - Random Forest regression → sign = direction
  - KNN classifier → direct direction classification
  - Ensemble rule: agree → follow; disagree → default to KNN

Includes ATR filter (from LSTM project): skip low-vol assets.
Includes SMA20 trend filter: adds conviction when aligned.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from agents.base_agent import BaseAgent, Signal

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class MLDirectionAgent(BaseAgent):
    """
    Per-asset direction prediction using RF + KNN ensemble.
    Outputs: direction score (-1 to +1) and confidence for each asset.
    """

    def __init__(self, retrain_every: int = 63, min_train_samples: int = 200):
        super().__init__("MLDirection")
        self.retrain_every = retrain_every
        self.min_train_samples = min_train_samples
        self._models: Dict[str, Dict] = {}  # ticker → {rf, knn, scaler, last_train}
        self._feature_cols = [
            "ret_1d", "ret_5d", "ret_20d",
            "ma_gap_20", "ma_gap_50",
            "vol_10d", "vol_ratio",
            "rsi_proxy", "momentum_accel",
        ]

    def _build_asset_features(self, series: pd.Series, idx: int) -> Optional[np.ndarray]:
        """Build feature vector for a single asset at index idx."""
        if idx < 60:
            return None

        s = series.iloc[:idx + 1].values
        if len(s) < 60:
            return None

        ret_1d = s[-1] / s[-2] - 1 if s[-2] != 0 else 0
        ret_5d = s[-1] / s[-6] - 1 if s[-6] != 0 else 0
        ret_20d = s[-1] / s[-21] - 1 if len(s) >= 21 and s[-21] != 0 else 0

        ma20 = np.mean(s[-20:])
        ma50 = np.mean(s[-50:]) if len(s) >= 50 else np.mean(s)
        ma_gap_20 = (s[-1] - ma20) / ma20 if ma20 != 0 else 0
        ma_gap_50 = (s[-1] - ma50) / ma50 if ma50 != 0 else 0

        rets = np.diff(s[-21:]) / s[-21:-1]
        rets = rets[~np.isnan(rets)]
        vol_10d = np.std(rets[-10:]) * np.sqrt(252) if len(rets) >= 10 else 0.15
        vol_60d = np.std(rets) * np.sqrt(252) if len(rets) >= 20 else vol_10d
        vol_ratio = vol_10d / max(vol_60d, 0.001)

        # RSI proxy
        recent = rets[-14:] if len(rets) >= 14 else rets
        rsi_proxy = np.sum(recent > 0) / max(len(recent), 1)

        # Momentum acceleration
        mom5_now = ret_5d
        mom5_prev = s[-6] / s[-11] - 1 if len(s) >= 11 and s[-11] != 0 else 0
        momentum_accel = mom5_now - mom5_prev

        return np.array([
            ret_1d, ret_5d, ret_20d,
            ma_gap_20, ma_gap_50,
            vol_10d, vol_ratio,
            rsi_proxy, momentum_accel,
        ])

    def _train_asset_model(self, series: pd.Series, up_to_idx: int) -> Optional[Dict]:
        """Train RF and KNN models for a single asset."""
        X_list = []
        y_reg_list = []  # next-day return (for RF)
        y_cls_list = []  # next-day direction (for KNN)

        # Use positional indexing on the full series
        n = len(series)
        start = 60
        end = min(up_to_idx, n - 1)  # ensure we don't go out of bounds

        for i in range(start, end):
            feat = self._build_asset_features(series, i)
            if feat is None:
                continue

            cur_price = series.iloc[i]
            next_price = series.iloc[i + 1]
            if cur_price == 0 or pd.isna(cur_price) or pd.isna(next_price):
                continue

            next_ret = next_price / cur_price - 1

            if np.any(np.isnan(feat)) or np.isnan(next_ret):
                continue

            X_list.append(feat)
            y_reg_list.append(next_ret)
            y_cls_list.append(1 if next_ret > 0 else 0)

        if len(X_list) < self.min_train_samples:
            return None

        X = np.array(X_list)
        y_reg = np.array(y_reg_list)
        y_cls = np.array(y_cls_list)

        # Standardize for KNN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # RF Regressor → predict return magnitude, then take sign
        rf = RandomForestRegressor(
            n_estimators=30, max_depth=3, min_samples_leaf=40,
            random_state=42, n_jobs=-1,
        )
        rf.fit(X, y_reg)

        # KNN Classifier → predict direction directly
        knn = KNeighborsClassifier(n_neighbors=31, weights="distance")
        knn.fit(X_scaled, y_cls)

        return {"rf": rf, "knn": knn, "scaler": scaler, "last_train": up_to_idx}

    def _predict_asset(self, ticker: str, series: pd.Series, idx: int) -> Tuple[float, float]:
        """
        Predict direction for one asset. Returns (direction, confidence).
        direction: +1 (bullish), -1 (bearish), 0 (no signal)
        confidence: 0.0 to 1.0
        """
        model_info = self._models.get(ticker)
        if model_info is None:
            return 0.0, 0.0

        feat = self._build_asset_features(series, idx)
        if feat is None or np.any(np.isnan(feat)):
            return 0.0, 0.0

        rf = model_info["rf"]
        knn = model_info["knn"]
        scaler = model_info["scaler"]

        # RF prediction: predict return, take sign
        rf_pred_return = rf.predict(feat.reshape(1, -1))[0]
        rf_direction = 1 if rf_pred_return > 0 else -1

        # KNN prediction
        feat_scaled = scaler.transform(feat.reshape(1, -1))
        knn_proba = knn.predict_proba(feat_scaled)[0]
        knn_direction = 1 if knn_proba[1] > 0.5 else -1
        knn_confidence = max(knn_proba)

        # Ensemble rule (from Metal Futures project):
        # If RF and KNN agree → follow that signal with higher confidence
        # If they disagree → default to KNN (more robust)
        if rf_direction == knn_direction:
            direction = float(rf_direction)
            confidence = min(knn_confidence + 0.1, 0.95)  # agreement bonus
        else:
            direction = float(knn_direction)
            confidence = knn_confidence * 0.8  # disagreement penalty

        # ATR filter (from LSTM project): reduce confidence in low-vol conditions
        s = series.iloc[:idx + 1].values
        if len(s) >= 20:
            rets = np.diff(s[-20:]) / s[-20:-1]
            atr_pct = np.mean(np.abs(rets)) if len(rets) > 0 else 0
            if atr_pct < 0.005:  # < 0.5% average daily range = noise
                confidence *= 0.3  # heavily penalize low-vol signals

        # SMA20 trend filter (from LSTM project):
        # If predicting short but price above SMA20, reduce confidence
        if len(s) >= 20:
            sma20 = np.mean(s[-20:])
            if direction < 0 and s[-1] > sma20:
                confidence *= 0.5  # don't short in uptrends

        return direction, round(confidence, 4)

    def analyze(self, date: pd.Timestamp, prices: pd.DataFrame,
                returns: pd.DataFrame, context: Dict[str, Any]) -> Signal:
        """
        Predict direction for all assets in the universe.
        Output: per-asset direction scores and confidence.
        """
        idx = prices.index.get_loc(date) if date in prices.index else len(prices) - 1

        # Only train/retrain models that need it — limit to 15 tickers max for speed
        tickers_to_process = list(prices.columns)[:15]
        for ticker in tickers_to_process:
            series = prices[ticker].dropna()
            if len(series) < self.min_train_samples + 60:
                continue

            model_info = self._models.get(ticker)
            needs_train = (
                model_info is None
                or (idx - model_info.get("last_train", 0)) >= self.retrain_every
            )

            if needs_train:
                trained = self._train_asset_model(series, min(idx, len(series) - 1))
                if trained is not None:
                    self._models[ticker] = trained

        # Predict direction for all assets
        predictions = {}
        for ticker in prices.columns:
            series = prices[ticker]
            direction, confidence = self._predict_asset(ticker, series, idx)
            if confidence > 0.01:  # only include meaningful predictions
                predictions[ticker] = {
                    "direction": direction,
                    "confidence": confidence,
                    "score": round(direction * confidence, 4),  # composite score
                }

        # Sort by composite score
        ranked = sorted(predictions.items(), key=lambda x: x[1]["score"], reverse=True)
        bullish = [(t, p) for t, p in ranked if p["direction"] > 0]
        bearish = [(t, p) for t, p in ranked if p["direction"] < 0]

        reasoning = (
            f"ML Direction: {len(bullish)} bullish, {len(bearish)} bearish out of {len(predictions)} assets. "
            f"Top bullish: {[t for t, _ in bullish[:3]]}. "
            f"Models trained for {len(self._models)} assets."
        )

        signal = Signal(
            agent_name=self.name,
            timestamp=datetime.now(),
            signal_type="direction",
            data={
                "predictions": {t: p for t, p in predictions.items()},
                "bullish_tickers": [t for t, _ in bullish],
                "bearish_tickers": [t for t, _ in bearish],
                "n_models_trained": len(self._models),
            },
            confidence=0.6 if len(self._models) > 5 else 0.3,
            reasoning=reasoning,
        )

        self._last_signal = signal
        self._log_audit("ml_direction", {
            "date": str(date.date()),
            "n_bullish": len(bullish),
            "n_bearish": len(bearish),
        })

        return signal
