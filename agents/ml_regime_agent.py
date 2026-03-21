"""
ML Regime Agent — XGBoost 5-Regime Classifier
Inspired by: Regime-Aware Options Engine (Gowrisankar, Tarun, Vedant)

5 regimes: PANIC / HIGH_FEAR / ELEVATED / NORMAL / COMPLACENT
Features: VIX, realized vol, momentum, vol ratio, RSI proxy, price vs MA50
LAG-1 protocol: all features use Day T data to predict Day T+1 regime.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import IntEnum

from agents.base_agent import BaseAgent, Signal

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier


class Regime5(IntEnum):
    PANIC = 0        # VIX > 30, extreme vol
    HIGH_FEAR = 1    # VIX 25-30, defensive
    ELEVATED = 2     # VIX 20-25, cautious
    NORMAL = 3       # VIX 15-20, balanced
    COMPLACENT = 4   # VIX <= 15, low vol


# Regime metadata for downstream strategy selection
REGIME_META = {
    Regime5.PANIC:      {"label": "PANIC",      "action": "NO_TRADE",   "var_mult": 0.0},
    Regime5.HIGH_FEAR:  {"label": "HIGH_FEAR",  "action": "NO_TRADE",   "var_mult": 0.0},
    Regime5.ELEVATED:   {"label": "ELEVATED",   "action": "DEFENSIVE",  "var_mult": 0.8},
    Regime5.NORMAL:     {"label": "NORMAL",     "action": "MODERATE",   "var_mult": 2.5},
    Regime5.COMPLACENT: {"label": "COMPLACENT", "action": "AGGRESSIVE", "var_mult": 3.5},
}


class MLRegimeAgent(BaseAgent):
    """
    XGBoost-based 5-regime classifier.
    Trains on historical VIX-labeled data, then predicts regimes with LAG-1.
    Falls back to rule-based if insufficient training data.
    """

    def __init__(self, retrain_every: int = 63):
        super().__init__("MLRegime")
        self.retrain_every = retrain_every
        self._model = None
        self._last_train_idx: int = 0
        self._feature_names = []
        self._is_trained = False

    def _label_regime_from_vix(self, vix: float) -> int:
        """Rule-based regime label from VIX (used for training labels)."""
        if vix > 30:
            return Regime5.PANIC
        elif vix > 25:
            return Regime5.HIGH_FEAR
        elif vix > 20:
            return Regime5.ELEVATED
        elif vix > 15:
            return Regime5.NORMAL
        else:
            return Regime5.COMPLACENT

    def _build_features(self, prices: pd.DataFrame, vix_series: pd.Series,
                         idx: int) -> Optional[Dict[str, float]]:
        """
        Build feature vector at a given index.
        All features use data up to idx (LAG-1 compliant).
        """
        if idx < 252:
            return None

        spy_col = None
        for col in ["SPY", "^GSPC"]:
            if col in prices.columns:
                spy_col = col
                break
        if spy_col is None:
            spy_col = prices.columns[0]

        spy = prices[spy_col].iloc[:idx + 1]

        # VIX features
        vix_val = vix_series.iloc[idx] if idx < len(vix_series) and pd.notna(vix_series.iloc[idx]) else 20.0
        vix_recent = vix_series.iloc[max(0, idx - 20):idx + 1].dropna()
        vix_ma20 = vix_recent.mean() if len(vix_recent) > 0 else vix_val
        vix_change = (vix_val / vix_series.iloc[idx - 1] - 1) if idx > 0 and pd.notna(vix_series.iloc[idx - 1]) and vix_series.iloc[idx - 1] > 0 else 0.0
        vix_spike = 1.0 if len(vix_recent) >= 20 and vix_val > 1.3 * vix_ma20 else 0.0

        # Returns and momentum
        ret_1d = spy.iloc[-1] / spy.iloc[-2] - 1 if len(spy) >= 2 else 0.0
        ret_5d = spy.iloc[-1] / spy.iloc[-6] - 1 if len(spy) >= 6 else 0.0
        ret_20d = spy.iloc[-1] / spy.iloc[-21] - 1 if len(spy) >= 21 else 0.0

        # Momentum acceleration
        mom_5d_now = ret_5d
        mom_5d_prev = spy.iloc[-6] / spy.iloc[-11] - 1 if len(spy) >= 11 else 0.0
        momentum_accel = mom_5d_now - mom_5d_prev

        # Realized vol (20-day)
        daily_ret = spy.pct_change().dropna()
        realized_vol = daily_ret.iloc[-20:].std() * np.sqrt(252) if len(daily_ret) >= 20 else 0.15
        vol_60d = daily_ret.iloc[-60:].std() * np.sqrt(252) if len(daily_ret) >= 60 else realized_vol
        vol_ratio = realized_vol / max(vol_60d, 0.001)
        vol_trend = (daily_ret.iloc[-5:].std() - daily_ret.iloc[-25:-20].std()) if len(daily_ret) >= 25 else 0.0

        # Price vs MA50
        ma50 = spy.iloc[-50:].mean() if len(spy) >= 50 else spy.mean()
        price_vs_ma50 = (spy.iloc[-1] - ma50) / ma50

        # RSI proxy (% of up days in last 14)
        recent_rets = daily_ret.iloc[-14:]
        rsi_proxy = (recent_rets > 0).sum() / max(len(recent_rets), 1)

        # Volume ratio (if available)
        volume_ratio = 1.0  # default

        # Breadth: % of universe above 200d SMA
        breadth = 0.5
        count = 0
        above = 0
        for col in prices.columns[:15]:
            s = prices[col].iloc[:idx + 1]
            if len(s) >= 200:
                count += 1
                if s.iloc[-1] > s.iloc[-200:].mean():
                    above += 1
        if count > 0:
            breadth = above / count

        return {
            "vix": vix_val,
            "vix_change": vix_change,
            "vix_spike": vix_spike,
            "returns_1d": ret_1d,
            "momentum_5d": ret_5d,
            "momentum_20d": ret_20d,
            "momentum_accel": momentum_accel,
            "realized_vol": realized_vol,
            "vol_ratio": vol_ratio,
            "vol_trend": vol_trend,
            "price_vs_ma50": price_vs_ma50,
            "rsi_proxy": rsi_proxy,
            "breadth": breadth,
        }

    def _train(self, prices: pd.DataFrame, vix_series: pd.Series, up_to_idx: int):
        """Train the XGBoost classifier on historical data up to idx."""
        X_rows = []
        y_rows = []

        start_idx = 252
        # Sample every 3rd day for speed (still gives 300+ samples for 3yr window)
        step = 3
        for i in range(start_idx, up_to_idx, step):
            features = self._build_features(prices, vix_series, i)
            if features is None:
                continue

            # Label from VIX on day i+1 (what regime will tomorrow be?)
            if i + 1 < len(vix_series) and pd.notna(vix_series.iloc[i + 1]):
                label = self._label_regime_from_vix(vix_series.iloc[i + 1])
            else:
                continue

            X_rows.append(features)
            y_rows.append(label)

        if len(X_rows) < 100:
            self._is_trained = False
            return

        X = pd.DataFrame(X_rows)
        y = np.array(y_rows)
        self._feature_names = list(X.columns)

        # XGBoost with conservative params (from Regime-Aware Options Engine)
        if XGBOOST_AVAILABLE:
            self._model = XGBClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.15,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="mlogloss",
                verbosity=0,
            )
        else:
            self._model = GradientBoostingClassifier(
                n_estimators=80,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
            )

        self._model.fit(X, y)
        self._is_trained = True
        self._last_train_idx = up_to_idx

        self.logger.info(f"ML Regime model trained on {len(X)} samples, "
                          f"classes: {np.unique(y, return_counts=True)}")

    def _predict(self, prices: pd.DataFrame, vix_series: pd.Series,
                  idx: int) -> tuple:
        """Predict regime at idx. Returns (regime, confidence, probabilities)."""
        features = self._build_features(prices, vix_series, idx)
        if features is None or not self._is_trained:
            # Fallback to rule-based
            vix = vix_series.iloc[idx] if idx < len(vix_series) and pd.notna(vix_series.iloc[idx]) else 20.0
            regime = self._label_regime_from_vix(vix)
            return regime, 0.5, {}

        X = pd.DataFrame([features])[self._feature_names]
        proba = self._model.predict_proba(X)[0]
        pred = int(self._model.predict(X)[0])

        # Map probabilities to regime names
        classes = self._model.classes_
        proba_dict = {REGIME_META[int(c)]["label"]: round(float(p), 4)
                       for c, p in zip(classes, proba)}

        confidence = float(max(proba))

        return pred, confidence, proba_dict

    def analyze(self, date: pd.Timestamp, prices: pd.DataFrame,
                returns: pd.DataFrame, context: Dict[str, Any]) -> Signal:
        """
        Predict the current market regime using XGBoost.
        """
        vix_series = context.get("vix_series", pd.Series(dtype=float))
        idx = prices.index.get_loc(date) if date in prices.index else len(prices) - 1

        # Retrain periodically
        if not self._is_trained or (idx - self._last_train_idx) >= self.retrain_every:
            if len(vix_series) > 0:
                self._train(prices, vix_series, idx)

        # Predict
        if len(vix_series) > 0:
            regime_int, confidence, proba_dict = self._predict(prices, vix_series, idx)
        else:
            # No VIX data — fallback to rule-based from price action
            regime_int = Regime5.NORMAL
            confidence = 0.4
            proba_dict = {}

        regime_meta = REGIME_META[regime_int]

        # Bull ratio (from LSTM project) — 10-day rolling fraction of bullish predictions
        # We compute it from price: what fraction of last 10 days had positive returns
        if "SPY" in prices.columns:
            spy_ret = prices["SPY"].pct_change()
            recent = spy_ret.iloc[max(0, idx - 10):idx + 1]
            bull_ratio = float((recent > 0).sum() / max(len(recent), 1))
        else:
            bull_ratio = 0.5

        reasoning = (
            f"ML Regime: {regime_meta['label']} (confidence {confidence:.0%}). "
            f"Action: {regime_meta['action']}. "
            f"VaR multiplier: {regime_meta['var_mult']}. "
            f"Bull ratio: {bull_ratio:.2f}. "
            f"{'Model trained.' if self._is_trained else 'Fallback to rules.'}"
        )

        signal = Signal(
            agent_name=self.name,
            timestamp=datetime.now(),
            signal_type="regime",
            data={
                "regime": regime_meta["label"],
                "regime_int": regime_int,
                "action": regime_meta["action"],
                "var_multiplier": regime_meta["var_mult"],
                "confidence": round(confidence, 4),
                "probabilities": proba_dict,
                "bull_ratio": round(bull_ratio, 4),
                "ml_trained": self._is_trained,
                # Map to old 3-regime system for compatibility
                "regime_3way": "CRISIS" if regime_int <= 1 else ("RISK_OFF" if regime_int == 2 else "RISK_ON"),
            },
            confidence=round(confidence, 4),
            reasoning=reasoning,
        )

        self._last_signal = signal
        self._log_audit("ml_regime", {
            "date": str(date.date()),
            "regime": regime_meta["label"],
            "confidence": round(confidence, 4),
        })

        return signal
