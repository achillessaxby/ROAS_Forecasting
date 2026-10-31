"""Per-game quantile models (bad / expected / good).

Review findings addressed here
------------------------------
M1  v1 used a hand-written pinball gradient with a constant Hessian and tuned the quantile
    models on *MSE*, which pulls the 0.25 / 0.75 models toward the mean; nothing prevented
    quantile crossing. Here: native quantile objectives (XGBoost ``reg:quantileerror`` or
    scikit-learn ``loss="quantile"``), tuning on pinball loss with time-based folds, and
    outputs sorted per row so bad <= expected <= good always holds.
Q7  Models are saved in a portable format (XGBoost JSON) with a metadata file recording the
    library versions, features and training window - no cloudpickle of notebook functions.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import logging
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_SETTINGS, Settings
from .splits import expanding_window_folds

log = logging.getLogger(__name__)


def resolve_backend(name: str = "auto") -> str:
    if name == "auto":
        return "xgboost" if importlib.util.find_spec("xgboost") else "sklearn"
    if name not in {"xgboost", "sklearn"}:
        raise ValueError(f"unknown backend {name!r}")
    return name


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = np.asarray(y, float) - np.asarray(pred, float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def _make_estimator(backend: str, q: float, params: dict, seed: int):
    if backend == "xgboost":
        import xgboost as xgb

        return xgb.XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=q,
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            tree_method="hist",
            random_state=seed,
        )
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        max_iter=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_samples_leaf=max(5, int(params["min_child_weight"] * 3)),
        random_state=seed,
    )


@dataclass
class QuantileModelSet:
    """Three (or more) quantile regressors sharing one feature list."""

    features: list[str]
    quantiles: Sequence[float] = DEFAULT_SETTINGS.quantiles
    backend: str = "auto"
    params: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    models: dict[float, object] = field(default_factory=dict)

    def __post_init__(self):
        self.backend = resolve_backend(self.backend)

    # ----------------------------------------------------------------- training
    def fit(self, X: pd.DataFrame, y: pd.Series, settings: Settings = DEFAULT_SETTINGS) -> QuantileModelSet:
        params = {
            "n_estimators": settings.n_estimators,
            "learning_rate": settings.learning_rate,
            "max_depth": settings.max_depth,
            "min_child_weight": settings.min_child_weight,
            **self.params,
        }
        self.params = params
        Xf = X[self.features].astype(float)
        for q in self.quantiles:
            est = _make_estimator(self.backend, q, params, settings.random_state)
            est.fit(Xf, y.astype(float))
            self.models[q] = est
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return shape (n_rows, n_quantiles), sorted so quantiles never cross."""
        Xf = X[self.features].astype(float)
        raw = np.column_stack([self.models[q].predict(Xf) for q in self.quantiles])
        return np.sort(raw, axis=1)

    # ----------------------------------------------------------------- tuning
    @classmethod
    def tune(
        cls,
        frame: pd.DataFrame,
        features: list[str],
        target: str,
        settings: Settings = DEFAULT_SETTINGS,
        grid: dict[str, Sequence] | None = None,
    ) -> dict[str, float]:
        """Pick hyper-parameters by mean pinball loss on expanding-window folds (no leakage)."""
        grid = grid or {"max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
        folds = expanding_window_folds(frame)
        if not folds:
            return {}
        best, best_score = {}, np.inf
        keys = list(grid)
        for combo in itertools.product(*[grid[k] for k in keys]):
            params = dict(zip(keys, combo, strict=False))
            scores = []
            for tr, va in folds:
                m = cls(features=features, quantiles=settings.quantiles, backend=settings.backend, params=params)
                m.fit(frame.iloc[tr], frame.iloc[tr][target], settings)
                pred = m.predict(frame.iloc[va])
                y = frame.iloc[va][target].to_numpy()
                scores.append(np.mean([pinball_loss(y, pred[:, i], q) for i, q in enumerate(settings.quantiles)]))
            score = float(np.mean(scores))
            if score < best_score:
                best, best_score = params, score
        log.info("tuned params %s (pinball %.4f over %d folds)", best, best_score, len(folds))
        return best

    # ----------------------------------------------------------------- persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        files = {}
        for q, est in self.models.items():
            if self.backend == "xgboost":
                fn = f"q{q:.2f}.json"
                est.save_model(d / fn)
            else:
                import joblib

                fn = f"q{q:.2f}.joblib"
                joblib.dump(est, d / fn)
            files[str(q)] = fn
        meta = {
            **self.metadata,
            "backend": self.backend,
            "quantiles": list(self.quantiles),
            "features": self.features,
            "params": self.params,
            "files": files,
            "versions": _library_versions(self.backend),
        }
        (d / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    @classmethod
    def load(cls, directory: str | Path) -> QuantileModelSet:
        d = Path(directory)
        meta = json.loads((d / "metadata.json").read_text())
        obj = cls(features=meta["features"], quantiles=meta["quantiles"], backend=meta["backend"],
                  params=meta.get("params", {}), metadata=meta)
        for q_str, fn in meta["files"].items():
            q = float(q_str)
            if obj.backend == "xgboost":
                import xgboost as xgb

                est = xgb.XGBRegressor()
                est.load_model(d / fn)
            else:
                import joblib

                est = joblib.load(d / fn)
            obj.models[q] = est
        obj.quantiles = [float(q) for q in meta["quantiles"]]
        return obj


def _library_versions(backend: str) -> dict[str, str]:
    import sklearn

    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
         "scikit-learn": sklearn.__version__}
    if backend == "xgboost":
        import xgboost

        v["xgboost"] = xgboost.__version__
    return v
