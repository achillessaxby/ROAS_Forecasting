"""Quantile models (M1), alpha estimation (M5), registry versioning (P3)."""
import numpy as np
import pandas as pd
import pytest

from roas.models import QuantileModelSet, pinball_loss
from roas.registry import Registry
from roas.scaling import estimate_alphas, out_of_range_factor


def test_pinball_loss_values():
    assert pinball_loss([1.0], [0.0], 0.25) == pytest.approx(0.25)
    assert pinball_loss([0.0], [1.0], 0.25) == pytest.approx(0.75)


def test_models_never_cross_and_roundtrip(registry_root, tmp_path):
    lv = Registry(registry_root).load()
    m = lv.models["SG01"]
    X = pd.DataFrame([{f: 0.0 for f in m.features} | {"log_total_cac": np.log1p(1e5), "elapsed_months": k, "cohort_month": 3,
                       "current_month": 3} for k in range(1, 13)])
    p = m.predict(X)
    assert (np.diff(p, axis=1) >= 0).all()
    m.save(tmp_path / "copy")
    np.testing.assert_allclose(QuantileModelSet.load(tmp_path / "copy").predict(X), p)


def test_out_of_range_factor():
    assert out_of_range_factor(100, 50, 200, alpha=0.3) == 1.0
    assert out_of_range_factor(400, 50, 200, alpha=0.3) < 1.0
    lo = out_of_range_factor(10, 50, 200, alpha=0.3)
    assert 1.0 < lo <= 2.0


def test_alpha_estimates_ranked_and_shrunk(games, data_dir):
    meta = pd.read_csv(data_dir / "game_metrics.csv")
    est = estimate_alphas(games, dict(zip(meta.game_id, meta.Category, strict=False)))
    truth = dict(zip(meta.game_id, meta.true_alpha, strict=False))
    # the steepest true decay (SG03) is estimated above the flattest (SG06)
    assert est["SG03"].alpha > est["SG06"].alpha
    assert all(0 <= e.alpha <= 2 for e in est.values())
    assert np.corrcoef([truth[g] for g in est], [est[g].game_alpha for g in est])[0, 1] > 0.5


def test_registry_manifest_and_promotion(registry_root):
    reg = Registry(registry_root)
    lv = reg.load()
    assert lv.version == reg.current() and lv.data_as_of == "2025-09-01"
    assert len(lv.manifest["data_hash"]) == 16
    with pytest.raises(FileNotFoundError):
        reg.promote("v1999-01")


def test_promotion_gate_blocks_bad_versions(registry_root, tmp_path):
    import json
    import shutil

    from roas.pipeline import promote_if_passing

    reg = Registry(registry_root)
    good_version = reg.current()
    trial = "v2099-01"
    shutil.copytree(reg.version_dir(good_version), reg.version_dir(trial))
    summary = tmp_path / "s.csv"
    row = {"level": "overall", "key": "all", "mae_model": 2.0, "mae_baseline": 1.0, "skill_vs_baseline": -1.0,
           "band_coverage": 0.5, "pinball": 1.0}
    pd.DataFrame([row]).to_csv(summary, index=False)
    assert promote_if_passing(registry_root, trial, summary) is False
    assert reg.current() == good_version  # CURRENT untouched
    assert json.loads((reg.version_dir(trial) / "manifest.json").read_text())["backtest"]["passed"] is False
    row.update(mae_model=0.5, skill_vs_baseline=0.5)
    pd.DataFrame([row]).to_csv(summary, index=False)
    assert promote_if_passing(registry_root, trial, summary) is True
    assert reg.current() == trial
    reg.promote(good_version)
