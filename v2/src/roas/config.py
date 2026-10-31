"""Central configuration.

Everything that v1 hard-coded inside functions (channel list, spend tiers, quantiles,
recency window, special-cased games) lives here so it can be reviewed and overridden
in one place (review finding Q5).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# Canonical marketing-channel columns (as they appear in the source data).
CANONICAL_CHANNELS: list[str] = [
    "MC_Google",
    "MC_Facebook",
    "MC_Applovin",
    "MC_Moloco",
    "MC_Unity",
    "MC_Ironsource",
    "MC_Tiktok",
    "MC_Organic",
    "MC_Other",
]

# Explicit aliases (finding C5). v1 used "MC_TicToc" while the data used "MC_Tiktok",
# so TikTok spend silently fell into "others". Every accepted spelling is listed here;
# anything else is rejected at the API boundary or summed into MC_Other during training
# (and logged), never silently dropped.
CHANNEL_ALIASES: dict[str, str] = {
    "mc_tiktok": "MC_Tiktok",
    "mc_tictoc": "MC_Tiktok",
    "mc_tik_tok": "MC_Tiktok",
    "mc_google": "MC_Google",
    "mc_adwords": "MC_Google",
    "mc_facebook": "MC_Facebook",
    "mc_meta": "MC_Facebook",
    "mc_applovin": "MC_Applovin",
    "mc_moloco": "MC_Moloco",
    "mc_unity": "MC_Unity",
    "mc_ironsource": "MC_Ironsource",
    "mc_organic": "MC_Organic",
    "mc_other": "MC_Other",
    "mc_others": "MC_Other",
}


@dataclass(frozen=True)
class Settings:
    # Quantiles produced for every game: bad / expected / good.
    quantiles: tuple[float, ...] = (0.25, 0.50, 0.75)
    # Cohorts acquired before this date are excluded (v1 used string matching on years).
    recency_start: str = "2023-01-01"
    # Number of ROAS lags used as features.
    n_lags: int = 3
    # Maximum forecast horizon served (months since acquisition).
    max_horizon: int = 24
    # Spend tiers for the out-of-range scaling multiplier gamma: (upper bound, gamma).
    # These are v1 business assumptions, kept but now explicit and documented.
    gamma_tiers: tuple[tuple[float, float], ...] = ((300_000.0, 1.0), (1_000_000.0, 1.4), (float("inf"), 1.7))
    # Fallback diminishing-returns coefficient when a category has no estimate.
    default_alpha: float = 0.05
    # Shrinkage strength (pseudo-observations) pulling a game's alpha toward its category.
    alpha_shrinkage_k: float = 4.0
    # Model backend: "xgboost" (production) or "sklearn" (fallback, no extra dependency).
    # Override with the ROAS_BACKEND environment variable.
    backend: str = os.environ.get("ROAS_BACKEND", "auto")
    random_state: int = 42
    # Hyper-parameters (kept small and explicit; tuned with temporal CV, see splits.py).
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 4
    min_child_weight: float = 3.0

    def gamma(self, total_cac: float) -> float:
        for upper, g in self.gamma_tiers:
            if total_cac < upper:
                return g
        return self.gamma_tiers[-1][1]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gamma_tiers"] = [[u if u != float("inf") else "inf", g] for u, g in self.gamma_tiers]
        return d

    @classmethod
    def from_json(cls, path: str | Path) -> Settings:
        raw = json.loads(Path(path).read_text())
        if "gamma_tiers" in raw:
            raw["gamma_tiers"] = tuple((float(u), float(g)) for u, g in raw["gamma_tiers"])
        if "quantiles" in raw:
            raw["quantiles"] = tuple(raw["quantiles"])
        return cls(**raw)


DEFAULT_SETTINGS = Settings()
