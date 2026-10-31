"""C5 - channel names are canonicalised; nothing is silently dropped."""
import pandas as pd
import pytest
from pydantic import ValidationError

from roas.config import CANONICAL_CHANNELS
from roas.data import canonical_channel, canonicalise_channels
from services.api.schemas import ScenarioRequest


def test_tiktok_aliases():
    for name in ["MC_Tiktok", "MC_TicToc", "mc_tiktok", "MC_Tik-Tok"]:
        assert canonical_channel(name) == "MC_Tiktok"


def test_unknown_columns_go_to_other_and_totals_preserved():
    df = pd.DataFrame({"MC_Tiktok": [10.0], "MC_adwords": [5.0], "MC_Rewarded Ads": [7.0]})
    out, unmapped = canonicalise_channels(df)
    assert unmapped == ["MC_Rewarded Ads"]
    assert out["MC_Tiktok"].iloc[0] == 10 and out["MC_Google"].iloc[0] == 5 and out["MC_Other"].iloc[0] == 7
    assert out[CANONICAL_CHANNELS].sum(axis=1).iloc[0] == 22


def test_api_rejects_unknown_channel():
    with pytest.raises(ValidationError, match="unknown channel"):
        ScenarioRequest(game_id="SG01", cohort_month="2025-07", channel_spends={"MC_Snapchat": 1000})


def test_api_maps_alias_and_merges():
    r = ScenarioRequest(game_id="SG01", cohort_month="2025-07", channel_spends={"MC_TicToc": 10, "MC_Tiktok": 5})
    assert r.channel_spends == {"MC_Tiktok": 15}


def test_api_rejects_negative_spend_and_bad_month():
    with pytest.raises(ValidationError):
        ScenarioRequest(game_id="SG01", cohort_month="2025-07", channel_spends={"MC_Google": -1})
    with pytest.raises(ValidationError):
        ScenarioRequest(game_id="SG01", cohort_month="2025-13", total_cac=1)
