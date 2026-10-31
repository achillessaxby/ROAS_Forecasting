"""Smoke test for the running stack (no dependencies beyond the standard library).

    python scripts/smoke_test.py                      # uses http://localhost:8080 and key from .env
    python scripts/smoke_test.py --url http://localhost:8080 --key local-demo-key
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def key_from_env_file() -> str:
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("ROAS_API_KEYS="):
                return line.split("=", 1)[1].split(",")[0].strip()
    return "local-demo-key"


def call(url, body=None, key=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", **({"X-API-Key": key} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--key", default=None)
    a = p.parse_args()
    key = a.key or key_from_env_file()
    body = {"scenarios": [{"game_id": "SG01", "cohort_month": "2025-10", "forecast_months": 12,
                           "channel_spends": {"MC_Google": 90000, "MC_TicToc": 40000, "MC_Organic": 20000}}]}
    checks = []
    s, h = call(f"{a.url}/health")
    checks.append(("health is open and reports a model version", s == 200 and h.get("model_version", "").startswith("v")))
    s, _ = call(f"{a.url}/v2/forecast", body)
    checks.append(("forecast without API key is rejected (401)", s == 401))
    s, r = call(f"{a.url}/v2/forecast", body, key)
    ok = s == 200 and len(r["results"][0]["months"]) == 12
    checks.append(("forecast with API key returns 12 months with bands", ok))
    bad = {"scenarios": [{"game_id": "SG01", "cohort_month": "2025-10", "channel_spends": {"MC_Snapchat": 1}}]}
    s, _ = call(f"{a.url}/v2/forecast", bad, key)
    checks.append(("unknown channel is rejected (422)", s == 422))
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    if ok:
        m = r["results"][0]["months"][-1]
        print(f"\nmodel {r['model_version']} (data as of {r['data_as_of']}): SG01, Oct-25 cohort, $150k -> 12-month "
              f"cumulative ROAS {m['cumulative_bad']:.3f} / {m['cumulative_expected']:.3f} / {m['cumulative_good']:.3f}")
    return 0 if all(p for _, p in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
