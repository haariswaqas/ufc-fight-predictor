"""Step 3: pre-fight snapshots and the matchup feature table."""
import json

import _path  # noqa: F401
from ufc import ROOT, load_config
from ufc.clean import load_clean
from ufc.features import FEATURES, MARKET, build_all

cfg = load_config()
data = ROOT / cfg["paths"]["data"]
_, pre, feats = build_all(load_clean(cfg), cfg)
pre.to_csv(data / "fighter_prefight.csv", index=False, date_format="%Y-%m-%d")
feats.to_csv(data / "features.csv", index=False, date_format="%Y-%m-%d")
with open(data / "feature_manifest.json", "w", encoding="utf-8") as f:
    json.dump({"fight_features": FEATURES, "market_columns": MARKET}, f, indent=1)
print(f"wrote features.csv: {len(feats)} fights, {len(FEATURES)} fight features")
print(feats.groupby("split").agg(fights=("y", "size"), decided=("is_target", "sum"),
                                 with_market=("has_market", "sum")).to_string())
