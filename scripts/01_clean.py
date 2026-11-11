"""Step 1: Datasets/ufc_fights_enriched.csv -> data/fights_clean.csv"""
import _path  # noqa: F401
from ufc import ROOT, load_config
from ufc.clean import build_clean

cfg = load_config()
clean = build_clean(cfg)
out = ROOT / cfg["paths"]["data"] / "fights_clean.csv"
clean.to_csv(out, index=False, date_format="%Y-%m-%d")
print(f"wrote {out}: {len(clean)} fights, {int(clean.is_target.sum())} decided, "
      f"{int(clean.has_market.sum())} with opening and closing odds")
