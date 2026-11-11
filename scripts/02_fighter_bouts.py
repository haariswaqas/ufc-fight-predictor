"""Step 2: data/fights_clean.csv -> data/fighter_bouts.csv (two rows per fight)"""
import _path  # noqa: F401
from ufc import ROOT, load_config
from ufc.clean import load_clean
from ufc.history import fighter_bouts

cfg = load_config()
bouts = fighter_bouts(load_clean(cfg))
out = ROOT / cfg["paths"]["data"] / "fighter_bouts.csv"
bouts.to_csv(out, index=False, date_format="%Y-%m-%d")
print(f"wrote {out}: {len(bouts)} rows, {bouts.fighter.nunique()} fighters")
