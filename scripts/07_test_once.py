"""Stage 3: the single test run.

Refits the frozen models on every fight before the test period, scores 2025-01-01 to
2026-10-10 once, and writes reports/final_test.md plus data/test_ledger.csv.
It refuses to run again once the report exists.

    python scripts/07_test_once.py --freeze     write config_frozen.json (no test data read)
    python scripts/07_test_once.py --dry-run    same code with 2023-24 standing in for the test
    python scripts/07_test_once.py              the real run
"""
import hashlib
import json
import sys

import _path  # noqa: F401
import numpy as np
import pandas as pd

from ufc import ROOT, load_config
from ufc.backtest import ledger, summarise
from ufc.market import fit_stretch, logit, sigmoid, stretch
from ufc.metrics import accuracy, brier, calibration_table, event_bootstrap, log_loss
from ufc.models import FEATURE_SETS, SymGBM, SymLogit, design, recency_weights

cfg = load_config()
data, reports = ROOT / cfg["paths"]["data"], ROOT / cfg["paths"]["reports"]
FROZEN = ROOT / "config_frozen.json"
SEEDS = [0, 1, 2, 3, 4]
REPS, SEED = cfg["bootstrap"]["reps"], cfg["seed"]


def freeze():
    best = json.load(open(data / "stage2_best_params.json", encoding="utf-8"))
    frozen = {
        "frozen_on": "2026-11-11",
        "primary_model": "B average: mean log-odds of B1 and B2",
        "models": {k: {kk: vv for kk, vv in best[k].items() if kk != "market_slope_full_fit"}
                   for k in ("A1", "A2", "B1", "B2")},
        "gbm_seeds": SEEDS,
        "calibration": "none (raw model output; rescaling did not help in 2020-24)",
        "betting_rule": {"price": "opening", "stake": "one unit", "edge_threshold": 0.05, "edge_cap": 0.20},
        "fit_window": "all decided fights before 2025-01-01 (B1 from 2015, B2 from 2013-02-16)",
        "test_window": [cfg["splits"]["test_start"], cfg["splits"]["test_end"]],
        "scoring_set": "decided fights with usable opening and closing odds (market_ok)",
        "pass_rule": {
            "1": "log loss lower than the opening odds, 95% event-bootstrap interval of the difference below zero",
            "2": "log loss lower than the stretched opening odds (slope fitted before 2025), interval below zero",
            "3": "mean closing-line value of the bets above zero, interval above zero",
            "note": "betting return is reported but is not part of the pass rule (too noisy on one test window)",
        },
        "features_sha256": hashlib.sha256((data / "features.csv").read_bytes()).hexdigest(),
    }
    FROZEN.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
    print("wrote", FROZEN)


def run(train_end, test_start, test_end, report_path, ledger_path, title):
    fz = json.load(open(FROZEN, encoding="utf-8"))
    f = pd.read_csv(data / "features.csv", parse_dates=["date"])
    pre = pd.read_csv(data / "fighter_prefight.csv", parse_dates=["date"])
    f = design(f, pre)
    fit = f[f.date < train_end]
    test = f[(f.date >= test_start) & (f.date <= test_end)].reset_index(drop=True)
    assert fit.date.max() < test.date.min()

    def train_rows(spec):
        r = fit.is_target & (fit.date >= str(spec["start"]))
        return fit[r & fit.market_ok] if spec["market"] else fit[r]

    Z = {}
    for name in ("A1", "B1"):
        s = fz["models"][name]
        tr = train_rows(s)
        m = SymLogit(FEATURE_SETS[s["feature_set"]], s["lam"], market=s["market"]).fit(
            tr, tr.y, recency_weights(tr.date, train_end, s["half_life"]))
        rows = test.has_market if s["market"] else pd.Series(True, index=test.index)
        Z[name] = pd.Series(m.decision(test[rows]), index=test.index[rows])
        if name == "B1":
            b1_slope, b1_n = m.market_slope_, len(tr)
    for name in ("A2", "B2"):
        s = fz["models"][name]
        tr = train_rows(s)
        slope = fit_stretch(tr.p_open, tr.y) if s["market"] else 1.0
        rows = test.has_market if s["market"] else pd.Series(True, index=test.index)
        zs = [SymGBM(s["params"], s["rounds"], market=s["market"], seed=sd)
              .fit(tr, tr.y, recency_weights(tr.date, train_end, s["half_life"]), market_slope=slope)
              .decision(test[rows]) for sd in fz["gbm_seeds"]]
        Z[name] = pd.Series(np.mean(zs, axis=0), index=test.index[rows])
    Z["Aens"], Z["Bens"] = (Z["A1"] + Z["A2"]) / 2, (Z["B1"] + Z["B2"]) / 2
    mk = fit[fit.is_target & fit.market_ok]
    s_stretch = fit_stretch(mk.p_open, mk.y)
    for k, z in Z.items():
        test["z_" + k] = z.reindex(test.index)
    test["z_open"], test["z_close"] = logit(test.p_open), logit(test.p_close)
    test["z_stretch"] = s_stretch * test.z_open
    test["z_elo"] = logit(test.p_elo.clip(1e-6, 1 - 1e-6))

    C = test[test.is_target & test.market_ok].reset_index(drop=True)
    names = {"elo": "Elo only", "A1": "A1 logistic, no odds", "A2": "A2 boosting, no odds", "Aens": "A average",
             "open": "Opening odds", "stretch": "Stretched opening odds", "B1": "B1 logistic + opening odds",
             "B2": "B2 boosting + opening odds", "Bens": "B average (primary)", "close": "Closing odds"}
    L = [f"# {title}\n",
         f"Models fitted on fights before {train_end}; scored on {test_start} to {test_end}. "
         f"Settings were frozen in `config_frozen.json` before this run.\n"]

    def table(df, nd=4):
        df = df.copy()
        for c in df.columns:
            if df[c].dtype.kind == "f":
                df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{nd}f}")
        L.extend(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|",
                  *["| " + " | ".join(map(str, r)) + " |" for r in df.values], ""])

    def delta(col, ref):
        y, a, b = C.y.values, sigmoid(C["z_" + col].values), sigmoid(C["z_" + ref].values)
        lo, hi = event_bootstrap(C.date.values, lambda i: log_loss(y[i], a[i]) - log_loss(y[i], b[i]), REPS, SEED)
        return log_loss(y, a) - log_loss(y, b), lo, hi

    L.append(f"## 1. Scores\n\n{len(C)} decided fights with usable opening and closing odds "
             f"({C.date.nunique()} event nights).\n")
    rows = []
    for k, nm in names.items():
        p = sigmoid(C["z_" + k].values)
        d, lo, hi = delta(k, "open")
        rows.append({"model": nm, "accuracy": accuracy(C.y, p), "log_loss": log_loss(C.y, p),
                     "brier": brier(C.y, p), "vs_opening": d, "low": lo, "high": hi})
    table(pd.DataFrame(rows))
    d_open = delta("Bens", "open")
    d_str = delta("Bens", "stretch")
    L.append(f"- Primary model against the stretched opening odds: {d_str[0]:+.4f} "
             f"(95% interval {d_str[1]:+.4f} to {d_str[2]:+.4f}). Stretch slope fitted before the test: {s_stretch:.3f}.")
    L.append(f"- B1 weight on the opening odds: {b1_slope:.3f}, fitted on {b1_n} fights.")
    allf = test[test.is_target]
    L.append(f"- No-odds A average on all {len(allf)} decided fights in the window: accuracy "
             f"{accuracy(allf.y, sigmoid(allf.z_Aens)):.4f}, log loss {log_loss(allf.y, sigmoid(allf.z_Aens)):.4f}.\n")
    L.append("### Calibration of the primary model\n")
    table(calibration_table(C.y, sigmoid(C.z_Bens)).astype({"band": str}), 3)

    rule = fz["betting_rule"]
    thr, cap = rule["edge_threshold"], rule["edge_cap"]
    M = test[test.market_ok].reset_index(drop=True)
    led = ledger(M, M.z_Bens, thr, cap)
    led.to_csv(ledger_path, index=False, date_format="%Y-%m-%d")
    s = summarise(led, REPS, SEED)
    L.append(f"## 2. Betting at opening prices (edge above {thr:.0%}, capped at {cap:.0%})\n")
    L.append(f"- Fights available: {len(M)}, of which {int((~M.is_target).sum())} were draws or no contests.")
    L.append(f"- Bets: {s['bets']}, {s['win_rate']:.1%} won, average price {s['avg_price']:.2f}.")
    L.append(f"- Return: {s['return']:+.1%} per unit staked (95% interval {s['return_low']:+.1%} to "
             f"{s['return_high']:+.1%}); {s['units_won']:+.1f} units.")
    L.append(f"- Closing-line value: {s['clv']:+.1%} (95% interval {s['clv_low']:+.1%} to {s['clv_high']:+.1%}); "
             f"{s['beat_close_share']:.0%} of bets beat the close.")
    L.append(f"- Worst peak-to-trough run: {s['max_drawdown']:.1f} units.\n")
    rows = []

    def add(label, ld):
        r = summarise(ld, REPS, SEED)
        rows.append({"check": label, "bets": r["bets"], "return": r.get("return"), "low": r.get("return_low"),
                     "high": r.get("return_high"), "clv": r.get("clv")})

    add("Primary rule", led)
    for hc in (0.02, 0.05):
        add(f"Prices {hc:.0%} worse than recorded", ledger(M, M.z_Bens, thr, cap, h=hc))
    add("Bet at the CLOSING price instead", ledger(M, M.z_Bens, thr, cap, price="close"))
    add("Without the 5% of bets with the largest line moves", led[led.move <= led.move.quantile(0.95)])
    add("No cap on the edge", ledger(M, M.z_Bens, thr))
    add("Favourites only", led[led.is_favourite])
    add("Underdogs only", led[~led.is_favourite])
    for t in (0.0, 0.02, 0.08, 0.10):
        add(f"Threshold {t:.0%} (not the frozen rule)", ledger(M, M.z_Bens, t, cap))
    add("B1 alone, same rule", ledger(M, M.z_B1, thr, cap))
    add("B2 alone, same rule", ledger(M, M.z_B2, thr, cap))
    add("Stretched opening odds, any positive edge", ledger(M, M.z_stretch, 0.0, cap))
    for yr, g in led.groupby(pd.to_datetime(led.date).dt.year):
        add(f"Year {yr}", g)
    table(pd.DataFrame(rows), 3)

    p1 = d_open[2] < 0
    p2 = d_str[2] < 0
    p3 = s["clv_low"] > 0
    L.append("## 3. Pass rule\n")
    L.append(f"1. Lower log loss than the opening odds, interval below zero: **{'PASS' if p1 else 'FAIL'}** "
             f"({d_open[0]:+.4f}, {d_open[1]:+.4f} to {d_open[2]:+.4f})")
    L.append(f"2. Lower log loss than the stretched opening odds, interval below zero: **{'PASS' if p2 else 'FAIL'}** "
             f"({d_str[0]:+.4f}, {d_str[1]:+.4f} to {d_str[2]:+.4f})")
    L.append(f"3. Closing-line value above zero, interval above zero: **{'PASS' if p3 else 'FAIL'}** "
             f"({s['clv']:+.4f}, {s['clv_low']:+.4f} to {s['clv_high']:+.4f})")
    L.append(f"\n**Overall: {'PASS' if (p1 and p2 and p3) else 'NOT PASSED'}** ({sum([p1, p2, p3])} of 3).\n")
    report_path.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    if "--freeze" in sys.argv:
        freeze()
    elif "--dry-run" in sys.argv:
        run("2023-01-01", "2023-01-01", "2024-12-31", reports / "dryrun_2023_24.md",
            data / "dryrun_ledger.csv", "Dry run: 2023-24 standing in for the test (not the real test)")
    else:
        out = reports / "final_test.md"
        if out.exists():
            sys.exit(f"{out} already exists. The test is run once; refusing to run again.")
        fz = json.load(open(FROZEN, encoding="utf-8"))
        sha = hashlib.sha256((data / "features.csv").read_bytes()).hexdigest()
        assert sha == fz["features_sha256"], "features.csv changed after the settings were frozen"
        s = cfg["splits"]
        run(s["test_start"], s["test_start"], s["test_end"], out, data / "test_ledger.csv",
            "Final test: 2025-01-01 to 2026-10-10")
