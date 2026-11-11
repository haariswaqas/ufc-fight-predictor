"""Fighter histories. This is the only place fighter state is updated.

The leak-safe rule: the features for a fight on date D are read from fighter state
BEFORE any fight on date D has been applied. Dates are processed as batches:
read state for every fight that day, then apply all of that day's results.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .clean import ROUND_COLS, STAT_COLS

# sums kept per fighter, career and exponentially weighted ("recent")
SUMS = ["secs", "sig_l", "sig_a", "osig_l", "osig_a", "kd", "okd", "td_l", "td_a", "otd_l", "otd_a",
        "sub", "csecs", "ctrl", "octrl"]


def fighter_bouts(clean):
    """Two rows per fight: one from each fighter's side."""
    rows = []
    for me, op in (("1", "2"), ("2", "1")):
        r = pd.DataFrame({
            "fight_id": clean.fight_id, "date": clean.date, "side": int(me),
            "fighter": clean[f"f{me}_key"], "opp": clean[f"f{op}_key"],
            "name": clean[f"fighter_{me}"],
            "result": clean.result, "method_group": clean.method_group,
            "score": np.where(clean.is_target, (clean.f1_won == (me == "1")).astype(float),
                              np.where(clean.result.eq("draw"), 0.5, np.nan)),
            "fight_secs": clean.fight_secs, "end_round": clean["round"],
            "sched_rounds": clean.sched_rounds, "r1_secs": clean.r1_secs, "late_secs": clean.late_secs,
            "height_cm": clean[f"f{me}_height_cm"], "reach_cm": clean[f"f{me}_reach_cm"],
            "stance": clean[f"f{me}_stance"], "dob": clean[f"f{me}_dob"],
        })
        for c in STAT_COLS:
            r[c] = clean[f"f{me}_{c}"]
            r["opp_" + c] = clean[f"f{op}_{c}"]
        for c in ROUND_COLS:
            r[c] = clean[f"f{me}_{c}"]
            r["opp_" + c] = clean[f"f{op}_{c}"]
        rows.append(r)
    b = pd.concat(rows, ignore_index=True)
    b["dob"] = pd.to_datetime(b.dob)
    return b.sort_values(["date", "fight_id", "side"], kind="stable").reset_index(drop=True)


def _new_state(cfg):
    s = {"n": 0, "decided": 0, "wins": 0, "losses": 0, "streak": 0, "fin_w": 0, "ko_w": 0,
         "sub_w": 0, "dec_w": 0, "ko_l": 0, "sub_l": 0, "last": None, "dates": [],
         "elo": cfg["elo"]["start"], "opp_elo_sum": 0.0, "ew_score": 0.0, "ew_n": 0.0,
         "total_secs": 0.0}
    for k in SUMS + PACE_SUMS:
        s["c_" + k] = 0.0
        s["r_" + k] = 0.0
    return s


def _bout_sums(r):
    """Per-fight contributions. Each stat only counts when it was recorded."""
    out = dict.fromkeys(SUMS, 0.0)
    if r["sig_str_att"] == r["sig_str_att"] and r["opp_sig_str_att"] == r["opp_sig_str_att"]:
        out.update(secs=r["fight_secs"], sig_l=r["sig_str_landed"], sig_a=r["sig_str_att"],
                   osig_l=r["opp_sig_str_landed"], osig_a=r["opp_sig_str_att"], kd=r["kd"],
                   okd=r["opp_kd"], td_l=r["td_landed"], td_a=r["td_att"],
                   otd_l=r["opp_td_landed"], otd_a=r["opp_td_att"], sub=r["sub_att"])
    if r["ctrl_seconds"] == r["ctrl_seconds"] and r["opp_ctrl_seconds"] == r["opp_ctrl_seconds"]:
        out.update(csecs=r["fight_secs"], ctrl=r["ctrl_seconds"], octrl=r["opp_ctrl_seconds"])
    return out


# ---------------------------------------------------------------- pace and timing
# Extra sums for the method and round models. They live beside the original sums and
# never feed the winner model's features.
PACE_SUMS = ["tsecs", "n5", "reach3", "reach3_dec", "reach3_w", "reach4", "late_total_secs",
             "early_fin_w", "r1_fin_w", "early_fin_l", "went_dec", "fin_round_w",
             "r1_secs", "r1_att", "r1_abs", "late_secs", "late_att", "late_abs"]
PACE = ["att_pm", "r_att_pm", "oatt_pm", "r1_att_pm", "r_r1_att_pm", "r1_abs_pm", "late_att_pm", "fade",
        "avg_fight_min", "dec_rate", "early_fin_w_rate", "r1_fin_w_rate", "early_fin_l_rate",
        "ko_w_p15", "sub_w_p15", "ko_l_p15", "sub_l_p15", "avg_fin_round_w", "five_round_fights",
        "fights_reaching_r4", "late_minutes", "r3_win_rate", "fights_reaching_r3"]


def _pace_sums(r):
    """Per-fight contributions to the pace and timing sums."""
    out = dict.fromkeys(PACE_SUMS, 0.0)
    secs, rnd, sc, mg = r["fight_secs"], r["end_round"], r["score"], r["method_group"]
    if secs != secs or rnd != rnd:          # an upcoming fight: nothing to add
        return out
    won, lost = sc == 1.0, sc == 0.0
    fin = mg in ("KO", "SUB")
    out.update(tsecs=secs, n5=float(r["sched_rounds"] == 5), reach3=float(rnd >= 3),
               reach3_dec=float(rnd >= 3 and (won or lost)), reach3_w=float(rnd >= 3 and won),
               reach4=float(rnd >= 4), late_total_secs=max(secs - 900.0, 0.0),
               early_fin_w=float(won and fin and rnd <= 2), r1_fin_w=float(won and fin and rnd == 1),
               early_fin_l=float(lost and fin and rnd <= 2), went_dec=float((won or lost) and mg == "DEC"),
               fin_round_w=float(rnd) if (won and fin) else 0.0)
    if r["r1_att"] == r["r1_att"] and r["opp_r1_landed"] == r["opp_r1_landed"]:
        out.update(r1_secs=r["r1_secs"], r1_att=r["r1_att"], r1_abs=r["opp_r1_landed"],
                   late_secs=r["late_secs"], late_att=r["late_att"], late_abs=r["opp_late_landed"])
    return out


def _pace_priors(P):
    def div(a, b):
        return a / b if b > 0 else np.nan
    out = {"sig_a": div(P["sig_a"], P["secs"]), "osig_a": div(P["osig_a"], P["secs"]),
           "r1_att": div(P["r1_att"], P["r1_secs"]), "r1_abs": div(P["r1_abs"], P["r1_secs"]),
           "late_att": div(P["late_att"], P["late_secs"]), "r3_win": div(P["reach3_w"], P["reach3_dec"]),
           "fin_round_w": div(P["fin_round_w"], P["fin_w"])}
    for k in ("tsecs", "went_dec", "early_fin_w", "r1_fin_w", "early_fin_l"):
        out[k] = div(P[k], P["n"])
    for k in ("ko_w", "sub_w", "ko_l", "sub_l"):
        out[k + "_ps"] = div(P[k], P["tsecs"])
    return out


def _pace_snapshot(s, pp, cfg):
    """Pre-fight pace and timing values for one fighter."""
    h = cfg["history"]
    ks, kf, kr = h["k_seconds"], h["k_fights"], 300.0      # kr: one round of pseudo-time
    n, c = s["n"], lambda k: s["c_" + k]
    f = {}
    f["att_pm"] = 60 * _ratio(c("sig_a"), c("secs"), pp["sig_a"], ks)
    f["r_att_pm"] = 60 * _ratio(s["r_sig_a"], s["r_secs"], pp["sig_a"], ks)
    f["oatt_pm"] = 60 * _ratio(c("osig_a"), c("secs"), pp["osig_a"], ks)
    f["r1_att_pm"] = 60 * _ratio(c("r1_att"), c("r1_secs"), pp["r1_att"], kr)
    f["r_r1_att_pm"] = 60 * _ratio(s["r_r1_att"], s["r_r1_secs"], pp["r1_att"], kr)
    f["r1_abs_pm"] = 60 * _ratio(c("r1_abs"), c("r1_secs"), pp["r1_abs"], kr)
    f["late_att_pm"] = 60 * _ratio(c("late_att"), c("late_secs"), pp["late_att"], kr)
    f["fade"] = f["late_att_pm"] - f["r1_att_pm"]
    f["avg_fight_min"] = _ratio(c("tsecs"), n, pp["tsecs"], kf) / 60
    f["dec_rate"] = _ratio(c("went_dec"), n, pp["went_dec"], kf)
    f["early_fin_w_rate"] = _ratio(c("early_fin_w"), n, pp["early_fin_w"], kf)
    f["r1_fin_w_rate"] = _ratio(c("r1_fin_w"), n, pp["r1_fin_w"], kf)
    f["early_fin_l_rate"] = _ratio(c("early_fin_l"), n, pp["early_fin_l"], kf)
    for k in ("ko_w", "sub_w", "ko_l", "sub_l"):
        f[k + "_p15"] = 900 * _ratio(s[k], c("tsecs"), pp[k + "_ps"], ks)
    f["avg_fin_round_w"] = _ratio(c("fin_round_w"), s["fin_w"], pp["fin_round_w"], 1.0)
    f["five_round_fights"] = c("n5")
    f["fights_reaching_r4"] = c("reach4")
    f["late_minutes"] = c("late_total_secs") / 60
    f["r3_win_rate"] = _ratio(c("reach3_w"), c("reach3_dec"), pp["r3_win"], kf)
    f["fights_reaching_r3"] = c("reach3")
    for k in ("tsecs", "r1_att", "r1_secs", "late_att", "late_secs", "early_fin_w", "n5", "reach3_w"):
        f["rawp_" + k] = c(k)
    return f


def _ratio(num, den, prior, k):
    """num/den shrunk toward `prior` with k pseudo-units of denominator."""
    if prior != prior:
        return num / den if den > 0 else np.nan
    return (num + prior * k) / (den + k)


def _snapshot(s, prior, r, cfg):
    """Pre-fight features for one fighter, from state `s` and the debutant prior."""
    h = cfg["history"]
    ks, ka, kt, kf = h["k_seconds"], h["k_sig_attempts"], h["k_td_attempts"], h["k_fights"]
    f = {"fight_id": r["fight_id"], "side": r["side"], "fighter": r["fighter"], "name": r["name"],
         "date": r["date"]}
    n = s["n"]
    f["debut"] = float(n == 0)
    f["n_fights"] = float(n)
    f["log_n_fights"] = np.log1p(n)
    f["ufc_minutes"] = s["total_secs"] / 60
    f["win_rate"] = _ratio(s["wins"], s["decided"], prior["win"], kf)
    f["recent_win_rate"] = _ratio(s["ew_score"], s["ew_n"], prior["win"], kf / 2)
    f["streak"] = float(s["streak"])
    f["finish_rate"] = _ratio(s["fin_w"], n, prior["fin_w"], kf)
    f["ko_win_rate"] = _ratio(s["ko_w"], n, prior["ko_w"], kf)
    f["sub_win_rate"] = _ratio(s["sub_w"], n, prior["sub_w"], kf)
    f["ko_loss_rate"] = _ratio(s["ko_l"], n, prior["ko_l"], kf)
    f["sub_loss_rate"] = _ratio(s["sub_l"], n, prior["sub_l"], kf)
    f["ko_losses"] = float(s["ko_l"])
    f["days_since"] = (min((r["date"] - s["last"]).days, h["layoff_cap_days"])
                       if s["last"] is not None else np.nan)
    f["fights_2y"] = float(sum((r["date"] - d).days <= 730 for d in s["dates"]))
    f["elo"] = s["elo"]
    f["avg_opp_elo"] = s["opp_elo_sum"] / n if n else np.nan
    for p in ("c", "r"):
        g = lambda k: s[f"{p}_{k}"]
        secs, cs = g("secs"), g("csecs")
        f[f"{p}_slpm"] = 60 * _ratio(g("sig_l"), secs, prior["sig_l"], ks)
        f[f"{p}_sapm"] = 60 * _ratio(g("osig_l"), secs, prior["osig_l"], ks)
        f[f"{p}_sig_acc"] = _ratio(g("sig_l"), g("sig_a"), prior["sig_acc"], ka)
        f[f"{p}_sig_def"] = 1 - _ratio(g("osig_l"), g("osig_a"), prior["osig_acc"], ka)
        f[f"{p}_kd_p15"] = 900 * _ratio(g("kd"), secs, prior["kd"], ks)
        f[f"{p}_kda_p15"] = 900 * _ratio(g("okd"), secs, prior["okd"], ks)
        f[f"{p}_td_p15"] = 900 * _ratio(g("td_l"), secs, prior["td_l"], ks)
        f[f"{p}_tda_p15"] = 900 * _ratio(g("otd_l"), secs, prior["otd_l"], ks)
        f[f"{p}_td_acc"] = _ratio(g("td_l"), g("td_a"), prior["td_acc"], kt)
        f[f"{p}_td_def"] = 1 - _ratio(g("otd_l"), g("otd_a"), prior["otd_acc"], kt)
        f[f"{p}_sub_p15"] = 900 * _ratio(g("sub"), secs, prior["sub"], ks)
        f[f"{p}_ctrl_share"] = _ratio(g("ctrl"), cs, prior["ctrl"], ks)
        f[f"{p}_ctrla_share"] = _ratio(g("octrl"), cs, prior["octrl"], ks)
    # static attributes (current values from ufcstats; effectively fixed for an adult)
    f["age"] = (r["date"] - r["dob"]).days / 365.25 if pd.notna(r["dob"]) else np.nan
    f["height_cm"] = r["height_cm"]
    # a profile with height but no reach gets the reach implied by its height, so that a
    # missing reach is not recognisable as an exact zero difference
    rh = h["reach_from_height"]
    f["reach_cm"] = (r["reach_cm"] if r["reach_cm"] == r["reach_cm"]
                     else rh["intercept"] + rh["slope"] * r["height_cm"])
    f["southpaw"] = float(r["stance"] == "Southpaw") if isinstance(r["stance"], str) else np.nan
    # raw sums, kept so tests can re-derive them independently
    for k in ("wins", "losses", "decided", "ko_l"):
        f["raw_" + k] = float(s[k])
    for k in ("secs", "sig_l", "td_l", "ctrl"):
        f["raw_" + k] = s["c_" + k]
    return f


def _prior_rates(P):
    """Debutant averages from debut fights on earlier dates inside the prior window."""
    def div(a, b):
        return a / b if b > 0 else np.nan
    out = {k: div(P[k], P["secs"]) for k in ("sig_l", "osig_l", "kd", "okd", "td_l", "otd_l", "sub")}
    out["sig_acc"] = div(P["sig_l"], P["sig_a"])
    out["osig_acc"] = div(P["osig_l"], P["osig_a"])
    out["td_acc"] = div(P["td_l"], P["td_a"])
    out["otd_acc"] = div(P["otd_l"], P["otd_a"])
    out["ctrl"] = div(P["ctrl"], P["csecs"])
    out["octrl"] = div(P["octrl"], P["csecs"])
    out["win"] = div(P["wins"], P["decided"])
    for k in ("fin_w", "ko_w", "sub_w", "ko_l", "sub_l"):
        out[k] = div(P[k], P["n"])
    return out


def prefight(bouts, cfg):
    """Pre-fight feature snapshot for every fighter in every fight."""
    decay = 0.5 ** (1 / cfg["history"]["recent_half_life_fights"])
    e = cfg["elo"]
    state = defaultdict(lambda: _new_state(cfg))
    P = defaultdict(float)   # running sums over recent debut fights
    recent_debuts = deque()  # (date, contributions) so old debuts can be dropped
    window = pd.Timedelta(days=365.25 * cfg["history"]["prior_window_years"])
    out = []
    for date, g in bouts.groupby("date", sort=True):
        rows = g.to_dict("records")
        while recent_debuts and recent_debuts[0][0] < date - window:
            for k, v in recent_debuts.popleft()[1].items():
                P[k] -= v
        prior = _prior_rates(P)
        pace_prior = _pace_priors(P)
        updates = []
        for r in rows:
            s = state[r["fighter"]]
            snap = _snapshot(s, prior, r, cfg)
            snap.update(_pace_snapshot(s, pace_prior, cfg))
            out.append(snap)
            opp_elo = state[r["opp"]]["elo"]
            delta = 0.0
            if r["score"] == r["score"]:
                expect = 1 / (1 + 10 ** ((opp_elo - s["elo"]) / 400))
                k = e["k"] * (1 + e["new_boost"] / (1 + s["n"]))
                delta = k * (r["score"] - expect)
            updates.append((r, delta, opp_elo, s["n"] == 0))
        # ---- apply the whole date at once
        for r, delta, opp_elo, was_debut in updates:
            s = state[r["fighter"]]
            b = _bout_sums(r)
            sc, mg = r["score"], r["method_group"]
            won, lost = sc == 1.0, sc == 0.0
            flags = {"n": 1, "decided": int(won or lost), "wins": int(won), "fin_w": int(won and mg in ("KO", "SUB")),
                     "ko_w": int(won and mg == "KO"), "sub_w": int(won and mg == "SUB"),
                     "ko_l": int(lost and mg == "KO"), "sub_l": int(lost and mg == "SUB")}
            for k, v in flags.items():
                s[k] += v
            s["losses"] += int(lost)
            s["dec_w"] += int(won and mg == "DEC")
            if won:
                s["streak"] = max(s["streak"], 0) + 1
            elif lost:
                s["streak"] = min(s["streak"], 0) - 1
            if sc == sc:
                s["ew_score"] = s["ew_score"] * decay + sc
                s["ew_n"] = s["ew_n"] * decay + 1
            for k in SUMS:
                s["c_" + k] += b[k]
                s["r_" + k] = s["r_" + k] * decay + b[k]
            pb = _pace_sums(r)
            for k in PACE_SUMS:
                s["c_" + k] += pb[k]
                s["r_" + k] = s["r_" + k] * decay + pb[k]
            s["total_secs"] += r["fight_secs"]
            s["elo"] += delta
            s["opp_elo_sum"] += opp_elo
            s["last"] = r["date"]
            s["dates"].append(r["date"])
            if was_debut:
                add = {**b, **flags, **pb}
                for k, v in add.items():
                    P[k] += v
                recent_debuts.append((r["date"], add))
    return pd.DataFrame(out)
