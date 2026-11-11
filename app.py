"""Streamlit front end for the frozen UFC model and the paper-trading ledger.

    .venv\\Scripts\\streamlit run app.py

It changes nothing about the model: scoring goes through ufc.paper.score with the
settings in config_frozen.json. No money is staked anywhere in this app.
"""
import difflib
import sys
from datetime import date, datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ufc import load_config  # noqa: E402
from ufc.clean import ident, load_clean  # noqa: E402
from ufc.metrics import log_loss  # noqa: E402
from ufc.paper import (LEDGER, LEDGER_COLS, UPCOMING, UPCOMING_COLS, append_ledger, fit_finish_history,  # noqa: E402
                       fit_history, load_finish_frozen, load_frozen, load_ledger, score_full, settle)

st.set_page_config(page_title="UFC model: paper trading", layout="wide")

CLASSES = ["Strawweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
           "Middleweight", "Light Heavyweight", "Heavyweight"]
BOUT_TYPES = ([f"{c} Bout" for c in CLASSES[1:]]
              + [f"Women's {c} Bout" for c in CLASSES[:4]]
              + [f"UFC {c} Title Bout" for c in CLASSES[1:]]
              + [f"UFC Women's {c} Title Bout" for c in CLASSES[:4]]
              + ["Catch Weight Bout"])


# ---------------------------------------------------------------- cached data
@st.cache_resource(show_spinner="Fitting the frozen model on the fight history (about a minute, once)...")
def fitted_model(data_stamp):
    return fit_history(load_config())


@st.cache_resource(show_spinner="Fitting the method and round model (about half a minute, once)...")
def fitted_finish(data_stamp):
    return fit_finish_history(load_config())


METHODS = {"ko": "knockout", "sub": "submission", "dec": "decision"}


def outcome_cols(side=None, method=None, rnd=None):
    """Columns of the method/round table that match a choice; None means any."""
    cols = []
    for s in (["f1", "f2"] if side is None else [side]):
        for m in (list(METHODS) if method is None else [method]):
            if m == "dec":
                if rnd is None:
                    cols.append(f"{s}_dec")
            else:
                cols += [f"{s}_{m}_{r}" for r in ([rnd] if rnd else range(1, 6))]
    return cols


def outcome_text(col, f1, f2):
    side, method, *rnd = col.split("_")
    name = f1 if side == "f1" else f2
    return f"{name} by decision" if method == "dec" else f"{name} by {METHODS[method]} in round {rnd[0]}"


def fair_price(p):
    return f"{1 / p:.2f}" if p > 0 else "none"


@st.cache_data
def known_fighters(data_stamp):
    c = load_clean(load_config())
    long = pd.concat([c[["fighter_1", "date"]].set_axis(["name", "date"], axis=1),
                      c[["fighter_2", "date"]].set_axis(["name", "date"], axis=1)])
    g = long.groupby("name").date.agg(["size", "max"])
    return g.rename(columns={"size": "fights", "max": "last_fight"}).sort_index(), c.date.max()


def data_stamp():
    cfg = load_config()
    p = Path(__file__).resolve().parent / cfg["paths"]["data"] / "fights_clean.csv"
    return p.stat().st_mtime


def to_decimal(v, american):
    if pd.isna(v):
        return np.nan
    if not american:
        return float(v)
    return 1 + v / 100 if v > 0 else 1 + 100 / abs(v)


def blank_card():
    return pd.DataFrame({"fighter_1": [""] * 6, "fighter_2": [""] * 6, "bout_type": ["Lightweight Bout"] * 6,
                         "rounds": [3] * 6, "f1_odds": [np.nan] * 6, "f2_odds": [np.nan] * 6})


def card_from_file():
    if not UPCOMING.exists():
        return None
    u = pd.read_csv(UPCOMING)
    if u.empty or not set(UPCOMING_COLS) <= set(u.columns):
        return None
    card = u[["fighter_1", "fighter_2", "bout_type", "rounds"]].copy()
    card["f1_odds"], card["f2_odds"] = u.f1_decimal, u.f2_decimal
    book = u.bookmaker.dropna()
    return {"card": card, "date": pd.Timestamp(u.date.iloc[0]).date(), "event": str(u.event.iloc[0]),
            "bookmaker": str(book.iloc[0]) if len(book) else ""}


fighters, last_data_date = known_fighters(data_stamp())
frozen = load_frozen()
rule = frozen["betting_rule"]

st.title("UFC model: paper trading")
st.caption(f"Frozen model. Fight history through {last_data_date:%d %B %Y}. "
           f"A fight is a paper bet when the edge is above {rule['edge_threshold']:.0%} and not above "
           f"{rule['edge_cap']:.0%}. No money is staked.")

tab_score, tab_ledger, tab_results, tab_about = st.tabs(["Score a card", "Ledger", "Results", "About the model"])

# ================================================================ score a card
with tab_score:
    if "card" not in st.session_state:
        loaded = card_from_file()
        st.session_state.card = loaded["card"] if loaded else blank_card()
        st.session_state.event_date = loaded["date"] if loaded else date.today()
        st.session_state.event_name = loaded["event"] if loaded else ""
        st.session_state.bookmaker = loaded["bookmaker"] if loaded else ""

    c1, c2, c3, c4 = st.columns([1, 2, 2, 1.4])
    event_date = c1.date_input("Fight date", st.session_state.event_date)
    event_name = c2.text_input("Event", st.session_state.event_name)
    bookmaker = c3.text_input("Where the odds are from", st.session_state.bookmaker)
    fmt = c4.radio("Odds format", ["Decimal", "American"], horizontal=True)
    american = fmt == "American"

    st.write("Enter one fight per row. Use the odds you can get right now, and enter them as early as you can.")
    card = st.data_editor(
        st.session_state.card, num_rows="dynamic", width="stretch", key="card_editor",
        column_config={
            "fighter_1": st.column_config.TextColumn("Fighter 1", required=True),
            "fighter_2": st.column_config.TextColumn("Fighter 2", required=True),
            "bout_type": st.column_config.SelectboxColumn("Bout", options=BOUT_TYPES, required=True),
            "rounds": st.column_config.SelectboxColumn("Rounds", options=[3, 5], required=True),
            "f1_odds": st.column_config.NumberColumn("Fighter 1 odds", format="%.2f"),
            "f2_odds": st.column_config.NumberColumn("Fighter 2 odds", format="%.2f"),
        })
    card = card[(card.fighter_1.fillna("").str.strip() != "") & (card.fighter_2.fillna("").str.strip() != "")]

    # ---- name check: no model needed
    idents = {ident(n): n for n in fighters.index}
    problems = []
    for nm in pd.concat([card.fighter_1, card.fighter_2]).str.strip().unique():
        if ident(nm) not in idents:
            near = difflib.get_close_matches(nm, list(fighters.index), n=3, cutoff=0.75)
            problems.append(f"**{nm}** is not in the data and will be treated as a UFC debutant."
                            + (f" Did you mean: {', '.join(near)}?" if near else ""))
    if problems:
        st.warning("\n\n".join(problems))
    with st.expander("Look up how a fighter's name is spelled in the data"):
        q = st.text_input("Type part of a name", key="lookup")
        if q:
            hits = fighters[fighters.index.str.contains(q, case=False, regex=False)].copy()
            hits["last_fight"] = hits.last_fight.dt.date
            st.dataframe(hits.head(30), width="stretch")

    errors = []
    if len(card) == 0:
        errors.append("Add at least one fight.")
    if pd.Timestamp(event_date) <= last_data_date:
        errors.append(f"The fight date must be after {last_data_date:%d %B %Y}, the last fight in the data.")
    if (card.fighter_1.str.strip().map(ident) == card.fighter_2.str.strip().map(ident)).any():
        errors.append("A fight has the same fighter on both sides.")
    odds = pd.concat([card.f1_odds, card.f2_odds]).dropna()
    if american and ((odds.abs() < 100).any()):
        errors.append("American odds must be at least +100 or at most -100.")
    if not american and ((odds <= 1).any()):
        errors.append("Decimal odds must be greater than 1.")
    one_sided = card.f1_odds.isna() ^ card.f2_odds.isna()
    if one_sided.any():
        errors.append("Enter odds for both fighters in a fight, or leave both blank.")
    for e in errors:
        st.error(e)

    if st.button("Score this card", type="primary", disabled=bool(errors)):
        up = pd.DataFrame({
            "date": pd.Timestamp(event_date).strftime("%Y-%m-%d"), "event": event_name.strip() or "Unnamed event",
            "fighter_1": card.fighter_1.str.strip().map(lambda n: idents.get(ident(n), n)).values,
            "fighter_2": card.fighter_2.str.strip().map(lambda n: idents.get(ident(n), n)).values,
            "bout_type": card.bout_type.fillna("Catch Weight Bout").values,
            "rounds": card.rounds.fillna(3).astype(int).values,
            "f1_decimal": [to_decimal(v, american) for v in card.f1_odds],
            "f2_decimal": [to_decimal(v, american) for v in card.f2_odds],
            "bookmaker": bookmaker.strip()})[UPCOMING_COLS]
        fit, fin_fit = fitted_model(data_stamp()), fitted_finish(data_stamp())
        with st.spinner("Scoring the card..."):
            res, n_fit, last, fin = score_full(up, load_config(), fit, fin_fit)
        UPCOMING.parent.mkdir(exist_ok=True)
        up.to_csv(UPCOMING, index=False)
        st.session_state.result = res
        st.session_state.finish = fin
        st.session_state.rounds = up.rounds.values
        st.session_state.logged_msg = None

    res = st.session_state.get("result")
    if res is not None:
        scored = res[res.model_p_f1.notna()].copy()
        st.subheader("Predictions")
        if scored.empty:
            st.info("No fight has odds yet, so nothing was scored.")
        else:
            f1_fav = scored.model_p_f1 >= 0.5
            view = pd.DataFrame({
                "Fight": scored.fighter_1 + " vs " + scored.fighter_2,
                "Model's pick to win": np.where(f1_fav, scored.fighter_1, scored.fighter_2),
                "Model": np.where(f1_fav, scored.model_p_f1, 1 - scored.model_p_f1) * 100,
                "Odds imply": np.where(f1_fav, scored.market_p_f1, 1 - scored.market_p_f1) * 100,
                "Best-value side": scored.pick, "Price": scored.pick_price, "Edge": scored.edge * 100,
                "Paper bet": np.where(scored.bet, "Yes", "")})
            st.dataframe(view, width="stretch", hide_index=True, column_config={
                "Model": st.column_config.NumberColumn(format="%.1f%%"),
                "Odds imply": st.column_config.NumberColumn(format="%.1f%%", help="Bookmaker margin removed"),
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Edge": st.column_config.NumberColumn(format="%+.1f%%", help="Model probability x price - 1"),
            })
            bets = scored[scored.bet]
            m1, m2, m3 = st.columns(3)
            m1.metric("Fights scored", len(scored))
            m2.metric("Paper bets", len(bets))
            m3.metric("Average edge on bets", f"{bets.edge.mean():+.1%}" if len(bets) else "none")
        fin = st.session_state.get("finish")
        if fin is not None and len(scored):
            st.subheader("How and when each fight ends")
            st.caption("Knockout includes technical knockouts and doctor stoppages. These probabilities beat simple "
                       "base rates on fights the model had never seen, but they have not been shown to beat "
                       "bookmakers' method or round prices, so no paper bets are made on them.")
            P = fin.loc[scored.index] * 100
            names = {i: (scored.fighter_1[i], scored.fighter_2[i]) for i in scored.index}
            finish_cols = [c for c in P.columns if not c.endswith("_dec")]
            top, top_fin = P.idxmax(axis=1), P[finish_cols].idxmax(axis=1)
            summary = pd.DataFrame({
                "Fight": scored.fighter_1 + " vs " + scored.fighter_2,
                "Knockout": P[outcome_cols(method="ko")].sum(axis=1),
                "Submission": P[outcome_cols(method="sub")].sum(axis=1),
                "Decision": P[outcome_cols(method="dec")].sum(axis=1),
                "Most likely result": [outcome_text(top[i], *names[i]) for i in P.index],
                "Chance": [P.loc[i, top[i]] for i in P.index],
                "Most likely finish": [outcome_text(top_fin[i], *names[i]) for i in P.index],
                "Chance ": [P.loc[i, top_fin[i]] for i in P.index]})
            pct = st.column_config.NumberColumn(format="%.1f%%")
            st.dataframe(summary, width="stretch", hide_index=True, column_config={
                c: pct for c in ("Knockout", "Submission", "Decision", "Chance", "Chance ")})

            labels = {i: f"{a} vs {b}" for i, (a, b) in names.items()}
            pick = st.selectbox("Fight in detail", list(labels), format_func=labels.get, key="finish_fight")
            if pick not in P.index:
                pick = P.index[0]
            f1, f2 = names[pick]
            row = P.loc[pick]
            n_rounds = int(st.session_state.rounds[pick])
            val = lambda **kw: float(row[outcome_cols(**kw)].sum())
            left, right = st.columns(2)
            with left:
                st.write("**By method**")
                st.dataframe(pd.DataFrame({
                    "": ["Knockout", "Submission", "Decision", "Total"],
                    f1: [val(side="f1", method=m) for m in METHODS] + [val(side="f1")],
                    f2: [val(side="f2", method=m) for m in METHODS] + [val(side="f2")],
                    "Either": [val(method=m) for m in METHODS] + [100.0]}), hide_index=True, width="stretch",
                    column_config={c: pct for c in (f1, f2, "Either")})
                st.write("**Most likely exact results**")
                best = row.sort_values(ascending=False).head(6)
                st.dataframe(pd.DataFrame({"Result": [outcome_text(c, f1, f2) for c in best.index],
                                           "Chance": best.values,
                                           "Fair price": [fair_price(v / 100) for v in best.values]}),
                             hide_index=True, width="stretch", column_config={"Chance": pct})
            with right:
                st.write("**By round**")
                rounds = list(range(1, n_rounds + 1))
                ends = [val(rnd=r) for r in rounds]
                st.dataframe(pd.DataFrame({
                    "Round": [str(r) for r in rounds] + ["Goes the distance"],
                    f"{f1} knockout": [val(side="f1", method="ko", rnd=r) for r in rounds] + [None],
                    f"{f1} submission": [val(side="f1", method="sub", rnd=r) for r in rounds] + [None],
                    f"{f2} knockout": [val(side="f2", method="ko", rnd=r) for r in rounds] + [None],
                    f"{f2} submission": [val(side="f2", method="sub", rnd=r) for r in rounds] + [None],
                    "Fight ends here": ends + [val(method="dec")]}), hide_index=True, width="stretch",
                    column_config={c: pct for c in (f"{f1} knockout", f"{f1} submission", f"{f2} knockout",
                                                    f"{f2} submission", "Fight ends here")})
                st.write("**Build a prediction**")
                b1, b2, b3 = st.columns(3)
                who = b1.selectbox("Winner", ["Either", f1, f2], key="combo_who")
                how = b2.selectbox("Method", ["Any", "Knockout", "Submission", "Decision"], key="combo_how")
                when = b3.selectbox("Round", ["Any"] + rounds, key="combo_when", disabled=how == "Decision")
                side = {f1: "f1", f2: "f2"}.get(who)
                method = {"Knockout": "ko", "Submission": "sub", "Decision": "dec"}.get(how)
                rnd = None if when == "Any" or how == "Decision" else int(when)
                p = val(side=side, method=method, rnd=rnd) / 100
                said = (f"{who} wins" if side else "The fight ends") + (f" by {how.lower()}" if method else "") \
                    + (f" in round {rnd}" if rnd else "")
                if rnd and not method:
                    said += " (by knockout or submission)"
                st.metric(said, f"{p:.1%}", help="Fair price is 1 divided by the probability.")
                st.caption(f"Fair decimal price: {fair_price(p)}")

        notes = res[res.notes.fillna("") != ""]
        if len(notes):
            st.write("**Notes**")
            for r in notes.itertuples():
                st.write(f"- {r.fighter_1} vs {r.fighter_2}: {r.notes}")
        if len(scored):
            st.write("Logging is permanent for the price: the first logged price for a fight stands.")
            if st.button("Log these fights to the ledger"):
                added, skipped = append_ledger(res, datetime.now().strftime("%Y-%m-%d %H:%M"))
                st.session_state.logged_msg = (f"Logged {added} fight(s)."
                                               + (f" {skipped} were already in the ledger and left unchanged."
                                                  if skipped else ""))
            if st.session_state.get("logged_msg"):
                st.success(st.session_state.logged_msg)

# ================================================================ ledger
with tab_ledger:
    led = load_ledger()
    if led.empty:
        st.info("The ledger is empty. Score a card and log it first.")
    else:
        st.write("After a card, fill in the closing odds (decimal) and the winner for each fight, then save. "
                 "Type the winner's name exactly as shown, or `draw`, or `nc`.")
        only_open = st.toggle("Show only fights without a result", value=False)
        show = led[led.winner.isna()] if only_open else led
        show = show.copy()
        show["bet"] = show.bet.astype(str).str.lower().isin(["true", "1"])
        show["winner"] = show.winner.astype("object")
        fixed = [c for c in LEDGER_COLS if c not in ("close_f1_decimal", "close_f2_decimal", "winner")]
        edited = st.data_editor(
            show, width="stretch", hide_index=True, disabled=fixed, key="ledger_editor",
            column_order=["date", "fighter_1", "fighter_2", "f1_decimal", "f2_decimal", "model_p_f1", "pick",
                          "pick_price", "edge", "bet", "close_f1_decimal", "close_f2_decimal", "winner",
                          "bookmaker", "notes", "logged_at"],
            column_config={
                "f1_decimal": st.column_config.NumberColumn("F1 price", format="%.2f"),
                "f2_decimal": st.column_config.NumberColumn("F2 price", format="%.2f"),
                "model_p_f1": st.column_config.NumberColumn("Model P(F1)", format="%.3f"),
                "pick_price": st.column_config.NumberColumn("Pick price", format="%.2f"),
                "edge": st.column_config.NumberColumn("Edge", format="%.3f"),
                "bet": st.column_config.CheckboxColumn("Paper bet"),
                "close_f1_decimal": st.column_config.NumberColumn("F1 closing price", format="%.2f", min_value=1.01),
                "close_f2_decimal": st.column_config.NumberColumn("F2 closing price", format="%.2f", min_value=1.01),
                "winner": st.column_config.TextColumn("Winner"),
            })
        bad = []
        for r in edited.itertuples():
            w = str(r.winner).strip() if pd.notna(r.winner) else ""
            if w and w not in (r.fighter_1, r.fighter_2) and w.lower() not in ("draw", "nc", "no contest", "void"):
                bad.append(f"{r.fighter_1} vs {r.fighter_2}: winner `{w}` is not one of the two fighters, `draw` or `nc`.")
        for b in bad:
            st.error(b)
        if st.button("Save results", type="primary", disabled=bool(bad)):
            out = led.copy()
            out["winner"] = out.winner.astype("object")
            for c in ("close_f1_decimal", "close_f2_decimal", "winner"):
                out.loc[edited.index, c] = edited[c]
            out[LEDGER_COLS].to_csv(LEDGER, index=False)
            st.success("Saved.")
            st.rerun()

# ================================================================ results
with tab_results:
    led = load_ledger()
    done = settle(led) if len(led) else pd.DataFrame()
    if len(done) == 0:
        st.info("No results yet. After a card, fill in the winners on the Ledger tab and the results appear here.")
    else:
        bets = done[done.is_bet].sort_values(["date", "logged_at"]).reset_index(drop=True)
        decided = done[done.decided]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Settled paper bets", len(bets), help=f"{int(led.bet.astype(str).str.lower().isin(['true', '1']).sum())} logged in total")
        if len(bets):
            k2.metric("Return per unit staked", f"{bets.bet_profit.mean():+.1%}")
            k3.metric("Units won or lost", f"{bets.bet_profit.sum():+.2f}")
            clv = bets.clv.dropna()
            k4.metric("Closing-line value", f"{clv.mean():+.1%}" if len(clv) else "no closing prices",
                      help="Value of your price against the closing price, margin removed. Above zero is good.")
        if len(bets) < 200:
            st.caption(f"{len(bets)} settled bets. Closing-line value starts to mean something at about 100 bets "
                       "and the return at about 200; before that these numbers are mostly luck.")
        if len(decided):
            ll_m, ll_o = log_loss(decided.f1_won, decided.model_p_f1), log_loss(decided.f1_won, decided.market_p_f1)
            st.write(f"On all {len(decided)} decided fights logged: model log loss **{ll_m:.4f}**, "
                     f"your prices **{ll_o:.4f}** (lower is better).")

        if len(bets):
            st.subheader("Running total of paper bets, in units")
            bets["bet_number"] = np.arange(1, len(bets) + 1)
            bets["units"] = bets.bet_profit.cumsum()
            bets["result"] = np.where(~bets.decided, "void", np.where(bets.bet_profit > 0, "won", "lost"))
            start = pd.DataFrame({"bet_number": [0], "units": [0.0]})
            line_data = pd.concat([start, bets[["bet_number", "units"]]], ignore_index=True)
            dark = st.context.theme.type == "dark" if hasattr(st, "context") and hasattr(st.context, "theme") else False
            series = "#3987e5" if dark else "#2a78d6"
            x = alt.X("bet_number:Q", title="Settled bets", axis=alt.Axis(tickMinStep=1, grid=False))
            y = alt.Y("units:Q", title="Units")
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeWidth=1, opacity=0.5).encode(y="y:Q")
            line = alt.Chart(line_data).mark_line(strokeWidth=2, color=series).encode(x=x, y=y)
            points = alt.Chart(bets).mark_point(size=70, filled=True, color=series, opacity=1).encode(
                x=x, y=y,
                tooltip=[alt.Tooltip("date:N", title="Date"), alt.Tooltip("pick:N", title="Bet on"),
                         alt.Tooltip("pick_price:Q", title="Price", format=".2f"),
                         alt.Tooltip("result:N", title="Result"),
                         alt.Tooltip("bet_profit:Q", title="This bet", format="+.2f"),
                         alt.Tooltip("units:Q", title="Running total", format="+.2f")])
            st.altair_chart((zero + line + points).properties(height=320), width="stretch")

            st.subheader("Settled bets")
            tbl = bets[["date", "pick", "pick_price", "edge", "result", "bet_profit", "clv", "units"]].rename(columns={
                "date": "Date", "pick": "Bet on", "pick_price": "Price", "edge": "Edge at logging",
                "result": "Result", "bet_profit": "Profit", "clv": "Closing-line value", "units": "Running total"})
            st.dataframe(tbl, width="stretch", hide_index=True, column_config={
                "Price": st.column_config.NumberColumn(format="%.2f"),
                "Edge at logging": st.column_config.NumberColumn(format="%+.3f"),
                "Profit": st.column_config.NumberColumn(format="%+.2f"),
                "Closing-line value": st.column_config.NumberColumn(format="%+.3f"),
                "Running total": st.column_config.NumberColumn(format="%+.2f")})
        if done.unrecognised.any():
            st.warning("Some `winner` entries are not a fighter name, `draw` or `nc`; fix them on the Ledger tab.")

# ================================================================ about
with tab_about:
    st.subheader("What this model is")
    st.write(
        "It starts from the odds you enter and adjusts them using each fighter's UFC history: age, recent form, "
        "striking and grappling rates, layoffs and experience. It is the average of a logistic regression and a "
        "gradient-boosting model, with the settings frozen on 11 November 2026.")
    st.subheader("How it scored on the one-time test")
    st.write("932 fights from January 2025 to October 2026 that the model had never seen.")
    st.dataframe(pd.DataFrame({
        "": ["Coin flip", "Opening odds", "This model", "Closing odds"],
        "Accuracy": ["50.0%", "66.6%", "67.3%", "69.7%"],
        "Log loss (lower is better)": [0.693, 0.606, 0.600, 0.584]}), hide_index=True, width="content")
    st.write(
        "- The model was slightly better than the opening odds, but by too little to rule out luck. "
        "It met 1 of the 3 conditions set in advance.\n"
        "- Bets at opening prices returned +5.2% on 322 bets, with a 95% range of -3.6% to +13.7%.\n"
        "- The same picks lost 10% at closing prices. Any edge depends on getting early prices.\n"
        "- This is why the bets here are on paper only.")
    st.subheader("How and when: the method and round model")
    st.write(
        "A second model takes the winner probabilities and splits them by method and round. For each fighter it "
        "walks through the rounds: knockout, submission, or the fight continues; a fight that reaches the end is a "
        "decision. It uses finishing and durability history, pace, weight class, the fight length and the year.")
    st.write("Its one-time test: 931 fights from January 2025 to October 2026.")
    st.dataframe(pd.DataFrame({
        "": ["Base rates", "This model"],
        "Method pick correct": ["48.9%", "53.9%"],
        "Method log loss": [1.019, 0.958],
        "Exact result in the top 3": ["61.4%", "63.8%"],
        "Exact result log loss": [2.163, 2.072]}), hide_index=True, width="content")
    st.write(
        "- It met all 3 conditions set in advance, each against base rates.\n"
        "- On 2019-2024, bookmakers' method prices were slightly better than the model. Treat these numbers as "
        "information, not as betting edges.\n"
        "- An exact result (fighter, method and round) is rarely above 10%, even when the model is right about "
        "the shape of the fight.")
    st.subheader("Limits")
    st.write(
        f"- Fight history ends on {last_data_date:%d %B %Y}. Later fights are unknown to the model until the "
        "dataset is updated.\n"
        "- A fighter not in the data is treated as a UFC debutant.\n"
        "- The model knows nothing about injuries, short-notice replacements or missed weight.")
    with st.expander("Frozen settings"):
        st.json(frozen)
    with st.expander("Frozen settings: method and round model"):
        st.json(load_finish_frozen())
