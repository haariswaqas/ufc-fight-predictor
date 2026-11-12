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

st.set_page_config(page_title="UFC fight predictor", layout="centered")

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
    return pd.DataFrame({"fighter_1": pd.Series(dtype="object"), "fighter_2": pd.Series(dtype="object"),
                         "bout_type": pd.Series(dtype="object"), "rounds": pd.Series(dtype="int64"),
                         "f1_odds": pd.Series(dtype="float64"), "f2_odds": pd.Series(dtype="float64")})


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

st.title("UFC fight predictor")
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
        st.session_state.card_v = 0

    def replace_card(new_card):
        """Swap the card and reset the table editor so it shows the new rows."""
        st.session_state.card = new_card.reset_index(drop=True)
        st.session_state.card_v += 1
        st.rerun()

    with st.expander("Event details", expanded=not st.session_state.event_name):
        c1, c2 = st.columns(2)
        event_date = c1.date_input("Fight date", st.session_state.event_date)
        event_name = c2.text_input("Event", st.session_state.event_name)
        c3, c4 = st.columns(2)
        bookmaker = c3.text_input("Where the odds are from", st.session_state.bookmaker)
        fmt = c4.radio("Odds format", ["Decimal", "American"], horizontal=True)
    american = fmt == "American"

    # ---- add a fight: a form works better than a table on a phone
    with st.expander("Add a fight", expanded=len(st.session_state.card) == 0):
        with st.form("add_fight", clear_on_submit=True):
            names = list(fighters.index)
            a, b = st.columns(2)
            n1 = a.selectbox("Fighter 1", names, index=None, placeholder="Type a name", accept_new_options=True)
            n2 = b.selectbox("Fighter 2", names, index=None, placeholder="Type a name", accept_new_options=True)
            a, b = st.columns(2)
            bout = a.selectbox("Bout", BOUT_TYPES, index=BOUT_TYPES.index("Lightweight Bout"))
            n_rounds = b.radio("Rounds", [3, 5], horizontal=True)
            a, b = st.columns(2)
            o1 = a.number_input("Fighter 1 odds", value=None, format="%.2f")
            o2 = b.number_input("Fighter 2 odds", value=None, format="%.2f")
            st.caption("Use the odds you can get right now. Leave both blank if there are none yet. "
                       "A name not in the list is treated as a UFC debutant.")
            if st.form_submit_button("Add to card", type="primary"):
                if not n1 or not n2:
                    st.error("Choose both fighters.")
                else:
                    st.session_state.pending_add = {"fighter_1": n1, "fighter_2": n2, "bout_type": bout,
                                                    "rounds": n_rounds, "f1_odds": o1, "f2_odds": o2}

    with st.expander("Edit the card as a table"):
        card = st.data_editor(
            st.session_state.card, num_rows="dynamic", width="stretch", key=f"card_editor_{st.session_state.card_v}",
            column_config={
                "fighter_1": st.column_config.TextColumn("Fighter 1", required=True),
                "fighter_2": st.column_config.TextColumn("Fighter 2", required=True),
                "bout_type": st.column_config.SelectboxColumn("Bout", options=BOUT_TYPES, required=True),
                "rounds": st.column_config.SelectboxColumn("Rounds", options=[3, 5], required=True),
                "f1_odds": st.column_config.NumberColumn("Fighter 1 odds", format="%.2f"),
                "f2_odds": st.column_config.NumberColumn("Fighter 2 odds", format="%.2f"),
            })
    card = card[(card.fighter_1.fillna("").str.strip() != "") & (card.fighter_2.fillna("").str.strip() != "")]
    card = card.reset_index(drop=True)
    if st.session_state.get("pending_add"):
        row = st.session_state.pop("pending_add")
        replace_card(pd.concat([card, pd.DataFrame([row])], ignore_index=True))

    st.subheader(f"Card: {len(card)} fight{'' if len(card) == 1 else 's'}")
    for i, r in card.iterrows():
        with st.container(border=True):
            left, right = st.columns([5, 1], vertical_alignment="center")
            priced = pd.notna(r.f1_odds) and pd.notna(r.f2_odds)
            left.markdown(f"**{r.fighter_1}** vs **{r.fighter_2}**  \n"
                          f"{r.bout_type} · {int(r.rounds) if pd.notna(r.rounds) else 3} rounds · "
                          + (f"odds {r.f1_odds:.2f} / {r.f2_odds:.2f}" if priced else "no odds yet"))
            if right.button("Remove", key=f"rm_{st.session_state.card_v}_{i}", width="stretch"):
                replace_card(card.drop(i))

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

    if st.button("Score this card", type="primary", disabled=bool(errors), width="stretch"):
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
        fin = st.session_state.get("finish")
        st.subheader("Predictions")
        if scored.empty:
            st.info("No fight has odds yet, so nothing was scored.")
        else:
            bets = scored[scored.bet]
            st.caption(f"{len(scored)} fight{'' if len(scored) == 1 else 's'} scored · {len(bets)} paper "
                       f"bet{'' if len(bets) == 1 else 's'}"
                       + (f" · average edge on bets {bets.edge.mean():+.1%}" if len(bets) else ""))
        pct = st.column_config.NumberColumn(format="%.1f%%")
        for i, r in scored.iterrows():
            f1, f2, p1 = r.fighter_1, r.fighter_2, float(r.model_p_f1)
            with st.container(border=True):
                st.markdown(f"**{f1} vs {f2}**" + ("  :green-badge[Paper bet]" if r.bet else ""))
                fav, p_fav = (f1, p1) if p1 >= 0.5 else (f2, 1 - p1)
                st.progress(p1, text=f"{f1} {p1:.0%} · {f2} {1 - p1:.0%}")
                st.markdown(f"Pick to win: **{fav}** ({p_fav:.1%}; the odds imply "
                            f"{r.market_p_f1 if fav == f1 else 1 - r.market_p_f1:.1%})  \n"
                            f"Best value: **{r.pick}** at {r.pick_price:.2f}, edge {r.edge:+.1%}")
                if fin is None:
                    continue
                row = fin.loc[i] * 100
                val = lambda **kw: float(row[outcome_cols(**kw)].sum())
                top = row.idxmax()
                st.markdown(f"Knockout {val(method='ko'):.0f}% · Submission {val(method='sub'):.0f}% · "
                            f"Decision {val(method='dec'):.0f}%  \n"
                            f"Most likely result: **{outcome_text(top, f1, f2)}** ({row[top]:.0f}%)")
                with st.expander("How and when"):
                    s1, s2 = f1.split()[-1], f2.split()[-1]
                    if s1 == s2:
                        s1, s2 = f1, f2
                    rounds = list(range(1, int(st.session_state.rounds[i]) + 1))
                    t_build, t_method, t_round, t_exact = st.tabs(["Build a prediction", "By method", "By round",
                                                                    "Exact results"])
                    with t_build:
                        who = st.selectbox("Winner", ["Either", f1, f2], key=f"who_{i}")
                        how = st.selectbox("Method", ["Any", "Knockout", "Submission", "Decision"], key=f"how_{i}")
                        when = st.selectbox("Round", ["Any"] + [str(x) for x in rounds], key=f"when_{i}", disabled=how == "Decision")
                        side = {f1: "f1", f2: "f2"}.get(who)
                        method = {"Knockout": "ko", "Submission": "sub", "Decision": "dec"}.get(how)
                        rnd = None if when == "Any" or how == "Decision" else int(when)
                        p = val(side=side, method=method, rnd=rnd) / 100
                        said = (f"{who} wins" if side else "The fight ends") + (f" by {how.lower()}" if method else "") \
                            + (f" in round {rnd}" if rnd else "")
                        if rnd and not method:
                            said += " (by knockout or submission)"
                        st.metric(said, f"{p:.1%}")
                        st.caption(f"Fair decimal price: {fair_price(p)} (the price at which this bet breaks even)")
                    with t_method:
                        st.dataframe(pd.DataFrame({
                            "": ["Knockout", "Submission", "Decision", "Total"],
                            s1: [val(side="f1", method=m) for m in METHODS] + [val(side="f1")],
                            s2: [val(side="f2", method=m) for m in METHODS] + [val(side="f2")],
                            "Either": [val(method=m) for m in METHODS] + [100.0]}), hide_index=True, width="stretch",
                            column_config={c: pct for c in (s1, s2, "Either")})
                    with t_round:
                        st.dataframe(pd.DataFrame({
                            "Round": [str(x) for x in rounds] + ["Distance"],
                            "Knockout": [val(method="ko", rnd=x) for x in rounds] + [None],
                            "Submission": [val(method="sub", rnd=x) for x in rounds] + [None],
                            "Ends here": [val(rnd=x) for x in rounds] + [val(method="dec")]}),
                            hide_index=True, width="stretch",
                            column_config={c: pct for c in ("Knockout", "Submission", "Ends here")})
                        st.caption("Either fighter. Use Build a prediction for one fighter in one round.")
                    with t_exact:
                        best = row.sort_values(ascending=False).head(6)
                        for c, v in best.items():
                            st.markdown(f"**{v:.1f}%** · {outcome_text(c, f1, f2)} · fair price {fair_price(v / 100)}")
        if fin is not None and len(scored):
            st.caption("Knockout includes technical knockouts and doctor stoppages. The method and round numbers beat "
                       "simple base rates on fights the model had never seen, but they have not been shown to beat "
                       "bookmakers' method or round prices, so no paper bets are made on them.")
        notes = res[res.notes.fillna("") != ""]
        if len(notes):
            st.write("**Notes**")
            for r in notes.itertuples():
                st.write(f"- {r.fighter_1} vs {r.fighter_2}: {r.notes}")
        if len(scored):
            if st.button("Log these fights to the ledger", width="stretch"):
                added, skipped = append_ledger(res, datetime.now().strftime("%Y-%m-%d %H:%M"))
                st.session_state.logged_msg = (f"Logged {added} fight(s)."
                                               + (f" {skipped} were already in the ledger and left unchanged."
                                                  if skipped else ""))
            st.caption("Logging is permanent for the price: the first logged price for a fight stands.")
            if st.session_state.get("logged_msg"):
                st.success(st.session_state.logged_msg)

# ================================================================ ledger
with tab_ledger:
    led = load_ledger()
    if led.empty:
        st.info("The ledger is empty. Score a card and log it first.")
    else:
        is_bet = led.bet.astype(str).str.lower().isin(["true", "1"])
        waiting = led[led.winner.isna()]
        st.caption(f"{len(led)} fights logged · {int(is_bet.sum())} paper bets · {len(waiting)} waiting for a result")
        if len(waiting):
            st.subheader("Enter a result")
            label = lambda i: f"{led.fighter_1[i]} vs {led.fighter_2[i]} ({led.date[i]})"
            i = st.selectbox("Fight", list(waiting.index), format_func=label, key="settle_fight")
            r = led.loc[i]
            with st.form(f"settle_{i}"):
                if is_bet[i]:
                    st.markdown(f":green-badge[Paper bet] {r.pick} at {r.pick_price:.2f}")
                winner = st.radio("Winner", [r.fighter_1, r.fighter_2, "draw", "nc"], horizontal=True)
                a, b = st.columns(2)
                c1v = a.number_input(f"Closing price: {r.fighter_1}", value=None, min_value=1.01, format="%.2f")
                c2v = b.number_input(f"Closing price: {r.fighter_2}", value=None, min_value=1.01, format="%.2f")
                st.caption("Closing prices are decimal and optional, but closing-line value needs them.")
                if st.form_submit_button("Save result", type="primary"):
                    out = led.copy()
                    out["winner"] = out.winner.astype("object")
                    out.loc[i, "winner"] = winner
                    out.loc[i, "close_f1_decimal"], out.loc[i, "close_f2_decimal"] = c1v, c2v
                    out[LEDGER_COLS].to_csv(LEDGER, index=False)
                    st.rerun()
        with st.expander("Full ledger as a table (edit closing prices and winners)"):
            st.write("Type the winner's name exactly as shown, or `draw`, or `nc`.")
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
            if st.button("Save table", disabled=bool(bad)):
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
        (k1, k2), (k3, k4) = st.columns(2), st.columns(2)
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
