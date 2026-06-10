"""
2026 FIFA World Cup Simulator — Streamlit dashboard
"""

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WC 2026 Simulator",
    page_icon="soccer",
    layout="wide",
)

# ── Imports (after page config) ───────────────────────────────────────────────
import config
from data_layer import load_results, compute_weights, get_elo_dict, all_teams, GROUPS
from dixon_coles import DixonColesModel
from daily_bets import fetch_odds, analyse_match, normalise
from charts import plot_title_odds, plot_advancement_heatmap

# ── Model caching ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Fitting Dixon-Coles model (first load only)...")
def get_model():
    df = load_results()
    elo_dict = {}
    try:
        elo_dict = get_elo_dict()
    except Exception:
        pass
    wc_teams = set(all_teams())
    df_fit = df[df["home"].isin(wc_teams) & df["away"].isin(wc_teams)].reset_index(drop=True)
    weights = compute_weights(df_fit)
    model = DixonColesModel(elo_dict=elo_dict)
    model.fit(df_fit, weights)
    missing = [t for t in all_teams() if t not in model.attack]
    for t in missing:
        model.attack[t] = 0.0
        model.defense[t] = 0.0
    if missing:
        model._teams = list(set(model._teams) | set(missing))
    return model, elo_dict


@st.cache_data(show_spinner="Running simulations...", ttl=3600)
def get_sim_probs(n_sims):
    from engine import run_simulations
    model, elo_dict = get_model()
    return run_simulations(model, elo_dict, n=n_sims, seed=config.RANDOM_SEED)


@st.cache_data(show_spinner="Fetching odds...", ttl=300)
def get_odds():
    if not config.ODDS_API_KEY:
        return []
    try:
        return fetch_odds(config.ODDS_API_KEY)
    except Exception as e:
        st.warning(f"Could not fetch odds: {e}")
        return []


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("WC 2026 Simulator")
st.sidebar.markdown("Dixon-Coles Monte Carlo")
page = st.sidebar.radio("View", ["Today's Value Bets", "Tournament Odds", "Group Probabilities"])

n_sims = st.sidebar.select_slider("Simulations", [1000, 5000, 10000, 50000], value=10000)
min_edge = st.sidebar.slider("Min edge % (value bets)", 0, 30, 5)
bookmaker = st.sidebar.selectbox("Bookmaker", ["williamhill", "paddypower", "betway", "pinnacle", "draftkings", "betmgm", "best available"], index=0)

# ── Page: Today's Value Bets ──────────────────────────────────────────────────

if page == "Today's Value Bets":
    st.title("Today's Value Bets")

    date_input = st.date_input("Date", value=date.today())
    days_ahead = st.number_input("Days ahead", min_value=1, max_value=7, value=1)

    model, elo_dict = get_model()
    events = get_odds()

    target = date_input
    end = target + timedelta(days=int(days_ahead) - 1)

    day_events = []
    for ev in events:
        try:
            ev_date = datetime.fromisoformat(
                ev["commence_time"].replace("Z", "+00:00")
            ).date()
        except Exception:
            continue
        if target <= ev_date <= end:
            day_events.append(ev)

    if not day_events:
        st.info("No matches with odds found for this date range. Odds are usually posted 1-3 days before kickoff.")
    else:
        value_count = 0
        for ev in sorted(day_events, key=lambda e: e.get("commence_time", "")):
            result = analyse_match(ev, model, min_edge, bookmaker)
            if result is None:
                continue

            home, away = result["home"], result["away"]
            try:
                dt = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
                dt_str = dt.strftime("%b %d  %H:%M UTC")
            except Exception:
                dt_str = ev.get("commence_time", "")

            has_value = len(result["value_bets"]) > 0
            label = f"{'*** ' if has_value else ''}{home} vs {away}  |  {dt_str}"

            with st.expander(label, expanded=has_value):
                rows = []
                for b in result["bets"]:
                    rows.append({
                        "Outcome": b["team"],
                        "Model": f"{b['model_prob']:.1%}",
                        "Odds (dec)": f"{b['decimal_odds']:.2f}",
                        "Implied": f"{b['implied_prob']:.1%}",
                        "Edge": f"{b['edge_pct']:+.1f}%",
                        "Book": b["bookmaker"],
                        "Value": "YES" if b["edge_pct"] >= min_edge else "",
                    })
                df_bets = pd.DataFrame(rows)

                def highlight_value(row):
                    if row["Value"] == "YES":
                        return ["background-color: #d4edda"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    df_bets.style.apply(highlight_value, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )
                value_count += len(result["value_bets"])

        if value_count == 0:
            st.info(f"No value bets found above {min_edge}% edge. Try lowering the threshold in the sidebar.")
        else:
            st.success(f"{value_count} value bet(s) flagged (green rows).")

# ── Page: Tournament Odds ─────────────────────────────────────────────────────

elif page == "Tournament Odds":
    st.title("Tournament Win Probabilities")
    st.caption(f"Dixon-Coles model, Elo-blended, {n_sims:,} Monte Carlo simulations")

    probs = get_sim_probs(n_sims)

    # Chart
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    top15 = probs.nlargest(15, "champion").sort_values("champion")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.85, len(top15)))
    bars = ax.barh(top15["team"], top15["champion"] * 100, color=colors, edgecolor="white")
    for bar, val in zip(bars, top15["champion"] * 100):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Championship probability (%)")
    ax.set_title("Title Odds — Top 15")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Full table
    st.subheader("All Teams")
    display = probs.copy()
    for col in ["group_advance", "r16", "qf", "sf", "final", "champion"]:
        display[col] = display[col].map(lambda x: f"{x:.1%}")
    display = display.rename(columns={
        "team": "Team", "group_advance": "Advance", "r16": "R16",
        "qf": "QF", "sf": "SF", "final": "Final", "champion": "Champion"
    })
    st.dataframe(display, use_container_width=True, hide_index=True)

# ── Page: Group Probabilities ─────────────────────────────────────────────────

elif page == "Group Probabilities":
    st.title("Group Stage Probabilities")
    st.caption(f"{n_sims:,} simulations")

    probs = get_sim_probs(n_sims)
    prob_dict = dict(zip(probs["team"], probs["group_advance"]))

    cols = st.columns(3)
    for i, (group_letter, teams) in enumerate(sorted(GROUPS.items())):
        col = cols[i % 3]
        with col:
            st.subheader(f"Group {group_letter}")
            rows = []
            for t in teams:
                rows.append({
                    "Team": t,
                    "P(Advance)": f"{prob_dict.get(t, 0):.1%}",
                })
            df_g = pd.DataFrame(rows).sort_values("P(Advance)", ascending=False)
            st.dataframe(df_g, use_container_width=True, hide_index=True)
