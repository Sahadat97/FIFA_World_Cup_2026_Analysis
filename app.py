"""
FIFA World Cup 2026 — analysis app.

Match Analysis: the animated momentum replay plus an EA FC-style match
statistics panel (Summary / Possession / Shooting / Passing / Defending /
Events tabs).  Player Analysis: tournament-wide leaderboards from the
FIFA.com stats, with an enhanced finishing-vs-xG view.  Final Prediction:
a gradient boosting model trained on this tournament's own results,
projecting who wins the final.

Run:  streamlit run app.py
"""

import base64
import os
import sys
from pathlib import Path

# pyarrow's bundled mimalloc allocator segfaults on macOS when Streamlit
# serializes dataframes from its script thread — use the system allocator
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from momentum_plot import build_figure as build_momentum_replay, smooth2d

HERE = Path(__file__).parent
LOGO_PATH = HERE / "assets" / "Fifa26logo.svg"

sys.path.insert(0, str(HERE / "scripts"))
from predict_final_winner import (
    build_team_profiles, build_training_set, train_and_evaluate,
    determine_next_matchup_pool, matchup_advance_prob, simulate_bracket,
    all_pairings_of_four,
)

st.set_page_config(page_title="FIFA World Cup 2026", page_icon="⚽", layout="wide")

# ---------------------------------------------------------------- palette
C_HOME = "#2a78d6"
C_AWAY = "#1baf7a"
C_YELLOW = "#eda100"
C_RED = "#e34948"
C_NEUTRAL = "#c3c2b7"
C_GRID = "#e1e0d9"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_SURFACE = "#fcfcfb"
C_DIV_MID = "#f0efec"     # diverging neutral midpoint (light mode)
C_SAVED = "#9ec5f4"

PLOTLY_BASE = dict(
    paper_bgcolor=C_SURFACE,
    plot_bgcolor=C_SURFACE,
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=C_INK2, size=13),
    margin=dict(l=10, r=10, t=30, b=10),
)

HEAT_HOME = [[0.0, "rgba(252,252,251,0)"], [0.4, "#9ec5f4"], [1.0, "#1c5cab"]]
HEAT_AWAY = [[0.0, "rgba(252,252,251,0)"], [0.4, "#9fdcc4"], [1.0, "#0e7a52"]]

# ---------------------------------------------------------------- flags
def _f(iso2: str) -> str:
    return "".join(chr(0x1F1E6 + ord(c) - 65) for c in iso2)

FLAGS = {
    "Algeria": _f("DZ"), "Argentina": _f("AR"), "Australia": _f("AU"), "Austria": _f("AT"),
    "Belgium": _f("BE"), "Bosnia & Herzegovina": _f("BA"), "Brazil": _f("BR"),
    "Cabo Verde": _f("CV"), "Canada": _f("CA"), "Colombia": _f("CO"), "Croatia": _f("HR"),
    "Curaçao": _f("CW"), "Czechia": _f("CZ"), "Côte d'Ivoire": _f("CI"), "DR Congo": _f("CD"),
    "Ecuador": _f("EC"), "Egypt": _f("EG"),
    "England": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
    "France": _f("FR"), "Germany": _f("DE"), "Ghana": _f("GH"), "Haiti": _f("HT"),
    "Iran": _f("IR"), "Iraq": _f("IQ"), "Japan": _f("JP"), "Jordan": _f("JO"),
    "Mexico": _f("MX"), "Morocco": _f("MA"), "Netherlands": _f("NL"), "New Zealand": _f("NZ"),
    "Norway": _f("NO"), "Panama": _f("PA"), "Paraguay": _f("PY"), "Portugal": _f("PT"),
    "Qatar": _f("QA"), "Saudi Arabia": _f("SA"),
    "Scotland": "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
    "Senegal": _f("SN"), "South Africa": _f("ZA"), "South Korea": _f("KR"), "Spain": _f("ES"),
    "Sweden": _f("SE"), "Switzerland": _f("CH"), "Tunisia": _f("TN"), "Türkiye": _f("TR"),
    "USA": _f("US"), "Uruguay": _f("UY"), "Uzbekistan": _f("UZ"),
}

def flag(team) -> str:
    return FLAGS.get(str(team), "")

@st.cache_data
def logo_data_uri() -> str:
    svg_bytes = LOGO_PATH.read_bytes()
    b64 = base64.b64encode(svg_bytes).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


LOGO_HTML = f"""
<div style="display:flex;align-items:center;gap:12px;padding:4px 0 10px 0">
  <img src="{logo_data_uri()}" alt="FIFA World Cup 2026" style="height:56px;width:auto">
  <div style="font-weight:800;font-size:1.05rem;line-height:1.25;color:inherit">
    FIFA World Cup 2026™
  </div>
</div>
"""

st.markdown(f"""
<style>
.statcard {{
  background: {C_SURFACE};
  border: 1px solid rgba(11,11,11,0.10);
  border-radius: 10px;
  padding: 14px 18px;
}}
.srow {{ display:flex; align-items:center; gap:8px; padding:5px 0;
        border-bottom:1px solid {C_GRID}; }}
.srow:last-child {{ border-bottom:none; }}
.sgroup {{ text-align:center; font-weight:700; color:{C_INK}; padding:12px 0 4px 0;
          letter-spacing:.06em; font-size:0.8rem; text-transform:uppercase; }}
.sval {{ flex:0 0 44px; font-weight:700; color:{C_INK}; font-variant-numeric:tabular-nums; }}
.sval.h {{ text-align:right; }}
.sval.a {{ text-align:left; }}
.slabel {{ flex:0 0 40%; text-align:center; color:{C_INK2}; font-size:0.9rem; }}
.sbar {{ flex:1 1 0; min-width:22px; height:8px; background:transparent; position:relative; }}
.sbar > div {{ position:absolute; top:0; height:8px; border-radius:4px; }}
.sbar.h > div {{ right:0; background:{C_HOME}; }}
.sbar.a > div {{ left:0; background:{C_AWAY}; }}
.ringwrap {{ text-align:center; padding:10px 0; }}
.ring {{ width:130px; height:130px; border-radius:50%; margin:0 auto;
        display:flex; align-items:center; justify-content:center; }}
.ring > div {{ width:104px; height:104px; border-radius:50%; background:{C_SURFACE};
              display:flex; align-items:center; justify-content:center;
              font-size:1.6rem; font-weight:700; color:{C_INK}; }}
.ringlabel {{ margin-top:8px; color:{C_MUTED}; font-size:0.78rem; letter-spacing:.08em;
             text-transform:uppercase; }}
.evrow {{ display:flex; align-items:center; gap:14px; padding:9px 0;
         border-bottom:1px solid {C_GRID}; }}
.evrow:last-child {{ border-bottom:none; }}
.evside {{ flex:1; }}
.evside.h {{ text-align:right; }}
.evside.a {{ text-align:left; }}
.evname {{ font-weight:600; color:{C_INK}; }}
.evsub {{ color:{C_MUTED}; font-size:0.8rem; }}
.evmin {{ width:56px; height:56px; border-radius:50%; border:2px solid {C_NEUTRAL};
         display:flex; align-items:center; justify-content:center; flex:0 0 56px;
         font-weight:700; color:{C_INK}; background:{C_SURFACE}; }}
.evicon {{ font-size:1.1rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load_data():
    matches = pd.read_csv("data/matches.csv")
    events = pd.read_csv("data/match_events.csv")
    shots = pd.read_csv("data/match_shots.csv")
    team_stats = pd.read_csv("data/team_match_stats.csv")
    matches["datetime"] = pd.to_datetime(matches["datetime"])
    matches = matches.sort_values("datetime").reset_index(drop=True)
    matches["label"] = (
        matches["datetime"].dt.strftime("%d %b")
        + " · " + matches["home_team"].map(FLAGS).fillna("") + " " + matches["home_team"]
        + " " + matches["home_score"].astype("Int64").astype(str)
        + "–" + matches["away_score"].astype("Int64").astype(str)
        + " " + matches["away_team"].map(FLAGS).fillna("") + " " + matches["away_team"]
    )
    return matches, events, shots, team_stats


@st.cache_data
def load_fifa_stats():
    base = "data/fifa_stats/players"
    frames = {}
    for name in ["adidas_golden_boot", "attacking", "distribution", "defending",
                 "discipline", "goalkeeping", "movement", "physical"]:
        try:
            frames[name] = pd.read_csv(f"{base}/{name}.csv")
        except FileNotFoundError:
            pass
    return frames


@st.cache_data
def momentum_replay_fig(mid: int):
    return build_momentum_replay(mid)


@st.cache_resource
def load_prediction_model():
    """Trains the final-winner classifier once per data version and caches
    the fitted model (st.cache_resource, not cache_data, since it holds a
    non-serializable-friendly sklearn estimator alongside the profile)."""
    profile = build_team_profiles()
    X, y = build_training_set(profile)
    model, metrics = train_and_evaluate(X, y)
    return profile, model, metrics


matches, events, shots, team_stats = load_data()


# ---------------------------------------------------------------- html helpers
def ring_html(pct: float, color: str, label: str) -> str:
    deg = max(0.0, min(1.0, pct / 100)) * 360
    return f"""
    <div class="ringwrap">
      <div class="ring" style="background:conic-gradient({color} {deg}deg, {C_GRID} {deg}deg);">
        <div>{pct:.0f}%</div>
      </div>
      <div class="ringlabel">{label}</div>
    </div>"""


def fmt(v, kind="int"):
    if pd.isna(v):
        return "0"
    if kind == "int":
        return f"{int(v)}"
    if kind == "2f":
        return f"{v:.2f}"
    return str(v)


def stat_rows_html(rows) -> str:
    out = ['<div class="statcard">']
    for r in rows:
        if r[0] == "group":
            out.append(f'<div class="sgroup">{r[1]}</div>')
            continue
        label, hv, av, kind = r
        h = 0 if pd.isna(hv) else float(hv)
        a = 0 if pd.isna(av) else float(av)
        tot = h + a
        hp = 0 if tot == 0 else h / tot * 100
        ap = 0 if tot == 0 else a / tot * 100
        out.append(
            f'<div class="srow">'
            f'<div class="sval h">{fmt(hv, kind)}</div>'
            f'<div class="sbar h"><div style="width:{hp:.0f}%"></div></div>'
            f'<div class="slabel">{label}</div>'
            f'<div class="sbar a"><div style="width:{ap:.0f}%"></div></div>'
            f'<div class="sval a">{fmt(av, kind)}</div>'
            f'</div>'
        )
    out.append("</div>")
    return "".join(out)


def short_name(name) -> str:
    if pd.isna(name):
        return ""
    parts = str(name).split()
    return f"{parts[0][0]}. {' '.join(parts[1:])}" if len(parts) > 1 else str(name)


# ---------------------------------------------------------------- pitch helpers
def half_pitch_shapes():
    line = dict(color=C_NEUTRAL, width=1.5)
    return [
        dict(type="rect", x0=0, y0=0, x1=100, y1=55, line=line),
        dict(type="rect", x0=21.1, y0=0, x1=78.9, y1=16.5, line=line),
        dict(type="rect", x0=36.8, y0=0, x1=63.2, y1=5.5, line=line),
        dict(type="circle", x0=48.9, y0=10, x1=51.1, y1=12.2, line=line, fillcolor=C_NEUTRAL),
        dict(type="path", path="M 38.4,16.5 C 42,22.5 58,22.5 61.6,16.5", line=line),
        dict(type="rect", x0=44.3, y0=-2.2, x1=55.7, y1=0, line=dict(color=C_INK2, width=3)),
    ]


def mini_shot_map(s: pd.DataFrame, is_home: bool, color: str, title: str) -> go.Figure:
    ss = s[s["is_home"] == is_home]
    fig = go.Figure()
    off_t = ss[~ss["shot_type"].isin(["goal", "save"])]
    saved = ss[ss["shot_type"] == "save"]
    goals = ss[ss["shot_type"] == "goal"]
    fig.add_trace(go.Scatter(
        x=off_t["player_y"], y=off_t["player_x"], mode="markers",
        marker=dict(size=7 + off_t["xg"].fillna(0.03) * 25, color=C_NEUTRAL,
                    line=dict(width=1, color=C_SURFACE)),
        hovertemplate="min %{customdata[0]} · %{customdata[1]}<extra></extra>",
        customdata=np.stack([off_t["time_minute"], off_t["shot_type"]], axis=-1) if len(off_t) else None,
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=saved["player_y"], y=saved["player_x"], mode="markers",
        marker=dict(size=8 + saved["xg"].fillna(0.03) * 25, color=color, opacity=0.55,
                    line=dict(width=1.5, color=C_SURFACE)),
        hovertemplate="min %{customdata} · saved<extra></extra>",
        customdata=saved["time_minute"] if len(saved) else None, showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=goals["player_y"], y=goals["player_x"], mode="markers",
        marker=dict(symbol="star", size=12 + goals["xg"].fillna(0.05) * 25, color=color,
                    line=dict(width=1.2, color=C_INK)),
        hovertemplate="min %{customdata} · GOAL<extra></extra>",
        customdata=goals["time_minute"] if len(goals) else None, showlegend=False,
    ))
    fig.update_layout(
        **PLOTLY_BASE, height=330, shapes=half_pitch_shapes(),
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=13, color=C_INK2)),
        xaxis=dict(range=[-3, 103], visible=False),
        yaxis=dict(range=[56, -4], visible=False),
    )
    return fig


def attack_heatmap(s: pd.DataFrame, is_home: bool, scale, title: str) -> go.Figure:
    ss = s[s["is_home"] == is_home]
    z, _, _ = np.histogram2d(
        ss["player_x"], ss["player_y"], bins=[24, 16], range=[[0, 100], [0, 100]],
        weights=1.0 + 3.0 * ss["xg"].fillna(0.05),
    )
    z = smooth2d(z, sigma=1.6)
    line = dict(color=C_NEUTRAL, width=1.2)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=100, y1=100, line=line),
        dict(type="line", x0=0, y0=50, x1=100, y1=50, line=line),
        dict(type="circle", x0=37.1, y0=41.4, x1=62.9, y1=58.6, line=line),
        dict(type="rect", x0=21.1, y0=0, x1=78.9, y1=16.5, line=line),
        dict(type="rect", x0=21.1, y0=83.5, x1=78.9, y1=100, line=line),
    ]
    fig = go.Figure(go.Heatmap(
        z=z, x=np.linspace(3.125, 96.875, 16), y=np.linspace(100/48, 100 - 100/48, 24),
        colorscale=scale, showscale=False, zsmooth="best",
        hovertemplate="attack density %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **PLOTLY_BASE, height=330, shapes=shapes,
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=13, color=C_INK2)),
        xaxis=dict(range=[-2, 102], visible=False),
        yaxis=dict(range=[-2, 102], visible=False),
    )
    return fig


# ================================================================ sidebar
st.sidebar.markdown(LOGO_HTML, unsafe_allow_html=True)
section = st.sidebar.radio(
    "Section", ["Match Analysis", "Player Analysis", "Final Prediction"], label_visibility="collapsed"
)


# ================================================================ MATCH ANALYSIS
if section == "Match Analysis":
    stage = st.sidebar.selectbox("Stage", ["All"] + matches["stage"].dropna().unique().tolist())
    pool = matches if stage == "All" else matches[matches["stage"] == stage]
    label = st.sidebar.selectbox("Match", pool["label"].tolist())
    match = pool[pool["label"] == label].iloc[0]
    mid = int(match["match_id"])
    home, away = match["home_team"], match["away_team"]

    ev = events[(events["match_id"] == mid) & events["time_minute"].between(0, 130)].copy()
    ev["is_home"] = ev["is_home"].astype(bool)
    ev = ev.sort_values(["time_minute", "added_time"], na_position="first")
    s = shots[(shots["match_id"] == mid) & (shots["situation"] != "shootout")].copy()
    s = s[s["time_minute"].notna()]
    ts = team_stats[team_stats["match_id"] == mid]
    h = ts[ts["is_home"]].iloc[0]
    a = ts[~ts["is_home"]].iloc[0]

    # ---------- header
    score = f"{int(match['home_score'])} – {int(match['away_score'])}"
    pens = ""
    if match["has_penalty_shootout"] and pd.notna(match["home_score_penalties"]):
        pens = f" ({int(match['home_score_penalties'])}–{int(match['away_score_penalties'])} pens)"
    sub = f"{match['stage']}" + (f" · {match['group']}" if pd.notna(match["group"]) else "")
    st.markdown(
        f"<div style='text-align:center'>"
        f"<div style='font-size:0.9rem;color:{C_MUTED}'>{sub} · {match['stadium']} · "
        f"{match['datetime'].strftime('%d %b %Y')}</div>"
        f"<div style='font-size:2.2rem;font-weight:700;color:{C_INK}'>"
        f"<span>{flag(home)}</span> <span style='color:{C_HOME}'>{home}</span>"
        f"<span style='margin:0 1.2rem'>{score}{pens}</span>"
        f"<span>{flag(away)}</span> <span style='color:{C_AWAY}'>{away}</span></div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # ---------- momentum replay
    st.subheader("Match momentum replay")
    st.caption(
        "Press ▶ Play (or drag the minute slider): the shaded window sweeps the match and the pitch "
        f"shows where the attacking threat was, {home} attacks the right goal, {away} the left."
    )
    st.plotly_chart(momentum_replay_fig(mid), use_container_width=True)

    # ---------- EA-style stats tabs
    st.subheader("Match statistics")
    tab_sum, tab_pos, tab_sho, tab_pas, tab_def, tab_evt = st.tabs(
        ["Summary", "Possession", "Shooting", "Passing", "Defending", "Events"]
    )

    with tab_sum:
        left, mid_c, right = st.columns([0.8, 2.8, 0.8])
        with left:
            st.markdown(ring_html(h["possession_pct"], C_HOME, f"{home} possession"), unsafe_allow_html=True)
        with right:
            st.markdown(ring_html(a["possession_pct"], C_AWAY, f"{away} possession"), unsafe_allow_html=True)
        with mid_c:
            st.markdown(stat_rows_html([
                ("group", "Overall"),
                ("Expected Goals (xG)", h["expected_goals"], a["expected_goals"], "2f"),
                ("Total Shots", h["shots_total"], a["shots_total"], "int"),
                ("Shots On Target", h["shots_on_target"], a["shots_on_target"], "int"),
                ("Big Chances", h["big_chances"], a["big_chances"], "int"),
                ("Corners", h["corners"], a["corners"], "int"),
                ("Passes", h["passes_total"], a["passes_total"], "int"),
                ("Pass Accuracy %", h["passes_accurate"] / max(h["passes_total"], 1) * 100,
                 a["passes_accurate"] / max(a["passes_total"], 1) * 100, "int"),
                ("Tackles", h["tackles"], a["tackles"], "int"),
                ("Saves", h["saves"], a["saves"], "int"),
                ("Fouls", h["fouls"], a["fouls"], "int"),
                ("Offsides", h["offsides"], a["offsides"], "int"),
            ]), unsafe_allow_html=True)

    with tab_pos:
        left, mid_c, right = st.columns([0.8, 2.8, 0.8])
        with left:
            st.markdown(ring_html(h["possession_pct"], C_HOME, "Overall possession"), unsafe_allow_html=True)
        with right:
            st.markdown(ring_html(a["possession_pct"], C_AWAY, "Overall possession"), unsafe_allow_html=True)
        with mid_c:
            st.markdown(stat_rows_html([
                ("group", "Ball control"),
                ("Possession %", h["possession_pct"], a["possession_pct"], "int"),
                ("Passes", h["passes_total"], a["passes_total"], "int"),
                ("Passes Completed", h["passes_accurate"], a["passes_accurate"], "int"),
                ("Passes In Final Third", h["passes_final_third"], a["passes_final_third"], "int"),
                ("Dribbles Completed", h["dribbles_completed"], a["dribbles_completed"], "int"),
                ("Dribbles Attempted", h["dribbles_total"], a["dribbles_total"], "int"),
                ("Corners", h["corners"], a["corners"], "int"),
                ("Throw-Ins", h["throw_ins"], a["throw_ins"], "int"),
                ("Offsides", h["offsides"], a["offsides"], "int"),
            ]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(attack_heatmap(s, True, HEAT_HOME, f"{home} — attack zones (↑ attacking)"),
                            use_container_width=True)
        with c2:
            st.plotly_chart(attack_heatmap(s, False, HEAT_AWAY, f"{away} — attack zones (↑ attacking)"),
                            use_container_width=True)

    with tab_sho:
        sh_h = s[s["is_home"]]
        sh_a = s[~s["is_home"].astype(bool)]
        acc_h = h["shots_on_target"] / max(h["shots_total"], 1) * 100
        acc_a = a["shots_on_target"] / max(a["shots_total"], 1) * 100
        left, mid_c, right = st.columns([0.8, 2.8, 0.8])
        with left:
            st.markdown(ring_html(acc_h, C_HOME, "Shot accuracy"), unsafe_allow_html=True)
        with right:
            st.markdown(ring_html(acc_a, C_AWAY, "Shot accuracy"), unsafe_allow_html=True)

        def count(df, col, val):
            return int((df[col] == val).sum())

        with mid_c:
            st.markdown(stat_rows_html([
                ("group", "Overall shooting"),
                ("Total Shots", h["shots_total"], a["shots_total"], "int"),
                ("On Target", h["shots_on_target"], a["shots_on_target"], "int"),
                ("Off Target", h["shots_off_target"], a["shots_off_target"], "int"),
                ("Blocked", h["shots_blocked"], a["shots_blocked"], "int"),
                ("Hit Woodwork", h["hit_woodwork"], a["hit_woodwork"], "int"),
                ("Expected Goals (xG)", h["expected_goals"], a["expected_goals"], "2f"),
                ("Big Chances", h["big_chances"], a["big_chances"], "int"),
                ("group", "Distance"),
                ("Inside The Box", h["shots_inside_box"], a["shots_inside_box"], "int"),
                ("Outside The Box", h["shots_outside_box"], a["shots_outside_box"], "int"),
                ("group", "Body part"),
                ("Right Foot", count(sh_h, "body_part", "right-foot"), count(sh_a, "body_part", "right-foot"), "int"),
                ("Left Foot", count(sh_h, "body_part", "left-foot"), count(sh_a, "body_part", "left-foot"), "int"),
                ("Header", count(sh_h, "body_part", "head"), count(sh_a, "body_part", "head"), "int"),
                ("group", "Situation"),
                ("Open Play (Assisted)", count(sh_h, "situation", "assisted"), count(sh_a, "situation", "assisted"), "int"),
                ("From Corner", count(sh_h, "situation", "corner"), count(sh_a, "situation", "corner"), "int"),
                ("Set Piece / Free Kick",
                 count(sh_h, "situation", "set-piece") + count(sh_h, "situation", "free-kick"),
                 count(sh_a, "situation", "set-piece") + count(sh_a, "situation", "free-kick"), "int"),
                ("Fast Break", count(sh_h, "situation", "fast-break"), count(sh_a, "situation", "fast-break"), "int"),
                ("Penalty", count(sh_h, "situation", "penalty"), count(sh_a, "situation", "penalty"), "int"),
            ]), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(mini_shot_map(s, True, C_HOME, f"{home} — all shots"), use_container_width=True)
        with c2:
            st.plotly_chart(mini_shot_map(s, False, C_AWAY, f"{away} — all shots"), use_container_width=True)
        st.caption("★ goals · solid dots on target (saved) · gray off target/blocked · size = xG · goal at the top.")

    with tab_pas:
        pa_h = h["passes_accurate"] / max(h["passes_total"], 1) * 100
        pa_a = a["passes_accurate"] / max(a["passes_total"], 1) * 100
        left, mid_c, right = st.columns([0.8, 2.8, 0.8])
        with left:
            st.markdown(ring_html(pa_h, C_HOME, "Pass accuracy"), unsafe_allow_html=True)
        with right:
            st.markdown(ring_html(pa_a, C_AWAY, "Pass accuracy"), unsafe_allow_html=True)
        with mid_c:
            st.markdown(stat_rows_html([
                ("group", "Overall passing"),
                ("Total Passes", h["passes_total"], a["passes_total"], "int"),
                ("Completed", h["passes_accurate"], a["passes_accurate"], "int"),
                ("Passes In Final Third", h["passes_final_third"], a["passes_final_third"], "int"),
                ("group", "Pass type"),
                ("Long Balls", h["long_balls_total"], a["long_balls_total"], "int"),
                ("Long Balls Accurate", h["long_balls_accurate"], a["long_balls_accurate"], "int"),
                ("Crosses", h["crosses_total"], a["crosses_total"], "int"),
                ("Crosses Accurate", h["crosses_accurate"], a["crosses_accurate"], "int"),
                ("group", "Restarts"),
                ("Throw-Ins", h["throw_ins"], a["throw_ins"], "int"),
                ("Goal Kicks", h["goal_kicks"], a["goal_kicks"], "int"),
                ("Free Kicks", h["free_kicks"], a["free_kicks"], "int"),
            ]), unsafe_allow_html=True)

    with tab_def:
        gd_h = h["ground_duels_won"] / max(h["ground_duels_total"], 1) * 100
        gd_a = a["ground_duels_won"] / max(a["ground_duels_total"], 1) * 100
        cards_ev = ev[ev["incident_type"] == "card"]
        reds_h = int(((cards_ev["incident_class"] == "red") & cards_ev["is_home"]).sum())
        reds_a = int(((cards_ev["incident_class"] == "red") & ~cards_ev["is_home"]).sum())
        left, mid_c, right = st.columns([0.8, 2.8, 0.8])
        with left:
            st.markdown(ring_html(gd_h, C_HOME, "Ground duels won"), unsafe_allow_html=True)
        with right:
            st.markdown(ring_html(gd_a, C_AWAY, "Ground duels won"), unsafe_allow_html=True)
        with mid_c:
            st.markdown(stat_rows_html([
                ("group", "Overall defending"),
                ("Tackles", h["tackles"], a["tackles"], "int"),
                ("Interceptions", h["interceptions"], a["interceptions"], "int"),
                ("Clearances", h["clearances"], a["clearances"], "int"),
                ("Saves", h["saves"], a["saves"], "int"),
                ("group", "Duels"),
                ("Ground Duels Won", h["ground_duels_won"], a["ground_duels_won"], "int"),
                ("Ground Duels Total", h["ground_duels_total"], a["ground_duels_total"], "int"),
                ("Aerial Duels Won", h["aerial_duels_won"], a["aerial_duels_won"], "int"),
                ("Aerial Duels Total", h["aerial_duels_total"], a["aerial_duels_total"], "int"),
                ("group", "Infractions"),
                ("Fouls Committed", h["fouls"], a["fouls"], "int"),
                ("Yellow Cards", h["yellow_cards"], a["yellow_cards"], "int"),
                ("Red Cards", reds_h, reds_a, "int"),
            ]), unsafe_allow_html=True)

    with tab_evt:
        icon_for = {
            "goal": "⚽", "inGamePenalty": "⚽", "card": "🟨",
            "substitution": "🔁", "varDecision": "📺",
        }
        show = ev[ev["incident_type"].isin(icon_for)].copy()
        rows = ['<div class="statcard">']
        for _, r in show.iterrows():
            icon = icon_for[r["incident_type"]]
            if r["incident_type"] == "card" and r["incident_class"] == "red":
                icon = "🟥"
            if r["incident_type"] == "goal" and r["incident_class"] == "ownGoal":
                icon = "⚽ (OG)"
            minute = f"{int(r['time_minute'])}'"
            if pd.notna(r["added_time"]) and r["added_time"]:
                minute = f"{int(r['time_minute'])}+{int(r['added_time'])}'"
            if r["incident_type"] == "substitution":
                main = f"{short_name(r['player_in'])} ⇄ {short_name(r['player_out'])}"
                subtext = "Substitution"
            elif r["incident_type"] in ("goal", "inGamePenalty"):
                main = short_name(r["player"])
                subtext = ""
                if pd.notna(r["assist_player"]):
                    subtext = f"assist {short_name(r['assist_player'])}"
                if pd.notna(r["home_score"]):
                    subtext += f"{' · ' if subtext else ''}{int(r['home_score'])}–{int(r['away_score'])}"
            else:
                main = short_name(r["player"]) if pd.notna(r["player"]) else str(r["incident_class"])
                subtext = str(r["incident_class"])
            side_h = f'<div class="evname">{main} <span class="evicon">{icon}</span></div><div class="evsub">{subtext}</div>' if r["is_home"] else ""
            side_a = f'<div class="evname"><span class="evicon">{icon}</span> {main}</div><div class="evsub">{subtext}</div>' if not r["is_home"] else ""
            rows.append(
                f'<div class="evrow">'
                f'<div class="evside h">{side_h}</div>'
                f'<div class="evmin">{minute}</div>'
                f'<div class="evside a">{side_a}</div>'
                f'</div>'
            )
        rows.append("</div>")
        st.markdown("".join(rows), unsafe_allow_html=True)
        st.caption(f"{flag(home)} {home} events on the left · {flag(away)} {away} on the right · "
                   "⚽ goal · 🟨🟥 cards · 🔁 substitution · 📺 VAR")


# ================================================================ PLAYER ANALYSIS
elif section == "Player Analysis":
    st.title("Player analysis")
    fifa = load_fifa_stats()
    gb = fifa.get("adidas_golden_boot")
    att = fifa.get("attacking")
    if gb is None or att is None:
        st.error("FIFA stats CSVs missing — run fetch_fifa_stats.py first.")
        st.stop()

    played = gb[gb["goals"].notna()].copy()

    # KPI row
    top_scorer = played.sort_values("rank", na_position="last").iloc[0]
    top_assists = played.sort_values("assists", ascending=False).iloc[0]
    att_p = att[att["xg"].notna()].copy()
    top_xg = att_p.sort_values("xg", ascending=False).iloc[0]
    k1, k2, k3 = st.columns(3)
    k1.metric("Golden Boot leader", top_scorer["name"],
              f"{int(top_scorer['goals'])} goals · {top_scorer['team']}", delta_color="off")
    k2.metric("Most assists", top_assists["name"],
              f"{int(top_assists['assists'])} assists · {top_assists['team']}", delta_color="off")
    k3.metric("Highest xG", top_xg["name"],
              f"{top_xg['xg']:.2f} xG · {top_xg['team']}", delta_color="off")
    st.markdown("")

    colA, colB = st.columns(2)

    # shots on target -> goal bar
    with colA:
        st.subheader("Shots on target: who finishes?")
        st.caption("Each bar = attempts on target; the solid segment is the share that became goals.")
        f = att_p.copy()
        f["on_target"] = pd.to_numeric(f["attempt_at_goal_on_target"], errors="coerce")
        f = f.drop(columns=["goals"], errors="ignore").merge(
            played[["person_id", "goals"]], on="person_id", how="left")
        f["goals"] = pd.to_numeric(f["goals"], errors="coerce")
        f = f[f["on_target"].notna() & f["goals"].notna()]
        f = f.sort_values("on_target", ascending=False).head(12)
        f["saved"] = (f["on_target"] - f["goals"]).clip(lower=0)
        f["who"] = f["name"] + "  ·  " + f["team_abbreviation"].fillna("")

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            y=f["who"], x=f["goals"], orientation="h", name="Goals",
            marker=dict(color=C_HOME, line=dict(color=C_SURFACE, width=2)),
            text=f["goals"].astype(int), textposition="inside",
        ))
        fig5.add_trace(go.Bar(
            y=f["who"], x=f["saved"], orientation="h", name="On target, saved",
            marker=dict(color=C_SAVED, line=dict(color=C_SURFACE, width=2)),
        ))
        fig5.update_layout(
            **PLOTLY_BASE, height=430, barmode="stack",
            xaxis=dict(title="attempts on target", showgrid=True, gridcolor=C_GRID, color=C_MUTED),
            yaxis=dict(autorange="reversed", color=C_INK2, tickfont=dict(size=12)),
            legend=dict(orientation="h", y=1.1, x=0),
        )
        st.plotly_chart(fig5, use_container_width=True)

    #enhanced finishing vs chance quality
    with colB:
        st.subheader("Finishing vs chance quality")
        st.caption(
            "Marker color = goals − xG (red running hot, blue cold, gray = par) · "
            "marker size = shots on target · dashed line = scoring exactly to xG."
        )
        min_ot = st.select_slider("Minimum shots on target", options=[0, 2, 4, 6, 8], value=2)

        sc = att_p.copy()
        sc = sc.drop(columns=["goals"], errors="ignore").merge(
            played[["person_id", "goals"]].rename(columns={"goals": "goals_n"}),
            on="person_id", how="left")
        sc["goals_n"] = pd.to_numeric(sc["goals_n"], errors="coerce")
        sc["on_t"] = pd.to_numeric(sc["attempt_at_goal_on_target"], errors="coerce").fillna(0)
        sc = sc[(sc["xg"].notna()) & (sc["goals_n"].notna())
                & ((sc["xg"] > 0.5) | (sc["goals_n"] > 0)) & (sc["on_t"] >= min_ot)]
        sc["delta"] = sc["goals_n"] - sc["xg"]
        dmax = max(sc["delta"].abs().max(), 0.5)
        lim = max(sc["xg"].max(), sc["goals_n"].max()) * 1.1

        fig6 = go.Figure()
        # polarity regions
        fig6.add_shape(type="path", path=f"M 0,0 L {lim},{lim} L 0,{lim} Z",
                       fillcolor="rgba(227,73,72,0.045)", line_width=0, layer="below")
        fig6.add_shape(type="path", path=f"M 0,0 L {lim},{lim} L {lim},0 Z",
                       fillcolor="rgba(42,120,214,0.045)", line_width=0, layer="below")
        fig6.add_annotation(x=lim * 0.13, y=lim * 0.93, text="finishing above xG",
                            showarrow=False, font=dict(size=11, color=C_MUTED))
        fig6.add_annotation(x=lim * 0.85, y=lim * 0.07, text="finishing below xG",
                            showarrow=False, font=dict(size=11, color=C_MUTED))
        fig6.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines",
                                  line=dict(color=C_NEUTRAL, dash="dot", width=1.5),
                                  hoverinfo="skip", showlegend=False))
        fig6.add_trace(go.Scatter(
            x=sc["xg"], y=sc["goals_n"], mode="markers",
            marker=dict(
                size=(7 + sc["on_t"] * 1.1).clip(upper=26),
                color=sc["delta"], cmin=-dmax, cmax=dmax,
                colorscale=[[0.0, C_HOME], [0.5, C_DIV_MID], [1.0, C_RED]],
                line=dict(width=1, color=C_NEUTRAL),
                colorbar=dict(title=dict(text="goals − xG", font=dict(size=11)),
                              thickness=12, len=0.65, tickfont=dict(size=10)),
            ),
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})"
                          "<br>goals %{y} · xG %{x:.2f} · Δ %{customdata[2]:+.2f}"
                          "<br>shots on target %{customdata[3]:.0f}<extra></extra>",
            customdata=np.stack([sc["name"], sc["team"].fillna(""), sc["delta"], sc["on_t"]], axis=-1),
            showlegend=False,
        ))
        labelled = pd.concat([
            sc.sort_values("delta", ascending=False).head(3),   # hottest
            sc.sort_values("delta").head(2),                    # coldest
            sc.sort_values("goals_n", ascending=False).head(2), # top scorers
        ]).drop_duplicates(subset="person_id")
        positions = ["top center", "bottom center", "top left", "bottom right",
                     "top right", "middle left", "bottom left"]
        fig6.add_trace(go.Scatter(
            x=labelled["xg"], y=labelled["goals_n"], mode="text",
            text=labelled["name"].str.split().str[-1],
            textposition=positions[: len(labelled)],
            textfont=dict(size=11, color=C_INK2), hoverinfo="skip", showlegend=False,
        ))
        fig6.update_layout(
            **PLOTLY_BASE, height=430,
            xaxis=dict(title="expected goals (xG)", showgrid=True, gridcolor=C_GRID,
                       zeroline=False, color=C_MUTED, range=[-0.2, lim]),
            yaxis=dict(title="goals", showgrid=True, gridcolor=C_GRID,
                       zeroline=False, color=C_MUTED, range=[-0.3, lim]),
        )
        st.plotly_chart(fig6, use_container_width=True)

    # category explorer
    st.subheader("Leaderboard explorer")
    cat_names = {
        "adidas_golden_boot": "adidas Golden Boot", "attacking": "Attacking",
        "distribution": "Distribution", "defending": "Defending", "discipline": "Discipline",
        "goalkeeping": "Goalkeeping", "movement": "Movement", "physical": "Physical",
    }
    c1, c2, c3 = st.columns([1, 1, 1])
    cat = c1.selectbox("Category", list(cat_names), format_func=cat_names.get)
    df_cat = fifa[cat]
    id_cols = {"person_id", "name", "team_id", "team", "team_abbreviation", "position", "rank"}
    metric_cols = [c for c in df_cat.columns if c not in id_cols and df_cat[c].notna().any()
                   and pd.to_numeric(df_cat[c], errors="coerce").notna().any()]
    metric = c2.selectbox("Metric", metric_cols)
    top_n = c3.slider("Top N", 5, 30, 15)

    d = df_cat.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    d = d[d[metric].notna()].sort_values(metric, ascending=False).head(top_n)
    d["who"] = d["name"] + "  ·  " + d["team_abbreviation"].fillna("")

    fig7 = go.Figure(go.Bar(
        y=d["who"], x=d[metric], orientation="h",
        marker=dict(color=C_HOME, line=dict(color=C_SURFACE, width=2)),
        text=[f"{v:g}" for v in d[metric]], textposition="outside",
        hovertemplate="%{y}: %{x:g}<extra></extra>",
    ))
    fig7.update_layout(
        **PLOTLY_BASE, height=max(320, 26 * len(d) + 80),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, color=C_MUTED,
                   title=metric.replace("_", " ")),
        yaxis=dict(autorange="reversed", color=C_INK2, tickfont=dict(size=12)),
        showlegend=False,
    )
    st.plotly_chart(fig7, use_container_width=True)

    with st.expander("Full table"):
        st.dataframe(df_cat, use_container_width=True, hide_index=True)


# ================================================================ FINAL PREDICTION
else:
    st.title("Final prediction")
    st.caption(
        "A gradient boosting model trained on every completed match this tournament, using "
        "team box-score averages, results-derived form, and FIFA.com's cumulative team "
        "leaderboards as features. See scripts/predict_final_winner.py."
    )

    profile, model, metrics = load_prediction_model()

    k1, k2, k3 = st.columns(3)
    k1.metric("Model accuracy (5-fold CV)", f"{metrics['accuracy']:.1%}",
              f"baseline {metrics['baseline_accuracy']:.1%}", delta_color="off")
    k2.metric("Log-loss (5-fold CV)", f"{metrics['log_loss']:.3f}",
              "lower is better", delta_color="off")
    k3.metric("Matches analyzed", metrics["n_matches"])
    st.markdown("")

    stage, pool = determine_next_matchup_pool(matches)
    # color follows the team, not its rank in any given chart — assign once
    # so a team keeps the same color across the summary and every scenario
    palette = [C_HOME, C_AWAY, C_YELLOW, C_RED]
    team_colors = {t: palette[i % len(palette)] for i, t in enumerate(pool)} if pool else {}

    def prob_bar_chart(probs: dict) -> go.Figure:
        order = sorted(probs, key=probs.get, reverse=True)
        fig = go.Figure(go.Bar(
            y=[f"{flag(t)} {t}" for t in order],
            x=[probs[t] * 100 for t in order],
            orientation="h",
            marker=dict(color=[team_colors[t] for t in order], line=dict(color=C_SURFACE, width=2)),
            text=[f"{probs[t]:.1%}" for t in order], textposition="outside",
            hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            **PLOTLY_BASE, height=90 + 60 * len(order), showlegend=False,
            xaxis=dict(title="chance of winning the final (%)", showgrid=True, gridcolor=C_GRID,
                       color=C_MUTED, range=[0, max(probs.values()) * 130]),
            yaxis=dict(autorange="reversed", color=C_INK2, tickfont=dict(size=13)),
        )
        return fig

    if stage is None:
        st.info("The knockout stage hasn't started yet, so there's nothing to predict.")
    elif stage == "champion":
        st.success(f"The tournament is decided: **{flag(pool)} {pool}** won the FIFA World Cup 2026.")
    elif len(pool) == 2:
        st.subheader(f"Final: {flag(pool[0])} {pool[0]} vs {flag(pool[1])} {pool[1]}")
        p_a = matchup_advance_prob(model, profile, pool[0], pool[1])
        st.plotly_chart(prob_bar_chart({pool[0]: p_a, pool[1]: 1 - p_a}), use_container_width=True)
    elif len(pool) == 4:
        st.subheader("Teams advancing to the " + stage)
        st.caption(
            "The actual semifinal pairing isn't in the pulled data yet, so this simulates all 3 "
            "possible ways to pair the remaining 4 teams (20,000 Monte Carlo trials each) and "
            "averages the result."
        )
        pairings = all_pairings_of_four(pool)
        totals = {t: 0.0 for t in pool}
        scenario_results = []
        for pairing in pairings:
            result = simulate_bracket(model, profile, pairing)
            scenario_results.append((pairing, result))
            for t in pool:
                totals[t] += result[t]
        avg = {t: totals[t] / len(pairings) for t in pool}

        st.plotly_chart(prob_bar_chart(avg), use_container_width=True)

        with st.expander("Breakdown by semifinal pairing"):
            for pairing, result in scenario_results:
                label = (f"{flag(pairing[0][0])} {pairing[0][0]} vs {flag(pairing[0][1])} {pairing[0][1]}"
                         f"   ·   {flag(pairing[1][0])} {pairing[1][0]} vs {flag(pairing[1][1])} {pairing[1][1]}")
                st.markdown(f"**{label}**")
                st.plotly_chart(prob_bar_chart(result), use_container_width=True)
    else:
        st.info(f"{len(pool)} teams are advancing to the {stage} — outside the 2-team/4-team "
                "matchup sizes this page projects a final winner for.")

    with st.expander("Team strength profile (model input features)"):
        st.dataframe(profile, use_container_width=True)
