"""
Animated match-momentum visualization (standalone, not yet wired into the app).

Layout (like the reference picture):
  top    — full-pitch heatmap, white -> red, of xG-weighted shot activity
  bottom — per-minute momentum bars (home above the line, away below) with
           goal / card / substitution markers

Behavior (like the reference video):
  a dark window slides across the match timeline; the pitch heatmap shows
  only the activity inside that window, and the title tracks the range.
  Press ▶ in the output, or drag the minute slider.

Usage:
    python momentum_plot.py            # defaults to match_id 1
    python momentum_plot.py 30         # any match_id from data/matches.csv

Writes momentum_animated.html next to this script.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent

# palette (same reference dataviz palette as app.py)
C_HOME = "#2a78d6"
C_AWAY = "#1baf7a"
C_YELLOW = "#eda100"
C_RED = "#e34948"
C_NEUTRAL = "#c3c2b7"
C_INK = "#0b0b0b"
C_INK2 = "#52514e"
C_MUTED = "#898781"
C_SURFACE = "#fcfcfb"
HEAT_SCALE = [
    [0.0, "rgba(252,252,251,0)"], [0.15, "#f6e3e0"], [0.35, "#efb7b0"],
    [0.6, "#e2766c"], [0.8, "#d03b3b"], [1.0, "#a81d1d"],
]

WINDOW = 15      # minutes covered by the sliding band on the momentum strip
STEP = 2         # minutes the window advances per frame
GRID_X, GRID_Y = 60, 40   # heatmap cells (fine grid, ~1.7% of pitch length per cell)
SIGMA_CELLS = 2.4         # spatial smoothing radius, in cells
TIME_TAU = WINDOW / 2.2   # soft time falloff: shots near the window still glow faintly


def load_match(match_id: int):
    matches = pd.read_csv(HERE / "data/matches.csv")
    events = pd.read_csv(HERE / "data/match_events.csv")
    shots = pd.read_csv(HERE / "data/match_shots.csv")
    match = matches[matches["match_id"] == match_id].iloc[0]
    ev = events[(events["match_id"] == match_id) & events["time_minute"].between(0, 130)]
    s = shots[(shots["match_id"] == match_id) & (shots["situation"] != "shootout")].copy()
    s = s[s["time_minute"].notna()]
    return match, ev.sort_values("time_minute"), s


def pitch_coords(s: pd.DataFrame) -> pd.DataFrame:
    """Map shots onto one full pitch: home attacks the right goal (x=100),
    away attacks the left goal (x=0). player_x is distance from the
    attacked goal line, player_y is pitch width.

    Besides the strike point itself, each shot contributes extra real
    datapoints so the heatmap has more to draw on:
      - corner-situation shots also light up the corner flag the ball
        came from (a known, real location);
      - every shot lights up the goal mouth it was aimed at, weighted by
        xGOT for on-target shots (the ball demonstrably travelled there).
    """
    s = s.copy()
    s["px"] = np.where(s["is_home"], 100 - s["player_x"], s["player_x"])
    s["py"] = np.where(s["is_home"], s["player_y"], 100 - s["player_y"])
    s["w"] = 1.0 + 3.0 * s["xg"].fillna(0.05) + np.where(s["shot_type"] == "goal", 2.0, 0.0)

    points = [s[["px", "py", "w", "time_minute"]]]

    corners = s[s["situation"] == "corner"]
    if len(corners):
        flag_x = np.where(corners["px"] > 50, 100.0, 0.0)
        flag_y = np.where(corners["py"] > 50, 100.0, 0.0)
        points.append(pd.DataFrame({
            "px": flag_x, "py": flag_y,
            "w": np.full(len(corners), 0.6),
            "time_minute": corners["time_minute"].values,
        }))

    on_target = s["shot_type"].isin(["goal", "save", "post"])
    goal_x = np.where(s["px"] > 50, 99.5, 0.5)
    points.append(pd.DataFrame({
        "px": goal_x, "py": np.full(len(s), 50.0),
        "w": np.where(on_target, 0.5 + s["xgot"].fillna(0.1), 0.2),
        "time_minute": s["time_minute"].values,
    }))

    return pd.concat(points, ignore_index=True)


def smooth2d(z: np.ndarray, sigma: float = SIGMA_CELLS) -> np.ndarray:
    """Separable gaussian blur (no scipy)."""
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    z = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, z)
    z = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, z)
    return z


def heat_for_window(pts: pd.DataFrame, lo: float, hi: float) -> np.ndarray:
    """Time-weighted density: every point in the match contributes, weighted
    by a gaussian falloff from the window — so each frame is built from all
    the match's datapoints instead of only the few inside a hard cutoff."""
    center = (lo + hi) / 2
    half = (hi - lo) / 2
    dist = np.maximum(np.abs(pts["time_minute"] - center) - half, 0)
    tw = np.exp(-0.5 * (dist / TIME_TAU) ** 2)
    z, _, _ = np.histogram2d(
        pts["py"], pts["px"], bins=[GRID_Y, GRID_X],
        range=[[0, 100], [0, 100]], weights=pts["w"] * tw,
    )
    return smooth2d(z)


def momentum_series(s: pd.DataFrame, ev: pd.DataFrame):
    max_min = int(max(90, s["time_minute"].max() if len(s) else 90,
                      ev["time_minute"].max() if len(ev) else 90))
    grid = np.arange(1, max_min + 1)
    home_sig, away_sig = np.zeros(len(grid)), np.zeros(len(grid))
    for _, r in s.iterrows():
        m = int(min(max(r["time_minute"], 1), max_min)) - 1
        w = 1.0 + 4.0 * (r["xg"] if pd.notna(r["xg"]) else 0.05)
        if r["shot_type"] == "goal":
            w += 3.0
        elif r["shot_type"] in ("save", "post"):
            w += 0.5
        (home_sig if r["is_home"] else away_sig)[m] += w
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.5) ** 2)
    k /= k.sum()
    return grid, np.convolve(home_sig, k, "same") - np.convolve(away_sig, k, "same")


def full_pitch_shapes(yref="y", xref="x"):
    """Full pitch in 0..100 x 0..100 coordinates, goals at x=0 and x=100."""
    ln = dict(color=C_INK, width=1.5)
    def rect(x0, y0, x1, y1):
        return dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=ln, xref=xref, yref=yref)
    shapes = [
        rect(0, 0, 100, 100),                                   # pitch
        dict(type="line", x0=50, y0=0, x1=50, y1=100, line=ln, xref=xref, yref=yref),
        dict(type="circle", x0=50 - 8.75, y0=50 - 12.9, x1=50 + 8.75, y1=50 + 12.9,
             line=ln, xref=xref, yref=yref),                    # centre circle
        rect(0, 21.1, 15.7, 78.9), rect(100 - 15.7, 21.1, 100, 78.9),      # penalty areas
        rect(0, 36.8, 5.2, 63.2), rect(100 - 5.2, 36.8, 100, 63.2),        # six-yard boxes
        rect(-1.5, 44.3, 0, 55.7), rect(100, 44.3, 101.5, 55.7),           # goals
        # penalty arcs
        dict(type="path", path=f"M 15.7,{50-7.5} C 21,{50-3} 21,{50+3} 15.7,{50+7.5}",
             line=ln, xref=xref, yref=yref),
        dict(type="path", path=f"M {100-15.7},{50-7.5} C {100-21},{50-3} {100-21},{50+3} {100-15.7},{50+7.5}",
             line=ln, xref=xref, yref=yref),
    ]
    for cx in (10.5, 50, 89.5):  # spots
        shapes.append(dict(type="circle", x0=cx - 0.5, y0=49.3, x1=cx + 0.5, y1=50.7,
                           line=dict(color=C_INK, width=1), fillcolor=C_INK,
                           xref=xref, yref=yref))
    return shapes


def window_band(lo, hi, peak):
    """The sliding dark window on the momentum strip (video-style)."""
    return dict(type="rect", x0=lo, x1=hi, y0=-peak * 1.35, y1=peak * 1.35,
                fillcolor="rgba(56,56,53,0.28)", line_width=0,
                xref="x2", yref="y2", layer="below")


def build_figure(match_id: int) -> go.Figure:
    match, ev, shots_raw = load_match(match_id)
    home, away = match["home_team"], match["away_team"]
    pts = pitch_coords(shots_raw)
    grid, mom = momentum_series(shots_raw, ev)
    peak = max(np.abs(mom).max(), 0.5)
    max_min = int(grid.max())

    # slide the window all the way to full time (narrowing near the very end)
    # rather than stopping at max_min - WINDOW, so the animation visibly
    # reaches 90' — or the actual final whistle, including extra time, since
    # max_min above is derived from the match's real event/shot data.
    centers = list(range(0, max_min, STEP))
    if not centers:
        centers = [0]
    windows = [(c, min(c + WINDOW, max_min)) for c in centers]
    heats = [heat_for_window(pts, lo, hi) for lo, hi in windows]
    zmax = max(h.max() for h in heats) or 1.0

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.72, 0.28], vertical_spacing=0.09,
        subplot_titles=(None, None),
    )

    # --- top: pitch heatmap (frame-animated)
    lo0, hi0 = windows[0]
    fig.add_trace(go.Heatmap(
        z=heats[0],
        x=np.linspace(100 / GRID_X / 2, 100 - 100 / GRID_X / 2, GRID_X),
        y=np.linspace(100 / GRID_Y / 2, 100 - 100 / GRID_Y / 2, GRID_Y),
        colorscale=HEAT_SCALE, zmin=0, zmax=zmax * 0.85, showscale=False,
        zsmooth="best",
        hovertemplate="attack intensity %{z:.2f}<extra></extra>",
    ), row=1, col=1)

    # --- bottom: momentum bars (static)
    fig.add_trace(go.Bar(
        x=grid, y=mom,
        marker_color=[C_HOME if v >= 0 else C_AWAY for v in mom],
        marker_line_width=0,
        hovertemplate="min %{x} · momentum %{y:.2f}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    # event markers on the momentum strip
    def ev_y(is_home, lane):
        return peak * lane if is_home else -peak * lane

    goals = ev[ev["incident_type"] == "goal"]
    cards = ev[ev["incident_type"] == "card"]
    subs = ev[ev["incident_type"] == "substitution"]
    fig.add_trace(go.Scatter(
        x=goals["time_minute"], y=[ev_y(h, 1.22) for h in goals["is_home"]],
        mode="text", text=["⚽"] * len(goals), textfont=dict(size=15),
        hovertext=[f"{int(m)}' GOAL — {p}" for m, p in zip(goals["time_minute"], goals["player"].fillna("Goal"))],
        hoverinfo="text", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=cards["time_minute"], y=[ev_y(h, 1.08) for h in cards["is_home"]],
        mode="markers",
        marker=dict(symbol="square", size=8,
                    color=[C_RED if c == "red" else C_YELLOW for c in cards["incident_class"]]),
        hovertext=[f"{int(m)}' {c} card — {p}" for m, c, p in
                   zip(cards["time_minute"], cards["incident_class"], cards["player"].fillna(""))],
        hoverinfo="text", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=subs["time_minute"], y=[ev_y(h, 1.0) for h in subs["is_home"]],
        mode="markers", marker=dict(symbol="triangle-up", size=7, color=C_NEUTRAL),
        hovertext=[f"{int(m)}' sub — {i} ⇄ {o}" for m, i, o in
                   zip(subs["time_minute"], subs["player_in"].fillna(""), subs["player_out"].fillna(""))],
        hoverinfo="text", showlegend=False,
    ), row=2, col=1)

    base_shapes = full_pitch_shapes() + [
        dict(type="line", x0=0.5, x1=max_min + 0.5, y0=0, y1=0,
             line=dict(color=C_NEUTRAL, width=1), xref="x2", yref="y2"),
        dict(type="line", x0=45.5, x1=45.5, y0=-peak * 1.35, y1=peak * 1.35,
             line=dict(color=C_NEUTRAL, width=1, dash="dot"), xref="x2", yref="y2"),
    ]

    def title_for(lo, hi):
        return (f"<b>{home} {int(match['home_score'])}–{int(match['away_score'])} {away}</b>"
                f"  ·  attack heatmap, minute {lo}–{hi}"
                f"<br><sup>{home} attacks → right goal · {away} attacks ← left goal · "
                f"bars: momentum (blue {home} / green {away})</sup>")

    frames = []
    for (lo, hi), h in zip(windows, heats):
        frames.append(go.Frame(
            name=str(lo),
            data=[go.Heatmap(z=h)],
            traces=[0],
            layout=go.Layout(shapes=base_shapes + [window_band(lo, hi, peak)],
                             title_text=title_for(lo, hi)),
        ))
    fig.frames = frames

    fig.update_layout(
        paper_bgcolor=C_SURFACE, plot_bgcolor=C_SURFACE,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=C_INK2, size=13),
        title=dict(text=title_for(lo0, hi0), x=0.5, xanchor="center", font=dict(color=C_INK, size=17)),
        shapes=base_shapes + [window_band(lo0, hi0, peak)],
        margin=dict(l=40, r=40, t=90, b=150),
        height=940, width=980, bargap=0.15,
        updatemenus=[dict(
            type="buttons", direction="left", x=0.5, xanchor="center", y=-0.17, yanchor="top",
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(frame=dict(duration=220, redraw=True),
                                      fromcurrent=True, transition=dict(duration=120))]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.5, xanchor="center", y=-0.02, yanchor="top", len=0.9,
            currentvalue=dict(prefix="window start: min ", font=dict(color=C_INK2)),
            steps=[dict(label=str(lo), method="animate",
                        args=[[str(lo)], dict(mode="immediate",
                                              frame=dict(duration=0, redraw=True))])
                   for lo, _ in windows],
        )],
    )
    # pitch axes: hidden, fixed aspect (105x68 pitch drawn in 100x100 coords)
    fig.update_xaxes(visible=False, range=[-3, 103], row=1, col=1)
    fig.update_yaxes(visible=False, range=[-3, 103], scaleanchor="x", scaleratio=0.68, row=1, col=1)
    # momentum axes
    fig.update_xaxes(title_text="minute", showgrid=False, zeroline=False,
                     color=C_MUTED, range=[0, max_min + 1], row=2, col=1)
    fig.update_yaxes(visible=False, range=[-peak * 1.38, peak * 1.38], row=2, col=1)
    return fig


if __name__ == "__main__":
    match_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    fig = build_figure(match_id)
    out = HERE / "momentum_animated.html"
    fig.write_html(out, auto_play=False, include_plotlyjs=True)
    print(f"wrote {out}")
