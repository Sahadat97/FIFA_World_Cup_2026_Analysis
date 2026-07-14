"""
Animated 3D match-momentum visualization (option B, standalone).

Video-style look: dark scene, a 3D pitch surface whose terrain rises where a
team dominates — home team's zones in blue, away's in red — morphing as a
rolling window sweeps the match. Scoreboard with goal minutes top-left, clock
top-right, full-match momentum area chart with progress cursor at the bottom.

Usage:
    python momentum_plot_3d.py          # defaults to match_id 1
    python momentum_plot_3d.py 30       # any match_id from data/matches.csv

Writes momentum_animated_3d.html next to this script.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).parent

# dark-mode steps of the reference palette; home/away use its diverging pair
C_HOME = "#3987e5"      # blue (dark-surface step)
C_AWAY = "#e66767"      # red  (dark-surface step)
C_BG = "#0d0d0d"
C_SURF_MID = "#1a1a19"
C_TEXT = "#ffffff"
C_TEXT2 = "#c3c2b7"
C_MUTED = "#898781"

SURFACE_SCALE = [       # signed control: away red <- dark midpoint -> home blue
    [0.0, C_AWAY], [0.35, "#5a2a2a"], [0.5, C_SURF_MID], [0.65, "#1d3a5f"], [1.0, C_HOME],
]

WINDOW = 15
STEP = 2
GRID_X, GRID_Y = 42, 28
SIGMA = 2.6             # broad hills, video-like terrain


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
    s = s.copy()
    s["px"] = np.where(s["is_home"], 100 - s["player_x"], s["player_x"])
    s["py"] = np.where(s["is_home"], s["player_y"], 100 - s["player_y"])
    s["w"] = 1.0 + 3.0 * s["xg"].fillna(0.05) + np.where(s["shot_type"] == "goal", 2.0, 0.0)
    return s


def gaussian_kernel2d(sigma: float, radius: int) -> np.ndarray:
    ax = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def smooth2d(z: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    r = kernel.shape[0] // 2
    zp = np.pad(z, r, mode="constant")
    out = np.zeros_like(z, dtype=float)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            out[i, j] = (zp[i:i + 2*r + 1, j:j + 2*r + 1] * kernel).sum()
    return out


def fields_for_window(s: pd.DataFrame, lo: float, hi: float, kernel: np.ndarray):
    """Returns (height, signed_control) grids for one window."""
    win = s[(s["time_minute"] >= lo) & (s["time_minute"] < hi)]
    def grid_of(side: pd.DataFrame) -> np.ndarray:
        z, _, _ = np.histogram2d(side["py"], side["px"], bins=[GRID_Y, GRID_X],
                                 range=[[0, 100], [0, 100]], weights=side["w"])
        return smooth2d(z, kernel)
    h = grid_of(win[win["is_home"]])
    a = grid_of(win[~win["is_home"]])
    return h + a, h - a


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


def pitch_outline_traces(z_level: float):
    """White pitch lines floating just above the surface base (Scatter3d)."""
    def seg(xs, ys):
        return go.Scatter3d(x=xs, y=ys, z=[z_level] * len(xs), mode="lines",
                            line=dict(color="rgba(255,255,255,0.85)", width=3),
                            hoverinfo="skip", showlegend=False)
    t = [
        seg([0, 100, 100, 0, 0], [0, 0, 100, 100, 0]),           # boundary
        seg([50, 50], [0, 100]),                                  # halfway
        seg([0, 15.7, 15.7, 0], [21.1, 21.1, 78.9, 78.9]),        # left box
        seg([100, 84.3, 84.3, 100], [21.1, 21.1, 78.9, 78.9]),    # right box
        seg([0, 5.2, 5.2, 0], [36.8, 36.8, 63.2, 63.2]),          # left six-yard
        seg([100, 94.8, 94.8, 100], [36.8, 36.8, 63.2, 63.2]),    # right six-yard
    ]
    th = np.linspace(0, 2 * np.pi, 60)
    t.append(seg(list(50 + 8.75 * np.cos(th)), list(50 + 12.9 * np.sin(th))))  # centre circle
    return t


def build_figure(match_id: int) -> go.Figure:
    match, ev, shots_raw = load_match(match_id)
    home, away = match["home_team"], match["away_team"]
    habbr = str(match.get("home_team") or "")[:3].upper()
    aabbr = str(match.get("away_team") or "")[:3].upper()
    s = pitch_coords(shots_raw)
    mgrid, mom = momentum_series(s, ev)
    max_min = int(mgrid.max())
    kernel = gaussian_kernel2d(SIGMA, radius=6)

    windows = [(c, min(c + WINDOW, max_min)) for c in range(0, max_min - WINDOW + STEP + 1, STEP)]
    all_fields = [fields_for_window(s, lo, hi, kernel) for lo, hi in windows]
    zmax = max(h.max() for h, _ in all_fields) or 1.0
    cmax = max(np.abs(c).max() for _, c in all_fields) or 1.0

    goals = ev[(ev["incident_type"] == "goal")].copy()

    xs = np.linspace(100 / GRID_X / 2, 100 - 100 / GRID_X / 2, GRID_X)
    ys = np.linspace(100 / GRID_Y / 2, 100 - 100 / GRID_Y / 2, GRID_Y)

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.82, 0.18], vertical_spacing=0.02,
        specs=[[{"type": "scene"}], [{"type": "xy"}]],
    )

    h0, c0 = all_fields[0]
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=h0, surfacecolor=c0,
        colorscale=SURFACE_SCALE, cmin=-cmax, cmax=cmax,
        showscale=False, opacity=0.97,
        contours=dict(x=dict(highlight=False), y=dict(highlight=False), z=dict(highlight=False)),
        hovertemplate="control %{surfacecolor:.2f}<extra></extra>",
        lighting=dict(ambient=0.55, diffuse=0.7, specular=0.25, roughness=0.75),
        lightposition=dict(x=50, y=-80, z=200),
    ), row=1, col=1)

    # goal markers inside the current window (updated per frame)
    def goal_marker_data(lo, hi):
        gs = s[(s["shot_type"] == "goal") & (s["time_minute"] >= lo) & (s["time_minute"] < hi)]
        labels = []
        for _, r in gs.iterrows():
            cand = goals[(goals["is_home"] == r["is_home"]) &
                         (abs(goals["time_minute"] - r["time_minute"]) <= 1)]
            nm = cand.iloc[0]["player"] if len(cand) and pd.notna(cand.iloc[0]["player"]) else "Goal"
            labels.append(f"⚽ {nm}")
        return gs["px"].tolist(), gs["py"].tolist(), [zmax * 1.12] * len(gs), labels

    gx, gy, gz, gl = goal_marker_data(*windows[0])
    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz, mode="markers+text", text=gl, textposition="top center",
        textfont=dict(color=C_TEXT, size=12),
        marker=dict(size=5, color=C_TEXT, symbol="circle"),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    for tr in pitch_outline_traces(zmax * 0.02):
        fig.add_trace(tr, row=1, col=1)

    # --- bottom: full-match momentum area + progress cursor
    fig.add_trace(go.Scatter(
        x=mgrid, y=np.clip(mom, 0, None), fill="tozeroy", mode="none",
        fillcolor="rgba(57,135,229,0.75)", hoverinfo="skip", showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=mgrid, y=np.clip(mom, None, 0), fill="tozeroy", mode="none",
        fillcolor="rgba(230,103,103,0.75)", hoverinfo="skip", showlegend=False,
    ), row=2, col=1)

    peak = max(np.abs(mom).max(), 0.5)

    def cursor_shape(hi):
        return dict(type="line", x0=hi, x1=hi, y0=-peak * 1.1, y1=peak * 1.1,
                    line=dict(color=C_TEXT, width=2), xref="x", yref="y")

    def annos(lo, hi):
        gsofar = goals[goals["time_minute"] <= hi].copy()
        gsofar["is_home"] = gsofar["is_home"].astype(bool)
        hs = int(gsofar[gsofar["is_home"]]["home_score"].max() or 0) if len(gsofar[gsofar["is_home"]]) else 0
        as_ = 0
        if len(gsofar):
            last = gsofar.iloc[-1]
            hs = int(last["home_score"]) if pd.notna(last["home_score"]) else 0
            as_ = int(last["away_score"]) if pd.notna(last["away_score"]) else 0
        half = "1ST HALF" if hi <= 45 else ("2ND HALF" if hi <= 90 else "EXTRA TIME")
        hgoals = "<br>".join(f"● {int(m)}'" for m in gsofar[gsofar["is_home"]]["time_minute"])
        agoals = "<br>".join(f"● {int(m)}'" for m in gsofar[~gsofar["is_home"]]["time_minute"])
        common = dict(xref="paper", yref="paper", showarrow=False, align="left")
        return [
            dict(x=0.02, y=0.98, text=f"<b>{habbr}</b>", font=dict(color=C_HOME, size=20), **common),
            dict(x=0.14, y=0.98, text=f"<b>{aabbr}</b>", font=dict(color=C_AWAY, size=20), **common),
            dict(x=0.02, y=0.88, text=f"<b>{hs}</b>", font=dict(color=C_TEXT, size=52), **common),
            dict(x=0.14, y=0.88, text=f"<b>{as_}</b>", font=dict(color=C_TEXT, size=52), **common),
            dict(x=0.02, y=0.74, text=hgoals or " ", font=dict(color=C_HOME, size=13), yanchor="top", **common),
            dict(x=0.14, y=0.74, text=agoals or " ", font=dict(color=C_AWAY, size=13), yanchor="top", **common),
            dict(x=0.98, y=0.98, text=f"<b>{hi}'</b>", font=dict(color=C_TEXT, size=40),
                 xanchor="right", **common),
            dict(x=0.98, y=0.90, text=half, font=dict(color=C_MUTED, size=13), xanchor="right", **common),
            dict(x=0.5, y=0.01, text=f"attack terrain · minute {lo}–{hi} · surface height = intensity, "
                                     f"color = who controls the zone",
                 font=dict(color=C_MUTED, size=11), xanchor="center", **common),
        ]

    frames = []
    for (lo, hi), (h, c) in zip(windows, all_fields):
        gx, gy, gz, gl = goal_marker_data(lo, hi)
        frames.append(go.Frame(
            name=str(lo),
            data=[go.Surface(z=h, surfacecolor=c),
                  go.Scatter3d(x=gx, y=gy, z=gz, text=gl)],
            traces=[0, 1],
            layout=go.Layout(shapes=[cursor_shape(hi)], annotations=annos(lo, hi)),
        ))
    fig.frames = frames

    lo0, hi0 = windows[0]
    fig.update_layout(
        paper_bgcolor=C_BG, plot_bgcolor=C_BG,
        font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', color=C_TEXT2),
        margin=dict(l=10, r=10, t=10, b=120),
        height=950, width=980,
        shapes=[cursor_shape(hi0)], annotations=annos(lo0, hi0),
        scene=dict(
            bgcolor=C_BG,
            xaxis=dict(visible=False, range=[-4, 104]),
            yaxis=dict(visible=False, range=[-4, 104]),
            zaxis=dict(visible=False, range=[0, zmax * 1.35]),
            aspectratio=dict(x=1.5, y=1.0, z=0.35),
            camera=dict(eye=dict(x=1.15, y=-1.5, z=0.75), up=dict(x=0, y=0, z=1)),
        ),
        updatemenus=[dict(
            type="buttons", direction="left", x=0.5, xanchor="center", y=-0.16, yanchor="top",
            bgcolor="#26262a", bordercolor="#3a3a3f", font=dict(color=C_TEXT),
            buttons=[
                dict(label="▶  Play", method="animate",
                     args=[None, dict(frame=dict(duration=240, redraw=True),
                                      fromcurrent=True, transition=dict(duration=140))]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0, x=0.5, xanchor="center", y=-0.04, yanchor="top", len=0.9,
            bgcolor="#26262a", bordercolor="#3a3a3f",
            font=dict(color=C_TEXT2),
            currentvalue=dict(prefix="window start: min ", font=dict(color=C_TEXT2)),
            steps=[dict(label=str(lo), method="animate",
                        args=[[str(lo)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
                   for lo, _ in windows],
        )],
    )
    fig.update_xaxes(visible=False, range=[0, max_min + 1], row=2, col=1)
    fig.update_yaxes(visible=False, range=[-peak * 1.15, peak * 1.15], row=2, col=1)
    return fig


if __name__ == "__main__":
    match_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    fig = build_figure(match_id)
    out = HERE / "momentum_animated_3d.html"
    fig.write_html(out, auto_play=False, include_plotlyjs=True)
    print(f"wrote {out}")
