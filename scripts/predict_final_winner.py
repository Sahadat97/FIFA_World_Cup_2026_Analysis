"""
Predicts each remaining team's chance of winning the FIFA World Cup 2026
final, using a gradient boosting classifier trained on every completed match
so far plus the FIFA.com tournament-cumulative team leaderboards.

Approach:
  1. Build a "strength profile" per team: match-box-score averages
     (possession, xG, shots, passing, duels, discipline) from
     team_match_stats.csv, results-derived form (win rate, goal difference)
     from matches.csv, and cumulative tournament totals from every
     data/fifa_stats/teams/*.csv category (attacking, defending,
     discipline, distribution, goalkeeping, physical, movement).
  2. Train a HistGradientBoostingClassifier on every completed match,
     framed as team-A-profile minus team-B-profile -> {A win, draw, B win},
     mirrored (A vs B and B vs A) to remove ordering bias. Evaluate with
     stratified cross-validation before trusting it on anything.
  3. For the remaining teams, predict pairwise win probability for every
     possible matchup, resolve draws 50/50 (there are no draws left, every
     remaining match goes to extra time / penalties on a level score),
     then Monte Carlo simulate the semifinal + final bracket. Since the
     actual semifinal pairing isn't in our pulled data yet, this is run
     for all 3 possible ways to pair 4 teams into 2 semifinals and averaged.

Usage:
    python scripts/predict_final_winner.py
"""

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, log_loss

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"
RNG = np.random.default_rng(42)


def build_team_profiles() -> pd.DataFrame:
    matches = pd.read_csv(DATA / "matches.csv")
    team_stats = pd.read_csv(DATA / "team_match_stats.csv")

    # map team_id -> team name via (match_id, is_home), same trick app.py uses
    side_names = matches.melt(
        id_vars="match_id", value_vars=["home_team", "away_team"],
        var_name="side", value_name="team",
    )
    side_names["is_home"] = side_names["side"] == "home_team"
    ts = team_stats.merge(side_names[["match_id", "is_home", "team"]], on=["match_id", "is_home"])

    ts["pass_accuracy"] = ts["passes_accurate"] / ts["passes_total"].replace(0, np.nan)
    ts["aerial_win_rate"] = ts["aerial_duels_won"] / ts["aerial_duels_total"].replace(0, np.nan)
    ts["ground_win_rate"] = ts["ground_duels_won"] / ts["ground_duels_total"].replace(0, np.nan)

    match_cols = [
        "possession_pct", "expected_goals", "shots_total", "shots_on_target",
        "big_chances", "corners", "fouls", "yellow_cards", "pass_accuracy",
        "tackles", "interceptions", "aerial_win_rate", "ground_win_rate",
    ]
    profile = ts.groupby("team")[match_cols].mean()
    profile.columns = [f"box_{c}" for c in profile.columns]

    # results-derived form
    home = matches[["home_team", "away_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "away_team": "opp", "home_score": "gf", "away_score": "ga"})
    away = matches[["away_team", "home_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "home_team": "opp", "away_score": "gf", "home_score": "ga"})
    results = pd.concat([home, away], ignore_index=True).dropna(subset=["gf", "ga"])
    results["win"] = (results["gf"] > results["ga"]).astype(float)
    results["draw"] = (results["gf"] == results["ga"]).astype(float)
    form = results.groupby("team").agg(
        form_matches_played=("gf", "size"),
        form_win_rate=("win", "mean"),
        form_draw_rate=("draw", "mean"),
        form_goals_for=("gf", "mean"),
        form_goals_against=("ga", "mean"),
    )
    form["form_goal_diff"] = form["form_goals_for"] - form["form_goals_against"]
    profile = profile.join(form, how="outer")

    # FIFA.com cumulative team leaderboards: pull a curated numeric subset
    # from every category, prefixed by category so names never collide.
    fifa_categories = {
        "team_attacking": ["goals", "xg", "attempt_at_goal_conversion_rate", "corners", "possession"],
        "team_defending": ["forced_turnovers", "ball_recovery_time", "defensive_pressures_applied", "goals_conceded"],
        "team_discipline": ["fouls_for", "fouls_against", "yellow_cards", "red_cards", "offsides"],
        "team_distribution": ["passing_accuracy_rate", "crosses", "crossing_accuracy_rate"],
        "team_goalkeeping": ["clean_sheets", "goalkeeper_saves"],
        "team_physical": ["avg_speed", "sprints", "total_distance"],
        "team_movement": ["offers_to_receive_total", "receptions_under_pressure"],
    }
    # FIFA.com spells a few names differently than balldontlie/matches.csv
    name_fix = {
        "Bosnia and Herzegovina": "Bosnia & Herzegovina",
        "Congo DR": "DR Congo",
        "Korea Republic": "South Korea",
        "IR Iran": "Iran",
    }
    for fname, cols in fifa_categories.items():
        df = pd.read_csv(DATA / "fifa_stats" / "teams" / f"{fname}.csv")
        df["team"] = df["team"].replace(name_fix)
        df = df.set_index("team")[cols].add_prefix(f"fifa_{fname.replace('team_', '')}_")
        profile = profile.join(df, how="outer")

    return profile


def build_training_set(profile: pd.DataFrame):
    matches = pd.read_csv(DATA / "matches.csv").dropna(subset=["home_score", "away_score"])
    matches = matches[matches["home_team"].isin(profile.index) & matches["away_team"].isin(profile.index)]

    rows, labels = [], []
    for _, m in matches.iterrows():
        a, b = m["home_team"], m["away_team"]
        diff = (profile.loc[a] - profile.loc[b]).values
        if m["home_score"] > m["away_score"]:
            label = 2  # A win
        elif m["home_score"] < m["away_score"]:
            label = 0  # B win
        else:
            label = 1  # draw
        # add both directions so the model doesn't learn a home/away bias
        rows.append(diff)
        labels.append(label)
        rows.append(-diff)
        labels.append(2 - label)

    X = np.nan_to_num(np.array(rows), nan=0.0)
    y = np.array(labels)
    return X, y


def train_and_evaluate(X, y):
    """Returns (fitted_model, metrics_dict). Caller decides whether to print."""
    model = HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.08,
        l2_regularization=1.0, random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    proba_cv = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    pred_cv = proba_cv.argmax(axis=1)
    metrics = {
        "accuracy": accuracy_score(y, pred_cv),
        "log_loss": log_loss(y, proba_cv, labels=[0, 1, 2]),
        "baseline_accuracy": max(np.bincount(y)) / len(y),
        "n_matches": len(y) // 2,
    }

    model.fit(X, y)
    return model, metrics


# World Cup knockout progression, in order. Group Stage has no single
# "winner" per match (many teams advance by table position, not a 1v1
# result), so stage-advancement logic only applies from Round of 32 on.
KNOCKOUT_STAGE_ORDER = ["Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"]


def determine_next_matchup_pool(matches: pd.DataFrame):
    """Looks at the latest completed knockout round and returns who's left.

    Returns one of:
      ("champion", winner_name)              - the Final has been played
      (next_stage_name, [team, team, ...])    - teams advancing into next_stage
      (None, None)                            - knockout stage hasn't started
    """
    completed = matches.dropna(subset=["home_score", "away_score"])
    completed_knockout_stages = [s for s in KNOCKOUT_STAGE_ORDER if s in set(completed["stage"])]
    if not completed_knockout_stages:
        return None, None

    latest_stage = completed_knockout_stages[-1]
    stage_matches = completed[completed["stage"] == latest_stage]

    winners = []
    for _, m in stage_matches.iterrows():
        if m["home_score"] > m["away_score"]:
            winners.append(m["home_team"])
        elif m["away_score"] > m["home_score"]:
            winners.append(m["away_team"])
        elif pd.notna(m.get("home_score_penalties")) and pd.notna(m.get("away_score_penalties")):
            winners.append(m["home_team"] if m["home_score_penalties"] > m["away_score_penalties"] else m["away_team"])
        # else: scores level with no shootout recorded yet -> match not truly settled, skip

    if latest_stage == "Final":
        return "champion", (winners[0] if winners else None)

    stage_idx = KNOCKOUT_STAGE_ORDER.index(latest_stage)
    next_stage = KNOCKOUT_STAGE_ORDER[stage_idx + 1]
    return next_stage, winners


def all_pairings_of_four(teams: list[str]):
    a, b, c, d = teams
    return [[(a, b), (c, d)], [(a, c), (b, d)], [(a, d), (b, c)]]


def matchup_advance_prob(model, profile, team_a, team_b) -> float:
    """P(team_a advances past team_b), resolving a predicted draw 50/50
    since knockout matches are always settled (extra time / penalties)."""
    diff = np.nan_to_num((profile.loc[team_a] - profile.loc[team_b]).values, nan=0.0).reshape(1, -1)
    p_b_win, p_draw, p_a_win = model.predict_proba(diff)[0]
    return p_a_win + 0.5 * p_draw


def simulate_bracket(model, profile, semifinal_pairs, n_sims=20000):
    """Monte Carlo the SF -> Final for one bracket pairing."""
    (sf1_a, sf1_b), (sf2_a, sf2_b) = semifinal_pairs
    p_sf1 = matchup_advance_prob(model, profile, sf1_a, sf1_b)
    p_sf2 = matchup_advance_prob(model, profile, sf2_a, sf2_b)

    finalists_a = RNG.random(n_sims) < p_sf1
    finalists_b = RNG.random(n_sims) < p_sf2
    finalist1 = np.where(finalists_a, sf1_a, sf1_b)
    finalist2 = np.where(finalists_b, sf2_a, sf2_b)

    champion = np.empty(n_sims, dtype=object)
    for a, b in set(zip(finalist1, finalist2)):
        mask = (finalist1 == a) & (finalist2 == b)
        n = mask.sum()
        if n == 0:
            continue
        p_a = matchup_advance_prob(model, profile, a, b)
        champion[mask] = np.where(RNG.random(n) < p_a, a, b)

    teams = [sf1_a, sf1_b, sf2_a, sf2_b]
    return {t: float((champion == t).mean()) for t in teams}


def main():
    print("Building team strength profiles from match box scores, results, and FIFA.com leaderboards...")
    profile = build_team_profiles()
    print(f"  {profile.shape[0]} teams, {profile.shape[1]} features each")

    X, y = build_training_set(profile)
    print(f"Training on {len(y)} directed match samples ({len(y)//2} completed matches, mirrored)...")
    model, metrics = train_and_evaluate(X, y)
    print(f"5-fold cross-validated accuracy: {metrics['accuracy']:.1%}  "
          f"(majority-class baseline: {metrics['baseline_accuracy']:.1%})")
    print(f"5-fold cross-validated log-loss: {metrics['log_loss']:.3f}  "
          f"(lower is better; 1.099 = uniform random over 3 classes)")

    matches = pd.read_csv(DATA / "matches.csv")
    stage, pool = determine_next_matchup_pool(matches)

    if stage is None:
        print("\nKnockout stage hasn't started yet, nothing to predict.")
        return
    if stage == "champion":
        print(f"\nThe tournament is decided: {pool} won the FIFA World Cup 2026.")
        return

    print(f"\nTeams advancing to the {stage}: {', '.join(pool)}")

    if len(pool) == 2:
        p_a = matchup_advance_prob(model, profile, pool[0], pool[1])
        avg = {pool[0]: p_a, pool[1]: 1 - p_a}
        print(f"  {pool[0]:12s} {avg[pool[0]]:.1%} chance of winning the final")
        print(f"  {pool[1]:12s} {avg[pool[1]]:.1%} chance of winning the final")
    elif len(pool) == 4:
        print("Semifinal pairing isn't in our pulled data yet, so simulating all 3 possible pairings:\n")
        totals = {t: 0.0 for t in pool}
        for pairing in all_pairings_of_four(pool):
            result = simulate_bracket(model, profile, pairing)
            label = f"{pairing[0][0]} vs {pairing[0][1]}  |  {pairing[1][0]} vs {pairing[1][1]}"
            print(f"  If SF pairing is [{label}]:")
            for t in pool:
                print(f"    {t:12s} {result[t]:.1%} chance of winning the final")
            print()
            for t in pool:
                totals[t] += result[t]
        avg = {t: totals[t] / 3 for t in pool}
        print("Averaged across all 3 possible pairings (final answer, since we don't know the real bracket):")
        for t in sorted(avg, key=avg.get, reverse=True):
            print(f"  {t:12s} {avg[t]:.1%}")
    else:
        print(f"({len(pool)} teams advancing - outside the 2/4-team cases this script simulates a bracket for.)")
        return

    out = pd.DataFrame({"team": list(avg.keys()), "win_final_probability": list(avg.values())})
    out = out.sort_values("win_final_probability", ascending=False).reset_index(drop=True)
    out_path = HERE / "recordings" / "final_winner_prediction.csv"
    out_path.parent.mkdir(exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
