"""
FIFA World Cup 2026 minute-by-minute data fetcher.

Pulls from the BALLDONTLIE FIFA World Cup API:
  - matches            (schedule, scores, stage/group/knockout structure)
  - match_events        (goals, cards, subs — each tagged with the match minute)
  - match_shots          (shot map: minute, coordinates, xG/xGOT)
  - player_match_stats  (per-player per-match box score)
  - team_match_stats    (per-team per-match box score)

Requires a BALLDONTLIE FIFA World Cup API key with at least GOAT-tier access
(match_events / match_shots / player_match_stats / team_match_stats are not
available on the free tier). Set it via a `.env` file (copy .env.example) or
the BALLDONTLIE_API_KEY environment variable — never hardcode it in this file.

Usage (run from the project root):
    python scripts/fetch_worldcup_data.py                  # fetch everything for completed/in-progress matches
    python scripts/fetch_worldcup_data.py --status completed
    python scripts/fetch_worldcup_data.py --match-ids 1030 1031
    python scripts/fetch_worldcup_data.py --skip-shots --skip-stats
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional; env vars can be set directly

BASE_URL = "https://api.balldontlie.io/fifa/worldcup/v1"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MAX_RETRIES = 5

_last_request_time = 0.0
_min_interval = 1.0  # seconds between requests; refined from x-ratelimit-limit header


def get_api_key() -> str:
    key = os.environ.get("BALLDONTLIE_API_KEY")
    if not key:
        sys.exit(
            "Missing API key. Copy .env.example to .env and set "
            "BALLDONTLIE_API_KEY, or export it in your shell."
        )
    return key


def paginated_get(endpoint: str, params: dict | None = None) -> list[dict]:
    """Fetch every page of a balldontlie cursor-paginated endpoint."""
    headers = {"Authorization": get_api_key()}
    params = dict(params or {})
    params.setdefault("per_page", 100)

    records: list[dict] = []
    cursor = None

    while True:
        if cursor is not None:
            params["cursor"] = cursor

        for attempt in range(MAX_RETRIES):
            global _last_request_time, _min_interval
            elapsed = time.time() - _last_request_time
            if elapsed < _min_interval:
                time.sleep(_min_interval - elapsed)

            resp = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params)
            _last_request_time = time.time()

            rate_limit = resp.headers.get("x-ratelimit-limit")
            if rate_limit:
                try:
                    _min_interval = max(60.0 / max(int(rate_limit), 1) * 1.15, 0.1)
                except ValueError:
                    pass

            if resp.status_code == 429:
                reset = resp.headers.get("x-ratelimit-reset")
                if reset:
                    wait = max(float(reset) - time.time(), 1.0)
                else:
                    wait = 2 ** attempt
                print(f"  rate limited, retrying in {wait:.0f}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            raise RuntimeError(f"Failed to fetch {endpoint} after {MAX_RETRIES} retries")

        payload = resp.json()
        page = payload.get("data", [])
        records.extend(page)

        next_cursor = payload.get("meta", {}).get("next_cursor")
        if not next_cursor or not page:
            break
        cursor = next_cursor

    return records


def flatten_match(m: dict) -> dict:
    home = m.get("home_team") or {}
    away = m.get("away_team") or {}
    return {
        "match_id": m["id"],
        "match_number": m.get("match_number"),
        "datetime": m.get("datetime"),
        "status": m.get("status"),
        "stage": (m.get("stage") or {}).get("name"),
        "group": (m.get("group") or {}).get("name") if m.get("group") else None,
        "round_name": m.get("round_name"),
        "home_team": home.get("name"),
        "away_team": away.get("name"),
        "home_score": m.get("home_score"),
        "away_score": m.get("away_score"),
        "has_extra_time": m.get("has_extra_time"),
        "has_penalty_shootout": m.get("has_penalty_shootout"),
        "home_score_penalties": m.get("home_score_penalties"),
        "away_score_penalties": m.get("away_score_penalties"),
        "stadium": (m.get("stadium") or {}).get("name"),
    }


def flatten_event(e: dict) -> dict:
    player = e.get("player") or {}
    assist = e.get("assist_player") or {}
    player_in = e.get("player_in") or {}
    player_out = e.get("player_out") or {}
    return {
        "match_id": e["match_id"],
        "time_minute": e.get("time_minute"),
        "added_time": e.get("added_time"),
        "period": e.get("period"),
        "incident_type": e.get("incident_type"),
        "incident_class": e.get("incident_class"),
        "is_home": e.get("is_home"),
        "player": player.get("name"),
        "assist_player": assist.get("name"),
        "player_in": player_in.get("name"),
        "player_out": player_out.get("name"),
        "home_score": e.get("home_score"),
        "away_score": e.get("away_score"),
        "rescinded": e.get("rescinded"),
    }


def flatten_shot(s: dict) -> dict:
    return {
        "match_id": s["match_id"],
        "player_id": s.get("player_id"),
        "team_id": s.get("team_id"),
        "is_home": s.get("is_home"),
        "time_minute": s.get("time_minute"),
        "added_time": s.get("added_time"),
        "shot_type": s.get("shot_type"),
        "situation": s.get("situation"),
        "body_part": s.get("body_part"),
        "goal_type": s.get("goal_type"),
        "xg": s.get("xg"),
        "xgot": s.get("xgot"),
        "player_x": s.get("player_x"),
        "player_y": s.get("player_y"),
    }


def fetch_matches(season: int, status: str | None, match_ids: list[int] | None) -> pd.DataFrame:
    params = {"seasons[]": season}
    if match_ids:
        params["match_ids[]"] = match_ids
    raw = paginated_get("matches", params)
    df = pd.DataFrame([flatten_match(m) for m in raw])
    if status and not df.empty:
        df = df[df["status"] == status]
    return df.sort_values("match_number").reset_index(drop=True) if not df.empty else df


def fetch_for_matches(endpoint: str, flatten_fn, match_ids: list[int]) -> pd.DataFrame:
    """balldontlie's event-level endpoints require match_ids[] in reasonably sized batches."""
    all_rows = []
    batch_size = 25
    for i in range(0, len(match_ids), batch_size):
        batch = match_ids[i : i + batch_size]
        print(f"  fetching {endpoint} for matches {batch[0]}..{batch[-1]} ({len(batch)} matches)")
        raw = paginated_get(endpoint, {"match_ids[]": batch})
        all_rows.extend(flatten_fn(r) for r in raw)
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Fetch FIFA World Cup 2026 data from balldontlie")
    parser.add_argument("--season", type=int, default=2026, choices=[2018, 2022, 2026])
    parser.add_argument("--status", choices=["scheduled", "in_progress", "completed", "postponed", "cancelled"])
    parser.add_argument("--match-ids", type=int, nargs="*", help="Only fetch these specific match IDs")
    parser.add_argument("--skip-events", action="store_true")
    parser.add_argument("--skip-shots", action="store_true")
    parser.add_argument("--skip-stats", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    print("Fetching matches...")
    matches_df = fetch_matches(args.season, args.status, args.match_ids)
    if matches_df.empty:
        print("No matches found for the given filters. Nothing to do.")
        return
    matches_df.to_csv(DATA_DIR / "matches.csv", index=False)
    print(f"  {len(matches_df)} matches -> data/matches.csv")

    match_ids = matches_df["match_id"].tolist()

    if not args.skip_events:
        print("Fetching match events (minute-by-minute goals/cards/subs)...")
        events_df = fetch_for_matches("match_events", flatten_event, match_ids)
        events_df.to_csv(DATA_DIR / "match_events.csv", index=False)
        print(f"  {len(events_df)} events -> data/match_events.csv")

    if not args.skip_shots:
        print("Fetching shot maps...")
        shots_df = fetch_for_matches("match_shots", flatten_shot, match_ids)
        shots_df.to_csv(DATA_DIR / "match_shots.csv", index=False)
        print(f"  {len(shots_df)} shots -> data/match_shots.csv")

    if not args.skip_stats:
        print("Fetching player match stats...")
        player_raw = []
        for i in range(0, len(match_ids), 25):
            batch = match_ids[i : i + 25]
            player_raw.extend(paginated_get("player_match_stats", {"match_ids[]": batch}))
        pd.DataFrame(player_raw).to_csv(DATA_DIR / "player_match_stats.csv", index=False)
        print(f"  {len(player_raw)} rows -> data/player_match_stats.csv")

        print("Fetching team match stats...")
        team_raw = []
        for i in range(0, len(match_ids), 25):
            batch = match_ids[i : i + 25]
            team_raw.extend(paginated_get("team_match_stats", {"match_ids[]": batch}))
        pd.DataFrame(team_raw).to_csv(DATA_DIR / "team_match_stats.csv", index=False)
        print(f"  {len(team_raw)} rows -> data/team_match_stats.csv")

    print("\nDone. CSVs are in the data/ folder.")


if __name__ == "__main__":
    main()
