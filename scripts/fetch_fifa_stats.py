"""
FIFA.com official statistics page scraper (Player Statistics + Team Statistics).

The FIFA.com stats pages (see README.md, section 1) are client-side rendered
and have no documented public API. Under the hood they call FIFA's Gameday
backend (gameday-prod.fifa.mangodev.co.uk) with a short-lived anonymous
bearer token minted per page load. This script:

  1. Launches headless Chromium (Playwright) against the FIFA.com stats page
     just long enough to capture that anonymous token.
  2. Uses the token to query the Gameday "stories" endpoint directly for
     every stat leaderboard (Golden Boot, Attacking, Distribution,
     Defending, Discipline, Goalkeeping, Movement, Physical for players;
     analogous team groups for teams).
  3. Within each stat category, fully paginates EVERY metric's leaderboard
     (all of FIFA.com's client-side "Load more" pages) and unions the stat
     tags per player/team. This matters because the backend embeds real
     stat values only for actors ranked near the top of the metric being
     sorted — a player deep in the assists ranking gets null tags there
     even if he leads the tournament in shots — so only the union across
     all of a category's metrics yields a complete row. Remaining nulls
     (players with zero involvement in every metric of the category) are
     written as 0, matching what FIFA's own table renders. One CSV per
     category.

This is a manual-cross-check data source per README.md — treat it as a
supplement to, not a replacement for, the balldontlie API pull in
fetch_worldcup_data.py.

Usage (run from the project root):
    python scripts/fetch_fifa_stats.py                # both player and team stats
    python scripts/fetch_fifa_stats.py --skip-teams
    python scripts/fetch_fifa_stats.py --skip-players
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import requests
from playwright.sync_api import sync_playwright

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "fifa_stats"
COMPETITION_ID = "285023"
GAMEDAY_BASE = "https://gameday-prod.fifa.mangodev.co.uk/1-0"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

PLAYER_GROUPS = {
    "gcp_top_scorer": "adidas Golden Boot",
    "gcp_attack": "Attacking",
    "gcp_distribution": "Distribution",
    "gcp_defending": "Defending",
    "gcp_discipline": "Discipline",
    "gcp_goalkeeping": "Goalkeeping",
    "gcp_movement": "Movement",
    "gcp_physical": "Physical",
}

TEAM_GROUPS = {
    "gct_attack": "Team Attacking",
    "gct_distribution": "Team Distribution",
    "gct_defending": "Team Defending",
    "gct_discipline": "Team Discipline",
    "gct_goalkeeping": "Team Goalkeeping",
    "gct_movement": "Team Movement",
    "gct_physical": "Team Physical",
}


def capture_gameday_token(page_url: str) -> str:
    """Load the FIFA.com stats page in headless Chromium just long enough
    to capture the anonymous Gameday bearer token it mints client-side."""
    token_holder = {}

    def on_request(request):
        if "mangodev" in request.url and "token" not in token_holder:
            auth = request.headers.get("authorization")
            if auth:
                token_holder["token"] = auth

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.on("request", on_request)
        page.goto(page_url, wait_until="load", timeout=60000)
        page.wait_for_timeout(2000)
        try:
            page.locator('button:has-text("Reject All")').first.click(timeout=5000)
        except Exception:
            pass
        page.wait_for_timeout(6000)
        browser.close()

    if "token" not in token_holder:
        raise RuntimeError("Could not capture a Gameday auth token from the page")
    return token_holder["token"]


MAX_LIMIT = 25  # gameday-prod caps limit at 25; higher values 429 with "Pagination limit threshold breached"


def gameday_get(token: str, query: str, skip: int, limit: int = MAX_LIMIT, sort: str | None = None) -> dict:
    params = {"query": query, "skip": skip, "limit": limit}
    if sort:
        params["sort"] = sort
    headers = {"authorization": token, "user-agent": UA, "accept": "application/json, text/plain, */*"}

    for attempt in range(5):
        resp = requests.get(f"{GAMEDAY_BASE}/stories", headers=headers, params=params)
        if resp.status_code == 429 and "Pagination limit" not in resp.text:
            wait = 2**attempt
            print(f"    rate limited, retrying in {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError("Failed to fetch stories after retries")


def metrics_for_group(token: str, group: str) -> list[str]:
    """Every metric in a category, in FIFA.com's column order (first = the
    tab's default sort metric)."""
    query = (
        "(and resourceStatus==`urn:gd:resourceStatus:active` "
        f"_externalId~`urn:gd:story:classification:{group}:competitionId:{COMPETITION_ID}:(.*):rank_asc:page:1$`)"
    )
    data = gameday_get(token, query, skip=0, sort="tags.name==urn:gd:tag:story:fifa:column_number:asc")
    items = data.get("items", [])
    if not items:
        raise RuntimeError(f"No metrics found for group {group}")
    return [it["_externalId"].split(":")[7] for it in items]


def fetch_all_pages(token: str, group: str, metric: str) -> list[dict]:
    """Fully paginate every 'page:N' story for one metric (FIFA.com's own
    'Load more' pages), not just page 1, so no player/team is skipped."""
    query = (
        "(and resourceStatus==`urn:gd:resourceStatus:active` "
        f"_externalId~`urn:gd:story:classification:{group}:competitionId:{COMPETITION_ID}:{metric}:rank_asc:page:(.*)$`)"
    )
    items: list[dict] = []
    skip = 0
    while True:
        data = gameday_get(token, query, skip=skip)
        page_items = data.get("items", [])
        items.extend(page_items)
        if not data.get("anotherPage"):
            break
        skip += MAX_LIMIT
    return items


def flatten_actor(actor: dict) -> dict:
    key = actor.get("key", {})
    tags = {t["name"]: t["value"] for t in actor.get("tags", [])}
    row = {
        "person_id": key.get("_externalSportsPersonId") or key.get("_externalId"),
        "name": actor.get("name", {}).get("eng"),
        "team_id": key.get("_externalTeamId"),
        "team": tags.get("urn:gd:tag:story:team:name:eng"),
        "team_abbreviation": tags.get("urn:gd:tag:story:team:abbreviation"),
        "position": tags.get("urn:gd:tag:story:staff:position"),
    }
    for tag_name, value in tags.items():
        if tag_name.startswith("urn:gd:tag:football:stats:"):
            stat_name = tag_name.replace("urn:gd:tag:football:stats:", "")
            row[stat_name] = value
    return row


ID_COLS = ["person_id", "name", "team_id", "team", "team_abbreviation", "position", "rank"]


def fetch_category(token: str, group: str) -> pd.DataFrame:
    """Every player/team with every stat column for one category, built by
    unioning all of the category's metric leaderboards (see module docstring).
    Ranked by the tab's default sort metric."""
    metrics = metrics_for_group(token, group)
    merged: dict[str, dict] = {}

    for idx, metric in enumerate(metrics):
        for story in fetch_all_pages(token, group, metric):
            for actor in story.get("actors", []):
                flat = flatten_actor(actor)
                row = merged.setdefault(flat["person_id"], {})
                for k, v in flat.items():
                    if v is not None and row.get(k) is None:
                        row[k] = v
                if idx == 0 and "rank" not in row:
                    row["rank"] = actor.get("number")

    df = pd.DataFrame(merged.values())
    # only the category's own metrics are real table columns — actors also
    # carry stray tags from other contexts (e.g. a sparse "goals" tag inside
    # Attacking) that would 0-fill into misleading columns
    stat_cols = [m for m in metrics if m in df.columns]
    for c in stat_cols:
        df[c] = df[c].fillna(0)
    df = df.sort_values("rank", na_position="last").reset_index(drop=True)
    return df[[c for c in ID_COLS if c in df.columns] + stat_cols]


def fetch_all_categories(token: str, groups: dict[str, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for group, title in groups.items():
        print(f"  fetching {title} ({group})...")
        frames[title] = fetch_category(token, group)
        time.sleep(0.5)

    # identity backfill: a player's team/position may be missing in one
    # category's payloads but present in another's
    id_fields = ["team", "team_abbreviation", "position"]
    identity = (
        pd.concat([f[["person_id"] + [c for c in id_fields if c in f.columns]] for f in frames.values()])
        .groupby("person_id").first()
    )
    # deep-ranked actors omit the team-name tags, but every player actor key
    # carries a team id — map id -> name from wherever both appear together
    # (team actors have no team_id, so this can be empty)
    tid_frames = [f[["team_id", "team", "team_abbreviation"]]
                  for f in frames.values() if "team_id" in f.columns]
    team_map = (
        pd.concat(tid_frames).dropna(subset=["team_id", "team"]).groupby("team_id").first()
        if tid_frames else pd.DataFrame()
    )
    for title, df in frames.items():
        for col in [c for c in id_fields if c in df.columns and c in identity.columns]:
            df[col] = df[col].fillna(df["person_id"].map(identity[col]))
        if "team_id" in df.columns and len(team_map):
            df["team"] = df["team"].fillna(df["team_id"].map(team_map["team"]))
            df["team_abbreviation"] = df["team_abbreviation"].fillna(
                df["team_id"].map(team_map["team_abbreviation"]))
        filename = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_") + ".csv"
        df.to_csv(out_dir / filename, index=False)
        print(f"    {len(df)} rows -> {out_dir.name}/{filename}")


def main():
    parser = argparse.ArgumentParser(description="Scrape FIFA.com official tournament statistics")
    parser.add_argument("--skip-players", action="store_true")
    parser.add_argument("--skip-teams", action="store_true")
    args = parser.parse_args()

    print("Capturing Gameday auth token from FIFA.com...")
    token = capture_gameday_token(
        "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics/player-statistics"
    )
    print("  token captured.")

    if not args.skip_players:
        print("Fetching player statistics (one CSV per category)...")
        fetch_all_categories(token, PLAYER_GROUPS, DATA_DIR / "players")

    if not args.skip_teams:
        print("Fetching team statistics (one CSV per category)...")
        fetch_all_categories(token, TEAM_GROUPS, DATA_DIR / "teams")

    print("\nDone.")


if __name__ == "__main__":
    main()
