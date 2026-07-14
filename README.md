# FIFA World Cup 2026 Analysis

A data pipeline and Streamlit app for exploring the 2026 World Cup: match-by-match event data, shot maps, an animated momentum replay, and tournament-wide player and team leaderboards.

**Live dashboard:** https://fifaworldcup2026analysis-duwyambu6y4buabctacbpt.streamlit.app

## Project structure

```
app.py                       Streamlit app (EA FC style match center, flags, logo)
momentum_plot.py             builds the animated momentum replay (2D pitch heatmap)
momentum_plot_3d.py          an alternate 3D terrain version of the same replay
scripts/
  fetch_worldcup_data.py     pulls match/event/shot/box-score data (balldontlie API)
  fetch_fifa_stats.py        scrapes FIFA.com's official stat leaderboards
assets/
  Fifa26logo.svg             tournament logo, shown in the app sidebar
data/                        all fetched CSVs live here
  matches.csv
  match_events.csv
  match_shots.csv
  player_match_stats.csv
  team_match_stats.csv
  fifa_stats/players/*.csv   one CSV per stat category (Golden Boot, Attacking, ...)
  fifa_stats/teams/*.csv     same, at team level
```

## Setup

```bash
pip install -r requirements.txt
playwright install chromium   # only needed for scripts/fetch_fifa_stats.py

cp .env.example .env
# edit .env and paste your BALLDONTLIE_API_KEY
```

## Pulling data

Two independent scripts, two different sources.

**`scripts/fetch_worldcup_data.py`** hits the balldontlie API for match-level data: schedule, scores, minute-by-minute events, shot maps, player and team box scores. Needs an API key with GOAT-tier access (match-level and event-level endpoints aren't on the free tier).

```bash
python scripts/fetch_worldcup_data.py --status completed      # all finished matches
python scripts/fetch_worldcup_data.py --match-ids 1030 1031    # a specific set
python scripts/fetch_worldcup_data.py --skip-shots --skip-stats  # schedule + events only
```

It's safe to re-run any time. It overwrites the CSVs in `data/` with the latest pull. The script paginates and retries automatically, respecting whatever rate limit your key's tier gives you.

**`scripts/fetch_fifa_stats.py`** scrapes FIFA.com's official statistics pages directly, no API key needed. It drives headless Chromium just long enough to grab the anonymous auth token FIFA's own frontend uses, then queries their backend for every stat category (Golden Boot, Attacking, Distribution, Defending, Discipline, Goalkeeping, Movement, Physical) for both players and teams, following every "Load more" page so nothing gets cut off at the top 50.

```bash
python scripts/fetch_fifa_stats.py
python scripts/fetch_fifa_stats.py --skip-teams
```

One quirk worth knowing: FIFA's backend only fills in a player's stat columns on the leaderboard page it's currently sorted by. The same player's row is missing values on a different metric's page. The script works around this by pulling every metric per category and merging the results, so each output CSV ends up with one row per player, fully populated.

## Running the app

```bash
streamlit run app.py
```

The app reads straight from the `data/` CSVs, so run the fetch scripts first. Match Analysis lets you pick any match and see the momentum replay, shot map, and team comparison. Player Analysis covers the tournament-wide leaderboards.

## How the momentum replay works

This is the animated pitch heatmap at the top of the Match Analysis page. A shaded window sweeps across the match timeline while the pitch above it lights up wherever the attacking threat was during that window.

The bottom strip is a per-minute momentum score: every shot contributes a weight based on its xG (plus bonuses for goals and saves), split into home and away signals, smoothed with a gaussian kernel, then subtracted (home − away) so the bars lean blue when home is on top and green when away is. Goals, cards, and substitutions sit on top as their own markers.

The pitch heatmap took a couple of passes to get right. The first version only looked at shots strictly inside each 15-minute window, and with maybe 25 shots in an entire match, most windows only had two or three points. It looked like isolated blobs instead of a field of play. The fix was to let every shot in the match contribute to every frame, weighted by a gaussian falloff based on how far its minute is from the window (shots inside count fully, shots just outside fade in gently, distant ones don't register at all). Each shot also throws in a couple of extra real datapoints beyond just the strike location: a corner shot lights up the corner flag it came from, and every shot lights up the goal mouth it was aimed at, weighted by xGOT for shots on target. Run through a 60×40 grid with a separable gaussian blur, and the result actually reads as a heat field.

The animation itself is native Plotly: one frame per 2-minute step of the sliding window, each frame swapping out just the heatmap trace and updating the title and the shaded band on the strip below. The window keeps sliding all the way to the real final whistle, narrowing near the end. That includes extra time, since the match length comes straight from the event data rather than being hardcoded to 90 minutes.

`momentum_plot_3d.py` builds a second, more stylized version: a dark 3D terrain surface instead of a flat heatmap, closer to a broadcast graphic. It's not wired into the app yet.

## Data sources

### FIFA.com official statistics pages

`https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/statistics/player-statistics`
(and the `/team-statistics` equivalent) are what `fetch_fifa_stats.py` scrapes. They're the canonical numbers FIFA publishes, so they're also the right place to spot-check a stat by eye if something from the API looks off. The page itself has no public API and is client-side rendered, so a plain HTTP request returns an empty shell. That's why the fetch script needs a real browser to grab FIFA's backend token first.

### BALLDONTLIE FIFA World Cup API

Base URL: `https://api.balldontlie.io/fifa/worldcup/v1`
Docs: https://fifa.balldontlie.io/
Get a key: https://app.balldontlie.io
Auth header: `Authorization: YOUR_API_KEY`

Covers 2018, 2022, and 2026 via the `seasons[]` param (defaults to 2026). List endpoints are cursor-paginated: pass `cursor` from the previous response's `meta.next_cursor`, stop when it's null or the page is empty.

| Tier | Requests/min | $/mo | Access |
|---|---|---|---|
| Free | 5 | 0 | `teams`, `stadiums` only |
| ALL-STAR | 60 | 9.99 | adds `group_standings` |
| GOAT | 600 | 39.99 | everything, including match-level and event-level endpoints |

A 48-hour free trial of GOAT tier is available (5 req/min during the trial, payment method required, no charge until it ends).

| Endpoint | Tier | Granularity | Key fields |
|---|---|---|---|
| `GET /teams` | Free | Tournament | `id`, `name`, `abbreviation`, `country_code`, `confederation` |
| `GET /stadiums` | Free | Tournament | `id`, `name`, `city`, `country`, `capacity`, `latitude`, `longitude` |
| `GET /group_standings` | ALL-STAR | Group | `team`, `group`, `position`, `played/won/drawn/lost`, `goals_for/against`, `points` |
| `GET /matches` | GOAT | Match | schedule, live clock, scores by half/ET, `stage`, `group`, `stadium`, teams, referee, formations |
| `GET /players` | GOAT | Player (all-time) | biographical fields, supports `search=` |
| `GET /player_injuries` | GOAT | Player (current) | `injury_type`, `status`, `updated_at` |
| `GET /rosters` | GOAT | Player × tournament | cumulative appearances, goals, assists, cards, avg rating |
| `GET /match_lineups` | GOAT | Player × match | starter/sub, shirt number, position, formation |
| `GET /match_events` | GOAT | Minute | goal/card/sub/period/etc., minute, player, assist, running score |
| `GET /match_shots` | GOAT | Minute + coordinate | shot type, situation, body part, xG, xGOT, pitch coordinates |
| `GET /player_match_stats` | GOAT | Player × match | rating, minutes, xG/xA, passing, tackling, duel, GK stats |
| `GET /team_match_stats` | GOAT | Team × match | possession, xG, shots breakdown, corners, fouls, cards, passing |
| `GET /match_momentum` | GOAT | Minute | `minute`, `value` (positive favors home) |
| `GET /match_best_players` | GOAT | Match | best XI, man of the match, rating, reason |
| `GET /match_avg_positions` | GOAT | Player × match | average position heatmap centroid |
| `GET /match_team_form` | GOAT | Team × match | pre-match form, recent points |
| `GET /odds`, `/odds/opening`, `/odds/futures`, `/player_props` | GOAT | Match/Tournament | betting markets by vendor |

### What this project currently fetches

| File | Source endpoint |
|---|---|
| `data/matches.csv` | `/matches` |
| `data/match_events.csv` | `/match_events` |
| `data/match_shots.csv` | `/match_shots` |
| `data/player_match_stats.csv` | `/player_match_stats` |
| `data/team_match_stats.csv` | `/team_match_stats` |

Not fetched yet, but available if needed: `/rosters`, `/match_lineups`, `/match_momentum`, `/match_best_players`, `/match_avg_positions`, `/match_team_form`, `/group_standings`, and the odds/props/futures endpoints.

### Error codes

| Code | Meaning |
|---|---|
| 401 | Missing API key, or your tier doesn't cover this endpoint |
| 400 | Bad request / invalid params |
| 404 | Not found |
| 406 | Requested a non-JSON format |
| 429 | Rate limited, back off and retry |
| 500/503 | Server error / maintenance |

## License

MIT. See [LICENSE](LICENSE).
