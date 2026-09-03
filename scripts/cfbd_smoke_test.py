import os

import httpx

api_key = os.environ["CFBD_API_KEY"]
response = httpx.get(
    "https://api.collegefootballdata.com/games",
    params={"year": 2023, "team": "Michigan"},
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30,
)
response.raise_for_status()
games = response.json()

print(f"Games returned: {len(games)}")
print(f"First matchup: {games[0]['awayTeam']} at {games[0]['homeTeam']}")
