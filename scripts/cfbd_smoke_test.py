import argparse
import os
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("season", type=int)
parser.add_argument("--classification", default="fbs", choices=("fbs", "fcs"))
args = parser.parse_args()

response = httpx.get(
    "https://api.collegefootballdata.com/games",
    params={"year": args.season, "classification": args.classification},
    headers={"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"},
    timeout=30,
)
response.raise_for_status()
games = response.json()

filename = f"{args.season}.json" if args.classification == "fbs" else f"{args.season}-{args.classification}.json"
output_path = Path("data/raw/cfbd/games") / filename
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_bytes(response.content)

print(f"Downloaded {len(games)} games to {output_path}")
