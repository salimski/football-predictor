import os

API_KEY  = os.environ.get("API_FOOTBALL_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
API_HOST = "v3.football.api-sports.io"
LEAGUE_ID = 78

# Season date ranges for Bundesliga
SEASONS = [
    {"label": "2021/22", "from": "2021-08-01", "to": "2022-06-30"},
    {"label": "2022/23", "from": "2022-08-01", "to": "2023-06-30"},
    {"label": "2023/24", "from": "2023-08-01", "to": "2024-06-30"},
    {"label": "2024/25", "from": "2024-08-01", "to": "2025-06-30"},
    {"label": "2025/26", "from": "2025-08-01", "to": "2026-06-30"},
]

# Understat season identifiers
UNDERSTAT_SEASONS = ["2021", "2022", "2023", "2024", "2025"]

DB_PATH = "data/raw.db"

REQUEST_DELAY = 1.5  # seconds between API calls
