import os
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# ===== 基本信息 =====
OWNER = os.environ.get("GITHUB_REPOSITORY").split("/")[0]
REPO = os.environ.get("GITHUB_REPOSITORY").split("/")[1]
TOKEN = os.environ.get("GITHUB_TOKEN")

OUTPUT = "traffic_views.png"

headers = {
    "Authorization": f"token {TOKEN}",
    "Accept": "application/vnd.github+json"
}

url = f"https://api.github.com/repos/{OWNER}/{REPO}/traffic/views"
response = requests.get(url, headers=headers)
response.raise_for_status()

data = response.json()["views"]

dates = [
    datetime.strptime(v["timestamp"], "%Y-%m-%dT%H:%M:%SZ").date()
    for v in data
]
views = [v["count"] for v in data]
uniques = [v["uniques"] for v in data]

# ===== 绘图 =====
plt.figure(figsize=(10, 4))
plt.plot(dates, views, marker="o", label="Views")
plt.plot(dates, uniques, marker="s", label="Unique Visitors")

plt.title("GitHub Repository Traffic (Last 14 Days)")
plt.xlabel("Date")
plt.ylabel("Count")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
plt.close()

print("Traffic image generated:", OUTPUT)
