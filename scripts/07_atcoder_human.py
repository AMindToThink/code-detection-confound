"""Build a REAL author-skill human cohort from AtCoder (confirmatory axis).

Addresses the #1 critique of the main run (problem difficulty != author skill). Here the
skill axis is the *author's own Elo rating*. No LLM side is needed: the question is simply
whether detectors score higher-skill HUMANS as more 'AI'.

Pipeline (polite ~1.2s between atcoder.jp / kenkoooo hits):
  1. stream recent AC Python submissions (kenkoooo /v3/from) -> candidate (user, contest, id)
  2. fetch each user's current rating once (atcoder.jp /users/<u>/history/json)
  3. bucket users: low_skill (rating <= LOW_MAX), high_skill (rating >= HIGH_MIN)
  4. scrape source for a few submissions per user until each band is filled

Output: data/atcoder_human.parquet (species='human', dataset='atcoder', band in
{low_skill, high_skill}, author_rating).
"""
from __future__ import annotations

import calendar
import html as H
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import config as C

LOW_MAX = 800            # author rating <= -> low skill (AtCoder gray/brown)
HIGH_MIN = 2000          # author rating >= -> high skill (AtCoder orange/red)
TARGET_PER_BAND = 45
MAX_PER_USER = 4
SLEEP = 1.2
MIN_CODE_CHARS = 60
WINDOWS = ["2019-09-01", "2020-06-01", "2021-03-01"]   # diverse time windows for user variety

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/json,text/html", "Accept-Language": "en-US,en;q=0.9",
})


def get(url: str):
    for attempt in range(3):
        try:
            r = S.get(url, timeout=30)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(SLEEP * (attempt + 1))
    return None


def user_rating(u: str, cache: dict) -> int | None:
    if u in cache:
        return cache[u]
    r = get(f"https://atcoder.jp/users/{u}/history/json")
    time.sleep(SLEEP)
    rating = None
    if r is not None:
        try:
            hist = r.json()
            rating = hist[-1]["NewRating"] if hist else None
        except Exception:
            rating = None
    cache[u] = rating
    return rating


def scrape_source(contest: str, sid: int) -> str | None:
    r = get(f"https://atcoder.jp/contests/{contest}/submissions/{sid}")
    time.sleep(SLEEP)
    if r is None:
        return None
    m = re.search(r'<pre[^>]*id="submission-code"[^>]*>(.*?)</pre>', r.text, re.DOTALL)
    if not m:
        return None
    code = H.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
    return code if len(code) >= MIN_CODE_CHARS else None


def main() -> None:
    # 1+2: gather candidate AC python submissions and user ratings
    rating_cache: dict[str, int | None] = {}
    band_users: dict[str, list[str]] = {"low_skill": [], "high_skill": []}
    cand: dict[str, list[tuple[str, int, str]]] = {}   # user -> [(contest, id, problem)]
    fails = 0

    for win in WINDOWS:
        sec = calendar.timegm(time.strptime(win, "%Y-%m-%d"))
        r = get(f"https://kenkoooo.com/atcoder/atcoder-api/v3/from/{sec}")
        time.sleep(SLEEP)
        if r is None:
            fails += 1
            continue
        subs = [d for d in r.json()
                if d.get("result") == "AC" and "Python" in (d.get("language") or "")]
        for d in subs:
            cand.setdefault(d["user_id"], []).append((d["contest_id"], d["id"], d["problem_id"]))
        print(f"window {win}: {len(subs)} AC python subs, {len(cand)} unique users so far", flush=True)
        if all(len(band_users[b]) >= TARGET_PER_BAND // MAX_PER_USER + 5 for b in band_users):
            break

    # classify users by rating (only as many as needed)
    users = list(cand)
    for u in users:
        if all(len(band_users[b]) >= (TARGET_PER_BAND // MAX_PER_USER + 6) for b in band_users):
            break
        rt = user_rating(u, rating_cache)
        if rt is None:
            continue
        if rt <= LOW_MAX:
            band_users["low_skill"].append(u)
        elif rt >= HIGH_MIN:
            band_users["high_skill"].append(u)
    print(f"users: low={len(band_users['low_skill'])} high={len(band_users['high_skill'])}")

    # 4: scrape source until each band filled
    rows = []
    for band in ("low_skill", "high_skill"):
        n = 0
        for u in band_users[band]:
            if n >= TARGET_PER_BAND:
                break
            for k, (contest, sid, problem) in enumerate(cand[u][:MAX_PER_USER]):
                if n >= TARGET_PER_BAND:
                    break
                code = scrape_source(contest, sid)
                if code is None:
                    fails += 1
                    continue
                rows.append({
                    "sample_id": f"AT::{u}::{sid}",
                    "problem_id": problem, "band": band,
                    "problem_rating": None, "author_rating": rating_cache[u],
                    "species": "human", "model": "human", "condition": "human",
                    "code": code, "skill_source": "atcoder_author_rating",
                    "dataset": "atcoder",
                })
                n += 1
        print(f"band {band}: {n} source files scraped", flush=True)

    df = pd.DataFrame(rows)
    if len(df) < 20:
        raise RuntimeError(f"only {len(df)} AtCoder samples scraped ({fails} fails) — too few")
    df.to_parquet(C.DATA / "atcoder_human.parquet", index=False)
    print(f"wrote data/atcoder_human.parquet: {len(df)} samples ({fails} fetch fails)")
    print(df.groupby('band').agg(n=('sample_id', 'count'),
                                 mean_rating=('author_rating', 'mean')))


if __name__ == "__main__":
    main()
