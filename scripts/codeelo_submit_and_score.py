#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


LANG_TAG_TO_CF_LANG_ID: dict[str, int] = {
    "kotlin": 88,
    "cpp": 91,
    "c++": 91,
    "c": 43,
    "python": 70,
    "pypy": 70,
    "ruby": 67,
    "rust": 75,
    "go": 32,
    "javascript": 34,
    "node.js": 55,
    "typescript": 55,
    "c#": 79,
    "csharp": 79,
    "java": 87,
    "php": 6,
    "perl": 13,
    "ocaml": 19,
    "haskell": 12,
    "scala": 20,
    "pascal": 51,
    "d": 28,
    "delphi": 3,
}


def extract_contest_id_from_prob(prob: str) -> str | None:
    # Prob format usually like "2000A"
    m = re.match(r"(\d+)", prob.strip())
    return m.group(1) if m else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit CodeElo responses via CodeElo API and (optionally) score.")
    p.add_argument("--input_jsonl", type=str, required=True, help="1-stage generated jsonl with codeelo_submission.")
    p.add_argument("--tag", type=str, default="specpv", help="Optional submission tag.")

    p.add_argument("--base_url", type=str, default=os.environ.get("BASE_URL", ""), help="CodeElo API BASE_URL.")
    p.add_argument("--token", type=str, default=os.environ.get("TOKEN", ""), help="CodeElo API TOKEN.")

    p.add_argument("--max_problems", type=int, default=None, help="Limit problems for smoke runs.")
    p.add_argument("--poll_secs", type=float, default=2.0, help="Polling interval.")
    p.add_argument("--max_polls", type=int, default=240, help="Max polls per problem.")

    p.add_argument(
        "--compute_rating",
        action="store_true",
        help="Compute Elo rating locally using Codeforces + sorted_ratings.json (network required).",
    )
    p.add_argument(
        "--sorted_ratings_url",
        type=str,
        default="https://raw.githubusercontent.com/QwenLM/CodeElo/main/sorted_ratings.json",
        help="Where to download sorted_ratings.json when computing Elo.",
    )
    return p.parse_args()


class CodeEloAPI:
    def __init__(self, *, base_url: str, token: str):
        self.base_url = base_url.rstrip("/")
        self.token = token
        if not self.base_url:
            raise SystemExit("Missing --base_url or BASE_URL env var.")
        if not self.token:
            raise SystemExit("Missing --token or TOKEN env var.")

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Authorization": self.token,
            }
        )

    def check_auth(self) -> dict[str, Any]:
        r = self.session.get(f"{self.base_url}/check_auth", timeout=60)
        r.raise_for_status()
        return r.json()

    def submit_code(self, *, prob: str, lang: int, code: str, tag: str) -> dict[str, Any]:
        payload = {"prob": prob, "lang": lang, "code": code, "tag": tag}
        r = self.session.post(f"{self.base_url}/submit_code", json=payload, timeout=180)
        r.raise_for_status()
        return r.json()

    def check_status(self, *, submission_id: str) -> dict[str, Any]:
        r = self.session.get(
            f"{self.base_url}/check_status", params={"submission_id": submission_id}, timeout=60
        )
        r.raise_for_status()
        return r.json()


def poll_until_final(api: CodeEloAPI, *, submission_id: str, max_polls: int, poll_secs: float) -> dict[str, Any]:
    final = None
    for _ in range(max_polls):
        status = api.check_status(submission_id=submission_id)
        status_canonical = status.get("status_canonical") or status.get("status") or None
        if status_canonical in {"AC", "WA", "CE", "TLE", "OLE", "RE", "SK"}:
            final = status
            break
        # If the server uses statuses like "PENDING"/"RUNNING", just wait.
        time.sleep(poll_secs)
    if final is None:
        # last attempt
        final = api.check_status(submission_id=submission_id)
    return final


def download_json_sorted_ratings(url: str) -> list[float]:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    # saved as a json array of ratings
    return json.loads(r.text)


def calc_elo_rating(contest_id: int, problem_status: dict[str, list[str]], *, sorted_ratings: list[float]) -> float | None:
    # This is a direct adaptation of CodeElo's calc_rating.py.
    standings = requests.get(
        f"https://codeforces.com/api/contest.standings?contestId={contest_id}&showUnofficial=false",
        timeout=120,
    ).json()
    rating_changes = requests.get(
        f"https://codeforces.com/api/contest.ratingChanges?contestId={contest_id}",
        timeout=120,
    ).json()

    try:
        left_handles = set(
            [
                standings["result"]["rows"][i]["party"]["members"][0]["handle"]
                for i in range(len(standings["result"]["rows"]))
            ]
        )
        right_handles = set(
            [rating_changes["result"][i]["handle"] for i in range(len(rating_changes["result"]))]
        )
        handle_set = left_handles & right_handles
        standings["result"]["rows"] = [
            standings["result"]["rows"][i]
            for i in range(len(standings["result"]["rows"]))
            if standings["result"]["rows"][i]["party"]["members"][0]["handle"] in handle_set
        ]
        rating_changes["result"] = [
            rating_changes["result"][i]
            for i in range(len(rating_changes["result"]))
            if rating_changes["result"][i]["handle"] in handle_set
        ]
        if len(standings["result"]["rows"]) != len(rating_changes["result"]) or len(standings["result"]["rows"]) <= 200:
            return None
    except Exception:
        return None

    max_rating = 0
    for i in range(len(rating_changes["result"])):
        max_rating = max(max_rating, rating_changes["result"][i]["oldRating"])

    score = 0
    penalty = 0
    for problem in standings["result"]["problems"]:
        prob = f"{problem['contestId']}{problem['index']}"
        if prob in problem_status:
            for ith, st in enumerate(problem_status[prob]):
                if st == "AC":
                    if "points" in problem:
                        score += max(0, problem["points"] - 50 * ith)
                    else:
                        score += 1
                    penalty += ith * 10
                    break

    n = len(standings["result"]["rows"])
    rank = n
    for i in range(n):
        if standings["result"]["rows"][i]["points"] < score or (
            standings["result"]["rows"][i]["points"] == score and standings["result"]["rows"][i]["penalty"] > penalty
        ):
            rank = i
            break

    l, r = 0, max_rating + 100
    while r - l > 1:
        mid = int((l + r) / 2)
        new_seed = 1
        for i in range(n):
            new_seed += 1 / (1 + 10 ** ((mid - rating_changes["result"][i]["oldRating"]) / 400))
        if new_seed < rank:
            r = mid
        else:
            l = mid
    return float(l)


def get_percentile(rating: float, sorted_ratings: list[float]) -> float:
    import bisect

    idx = bisect.bisect_left(sorted_ratings, float(rating))
    return round(idx / len(sorted_ratings) * 100, 1)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    api = CodeEloAPI(base_url=args.base_url, token=args.token)
    api.check_auth()

    out_path = input_path.with_suffix("")
    out_path = Path(str(out_path) + f".submitted.jsonl")

    lines = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            lines.append(json.loads(line))
    if args.max_problems is not None:
        lines = lines[: args.max_problems]

    accepted = 0
    evaluated = 0

    # contest_id -> problem -> list of statuses for Elo logic
    problem_status_by_contest: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    submitted_rows: list[dict[str, Any]] = []
    for row in lines:
        code_submission = row.get("codeelo_submission")
        if not code_submission:
            row["submit_skip_reason"] = "missing_codeelo_submission"
            submitted_rows.append(row)
            continue

        prob = code_submission.get("prob")
        code = code_submission.get("code")
        lang_tag = code_submission.get("lang_tag")
        if not prob or not isinstance(prob, str):
            row["submit_skip_reason"] = "missing_prob"
            submitted_rows.append(row)
            continue
        if not code:
            row["submit_skip_reason"] = "missing_code"
            submitted_rows.append(row)
            continue

        # language id
        lang_tag = (lang_tag or "").strip().lower()
        lang_id = LANG_TAG_TO_CF_LANG_ID.get(lang_tag)
        if lang_id is None:
            # Default to C++ if unknown.
            lang_id = LANG_TAG_TO_CF_LANG_ID["cpp"]
            row["lang_fallback"] = lang_tag or None

        contest_id = extract_contest_id_from_prob(prob)

        tag = args.tag
        submit_result = api.submit_code(prob=prob, lang=lang_id, code=code, tag=tag)
        submission_id = submit_result.get("submission_id")
        row["submission_id"] = submission_id

        final_status = None
        if submission_id:
            final = poll_until_final(
                api, submission_id=str(submission_id), max_polls=args.max_polls, poll_secs=args.poll_secs
            )
            final_status = final.get("status_canonical") or final.get("status")

        row["final_status"] = final_status

        # For rating/pass-rate
        evaluated += 1
        if final_status == "AC":
            accepted += 1
        if contest_id:
            if final_status is None:
                final_status = "UNKNOWN"
            problem_status_by_contest[str(contest_id)][str(prob)].append(str(final_status))

        submitted_rows.append(row)
        print(f"[{len(submitted_rows)}/{len(lines)}] prob={prob} final={final_status}")

    pass_rate = (accepted / evaluated * 100.0) if evaluated else 0.0
    row0 = {
        "pass_rate": pass_rate,
        "accepted": accepted,
        "evaluated": evaluated,
        "input": str(input_path),
    }
    # Save output
    with out_path.open("w", encoding="utf-8") as f:
        for r in submitted_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.write(json.dumps(row0, ensure_ascii=False) + "\n")

    print(f"[summary] pass_rate={pass_rate:.2f}% accepted={accepted}/{evaluated}")

    if args.compute_rating and problem_status_by_contest:
        print("[rating] downloading sorted_ratings.json...")
        sorted_ratings = download_json_sorted_ratings(args.sorted_ratings_url)
        ratings: dict[str, Any] = {}
        rating_list: list[float] = []
        for contest_id_str, problem_status in problem_status_by_contest.items():
            cid = int(contest_id_str)
            rating = calc_elo_rating(cid, dict(problem_status), sorted_ratings=sorted_ratings)
            ratings[contest_id_str] = {"rating": rating}
            if rating is not None:
                rating_list.append(rating)
        if rating_list:
            avg_rating = sum(rating_list) / len(rating_list)
            percentile = get_percentile(avg_rating, sorted_ratings)
            ratings["avg"] = {"rating": avg_rating, "percentile": percentile, "n_contests": len(rating_list)}
        print("[rating] done:", json.dumps(ratings, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

