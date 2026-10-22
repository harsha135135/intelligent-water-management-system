"""Fetch daily per-tank flow reports from the Waltr API into a dataset directory.

``water_forecast_dash/waltr_sync.py`` only ever moves forward from each tank's last file, so it
cannot backfill history, repair a day that was saved part-way through, or fetch an explicit date
range. This does all three, and writes files in exactly the format of ``dataset/``: the API
response unchanged, ``json.dumps(..., indent=2)``, no trailing newline. A day with no readings
is still written (``"data": []``), as the original scraper did.

Requests use the same endpoints and headers as ``waltr_sync``:

    GET https://api.waltr.in/v1/location/638/tanks                   tank list (show_flow_page)
    GET https://api.waltr.in/v1/tank/{id}/flow/daily/csv/{YYYY-MM-DD}  one day, returned as JSON

The token is a browser session JWT, read from a file and never printed. On a 401/403 the run
stops cleanly; every completed day is already on disk, so re-running resumes where it left off.

    python -m src.data.waltr_fetch --dest dataset_2026-10-22 --range 2026-04-23 2026-10-21
    python -m src.data.waltr_fetch --dest dataset_2026-10-22 --refetch-days 2026-03-16 2026-03-25
    python -m src.data.waltr_fetch --dest dataset_2026-10-22 --backfill
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from .curate import STALE_TANK_DIRS

logger = logging.getLogger(__name__)

BASE_URL = "https://api.waltr.in"
LOCATION_ID = "638"
TOKEN_FILE = Path("~/.config/waltr/token").expanduser()
# Two tanks at a time, 0.15 s apart within each: about 7 requests/s, the order of what the
# repository's own waltr_sync.py sent (0.05 s pacing, one at a time). Waltr's session tokens live
# 15 minutes, and the original scraper's 0.2 s single-thread pace needed a fresh one every
# ~2,800 requests.
WORKERS = 2
PACE_S = 0.15
TIMEOUT_S = 20
RETRIES = 3
BACKFILL_FLOOR = date(2022, 1, 1)
BACKFILL_EMPTY_RUN = 14           # this many consecutive empty days = before the sensor existed


class TokenRejected(RuntimeError):
    pass


STOP = threading.Event()          # set by the first worker whose token is rejected
_LOCK = threading.Lock()
_LOCAL = threading.local()


def tank_key(name: str) -> str:
    """The key ``waltr_sync`` matches tank names to folders by: BOY'S -> BOY_S, I&H -> I_H."""
    return re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")


def read_token(path: Path) -> str:
    token = path.read_text().strip()
    if token.lower().startswith("jwt "):
        token = token[4:].strip()
    if token.lower().startswith("authorization:"):
        raise SystemExit(f"{path} holds a whole header line; keep only the value after 'JWT '")
    if token.count(".") != 2:
        raise SystemExit(f"{path} does not hold a JWT (expected three dot-separated parts)")
    return token


def token_expiry(token: str) -> datetime | None:
    """The JWT's ``exp`` claim, if it has one. The payload is decoded, never verified."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
        return datetime.fromtimestamp(exp) if exp else None
    except Exception:
        return None


class Client:
    def __init__(self, token: str):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"JWT {token}", "Accept": "*/*",
            "Origin": "https://app.waltr.in", "Referer": "https://app.waltr.in/",
            "X-Client-ID": "WaltrAppAgent/Web/3.3.4",       # what the web app sends today
        })
    requests = 0                  # across every worker's client

    def get(self, url: str) -> requests.Response | None:
        """GET with retries. None means 404 (no report for that day)."""
        for attempt in range(1, RETRIES + 1):
            if STOP.is_set():
                raise TokenRejected("stopped: another worker's request was rejected")
            try:
                resp = self.session.get(url, timeout=TIMEOUT_S)
            except requests.RequestException as exc:
                logger.warning("network error (%s), attempt %d/%d", type(exc).__name__, attempt,
                               RETRIES)
                time.sleep(2 * attempt)
                continue
            finally:
                with _LOCK:
                    Client.requests += 1
                time.sleep(PACE_S)
            if resp.status_code in (401, 403):
                STOP.set()
                raise TokenRejected(f"HTTP {resp.status_code} - the session token was rejected "
                                    "or has expired; copy a fresh one and re-run")
            if resp.status_code == 404:
                return None
            if resp.status_code == 200:
                return resp
            logger.warning("HTTP %d for %s, attempt %d/%d", resp.status_code, url.rsplit("/", 2)[-2:],
                           attempt, RETRIES)
            time.sleep(2 * attempt)
        return None

    def tanks(self) -> list[dict]:
        resp = self.get(f"{BASE_URL}/v1/location/{LOCATION_ID}/tanks")
        if resp is None:
            raise SystemExit("could not read the tank list")
        return [t for t in resp.json().get("data", []) if t.get("show_flow_page", False)]

    def day(self, tank_id, day: date) -> dict | None:
        resp = self.get(f"{BASE_URL}/v1/tank/{tank_id}/flow/daily/csv/{day.isoformat()}")
        if resp is None:
            return None
        if "json" not in resp.headers.get("content-type", ""):
            raise RuntimeError(f"tank {tank_id} {day}: expected JSON, got "
                               f"{resp.headers.get('content-type')!r}")
        return resp.json()


def write_day(path: Path, payload: dict) -> None:
    """Exactly the format of ``dataset/``: indent=2, no trailing newline; atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def map_tanks(tanks: list[dict], dest: Path) -> dict[str, dict]:
    """Waltr tank -> the dataset folder it belongs in. Only folders the curation keeps are
    eligible, so BOY'S lands in ``*_BOY_S`` (kept), never the stale ``*_BOYS``."""
    folders = {tank_key(p.name): p.name for p in dest.iterdir()
               if p.is_dir() and p.name not in STALE_TANK_DIRS}
    mapped, unknown = {}, []
    for t in tanks:
        folder = folders.get(tank_key(t["name"]))
        if folder is None:
            unknown.append(t["name"])
            continue
        mapped[folder] = t
    if unknown:
        logger.warning("ignoring Waltr tanks with no dataset folder: %s", unknown)
    missing = sorted(set(folders.values()) - set(mapped))
    if missing:
        logger.warning("dataset folders with no Waltr tank: %s", missing)
    return mapped


def dates(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def is_current_format(path: Path) -> bool:
    """Reports downloaded since Waltr revised its flow figures carry a ``units`` key; the
    earlier downloads do not. That makes a full re-download resumable: a file already replaced
    is never requested twice."""
    return '"units"' in path.read_text()


def _worker_client(token: str) -> Client:
    if not hasattr(_LOCAL, "client"):
        _LOCAL.client = Client(token)
    return _LOCAL.client


def _bump(stats: dict, **counts) -> None:
    with _LOCK:
        for k, v in counts.items():
            stats[k] += v


def per_tank(fn, mapped: dict) -> None:
    """Run ``fn(folder, tank)`` for every tank, ``WORKERS`` at a time. A rejected token stops
    every worker; everything finished before that is already on disk."""
    rejected = None
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = [ex.submit(fn, folder, tank) for folder, tank in sorted(mapped.items())]
        for fut in as_completed(futures):
            try:
                fut.result()
            except TokenRejected as exc:
                rejected = rejected or exc
    if rejected:
        raise rejected


def fetch_range(token: str, mapped: dict, dest: Path, start: date, end: date, *,
                force: bool, stats: dict, replace_old: bool = False) -> None:
    def one(folder: str, tank: dict) -> None:
        client = _worker_client(token)
        for d in dates(start, end):
            path = dest / folder / f"{d.isoformat()}.json"
            if (path.exists() and path.stat().st_size > 10 and not force
                    and not (replace_old and not is_current_format(path))):
                _bump(stats, skipped=1)
                continue
            payload = client.day(tank["id"], d)
            if payload is None:
                _bump(stats, no_report=1)
                continue
            write_day(path, payload)
            _bump(stats, written=1, empty=int(not payload.get("data")))
        logger.info("%-28s %s..%s done  (%d requests so far)", folder, start, end,
                    Client.requests)
    per_tank(one, mapped)


def backfill(token: str, mapped: dict, dest: Path, stats: dict) -> None:
    """Walk back from the day before each tank's earliest file until a run of empty days.

    Empty days inside the series are kept, as in ``dataset/``; the trailing run of empty days
    that ends the walk is not written, since it only says the sensor did not exist yet."""
    def one(folder: str, tank: dict) -> None:
        client = _worker_client(token)
        existing = sorted(p.stem for p in (dest / folder).glob("*.json"))
        d = date.fromisoformat(existing[0]) - timedelta(days=1) if existing else None
        if d is None:
            return
        pending, run, first = [], 0, None
        while d >= BACKFILL_FLOOR and run < BACKFILL_EMPTY_RUN:
            payload = client.day(tank["id"], d)
            if payload is not None and payload.get("data"):
                for pd_, pl in pending:                       # gaps inside the series
                    write_day(dest / folder / f"{pd_.isoformat()}.json", pl)
                    _bump(stats, written=1, empty=1)
                pending, run = [], 0
                write_day(dest / folder / f"{d.isoformat()}.json", payload)
                _bump(stats, written=1)
                first = d
            else:
                run += 1
                if payload is not None:
                    pending.append((d, payload))
            d -= timedelta(days=1)
        with _LOCK:
            stats["backfill_first_day"][folder] = first.isoformat() if first else None
        logger.info("%-28s backfill: earliest reading %s  (%d requests so far)", folder,
                    first or "none before " + existing[0], Client.requests)
    per_tank(one, mapped)


def probe(client: Client, mapped: dict, dest: Path, day: date) -> None:
    """Which daily endpoint still returns the ``dataset/`` format? Prints shapes, never data."""
    folder, tank = sorted(mapped.items())[0]
    on_disk = dest / folder / f"{day.isoformat()}.json"
    for path in ("flow/daily/csv", "flow/daily"):
        resp = client.get(f"{BASE_URL}/v1/tank/{tank['id']}/{path}/{day.isoformat()}")
        if resp is None:
            print(f"{path:16s} -> 404 / no response")
            continue
        ctype = resp.headers.get("content-type", "")
        body = resp.json() if "json" in ctype else None
        keys = list(body) if isinstance(body, dict) else type(body).__name__
        same = (on_disk.exists() and body is not None
                and json.dumps(body, indent=2) == on_disk.read_text())
        print(f"{path:16s} -> {resp.status_code} {ctype}; top-level keys {keys}; "
              f"identical to {folder}/{on_disk.name}: {same}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path, help="dataset directory to write into")
    ap.add_argument("--token-file", type=Path, default=TOKEN_FILE)
    ap.add_argument("--range", nargs=2, metavar=("START", "END"),
                    help="fetch every day in [START, END] that is not already on disk")
    ap.add_argument("--refetch-days", nargs="+", default=[],
                    help="re-download these days for every tank, overwriting")
    ap.add_argument("--refetch", nargs="+", default=[], metavar="FOLDER:YYYY-MM-DD",
                    help="re-download single (tank, day) files, overwriting")
    ap.add_argument("--replace-old", action="store_true",
                    help="with --range: also re-download files saved before Waltr revised its "
                         "flow figures (no 'units' key); resumable")
    ap.add_argument("--backfill", action="store_true",
                    help="probe each tank backwards for history before its earliest file")
    ap.add_argument("--log", type=Path, default=None, help="write a JSON summary here")
    ap.add_argument("--probe", metavar="YYYY-MM-DD",
                    help="request one day for the first tank from both daily endpoints, compare "
                         "with the matching file in --dest, write nothing, and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    token = read_token(args.token_file)
    exp = token_expiry(token)
    if exp:
        left = (exp - datetime.now()).total_seconds() / 60
        logger.info("token expires %s (%.0f min from now)", exp.strftime("%Y-%m-%d %H:%M"), left)
        if left <= 0:
            raise SystemExit("the token has already expired; copy a fresh one")
    client = Client(token)
    stats = {"written": 0, "empty": 0, "skipped": 0, "no_report": 0, "backfill_first_day": {},
             "started": datetime.now().isoformat(timespec="seconds")}
    try:
        mapped = map_tanks(client.tanks(), args.dest)
        logger.info("%d Waltr tanks mapped to dataset folders", len(mapped))
        if args.probe:
            probe(client, mapped, args.dest, date.fromisoformat(args.probe))
            sys.exit(0)
        for d in args.refetch_days:
            fetch_range(token, mapped, args.dest, date.fromisoformat(d), date.fromisoformat(d),
                        force=True, stats=stats)
        for item in args.refetch:
            folder, d = item.split(":")
            fetch_range(token, {folder: mapped[folder]}, args.dest, date.fromisoformat(d),
                        date.fromisoformat(d), force=True, stats=stats)
        if args.range:
            fetch_range(token, mapped, args.dest, date.fromisoformat(args.range[0]),
                        date.fromisoformat(args.range[1]), force=False, stats=stats,
                        replace_old=args.replace_old)
        if args.backfill:
            backfill(token, mapped, args.dest, stats)
    except TokenRejected as exc:
        logger.error("%s", exc)
        stats["stopped"] = str(exc)
        status = 2
    else:
        status = 0
    stats["requests"] = Client.requests
    stats["finished"] = datetime.now().isoformat(timespec="seconds")
    logger.info("summary: %s", {k: v for k, v in stats.items() if k != "backfill_first_day"})
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(stats, indent=2))
    sys.exit(status)


if __name__ == "__main__":
    main()
