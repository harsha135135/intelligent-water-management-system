from __future__ import annotations

import csv
import io
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests


BASE_URL = "https://api.waltr.in"
DEFAULT_LOCATION_ID = "638"
DEFAULT_START_DATE = date(2025, 1, 1)


def _normalize_tank_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _headers(token: str) -> dict[str, str]:
    token_clean = token.strip()
    if token_clean.lower().startswith("jwt "):
        token_clean = token_clean[4:]
    return {
        "Authorization": f"JWT {token_clean}",
        "Accept": "*/*",
        "Origin": "https://app.waltr.in",
        "Referer": "https://app.waltr.in/",
    }


def _parse_existing_dates(tank_dir: Path) -> list[date]:
    dates: list[date] = []
    for file_path in tank_dir.glob("*.json"):
        try:
            dates.append(datetime.strptime(file_path.stem, "%Y-%m-%d").date())
        except ValueError:
            continue
    return sorted(dates)


def _parse_csv_flow(csv_text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[dict[str, Any]] = []
    for row in reader:
        if not row:
            continue

        dt = row.get("Date Time") or row.get("timestamp") or row.get("datetime")
        if not dt:
            continue

        def _f(name: str) -> float:
            raw = row.get(name, "")
            try:
                return float(raw)
            except (TypeError, ValueError):
                return 0.0

        rows.append(
            {
                "Date Time": str(dt),
                "Inflow in KL": _f("Inflow in KL"),
                "Outflow in KL": _f("Outflow in KL"),
                "Opening Value in KL": _f("Opening Value in KL"),
                "Closing Value in KL": _f("Closing Value in KL"),
            }
        )
    return rows


def _build_daily_json_payload(
    tank_meta: dict[str, Any],
    flow_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_inflow = sum(float(r.get("Inflow in KL", 0.0)) for r in flow_rows)
    total_outflow = sum(float(r.get("Outflow in KL", 0.0)) for r in flow_rows)

    return {
        "Report Type": "daily",
        "Tank Name": tank_meta.get("name", "Unknown Tank"),
        "Tank Type": tank_meta.get("tank_type", "unknown"),
        "Tank Shape": tank_meta.get("tank_shape", "unknown"),
        "Tank Dimensions": tank_meta.get("tank_dimensions", "unknown"),
        "data": flow_rows,
        "Total": {
            "Inflow in KL": round(total_inflow, 4),
            "Outflow in KL": round(total_outflow, 4),
        },
    }


@dataclass
class TankSyncResult:
    tank_name: str
    tank_id: str
    dataset_dir: str
    start_date: str
    end_date: str
    downloaded: int
    skipped_existing: int
    no_data_days: int
    errors: int
    downloaded_files_sample: list[str] = field(default_factory=list)



def sync_waltr_dataset_incremental(
    token: str,
    dataset_dir: Path,
    location_id: str | None = DEFAULT_LOCATION_ID,
    start_date: date = DEFAULT_START_DATE,
    end_date: date | None = None,
    sleep_seconds: float = 0.05,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    if end_date is None:
        end_date = date.today()

    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    location_id_str = str(location_id or DEFAULT_LOCATION_ID).strip()
    if not location_id_str:
        raise ValueError("location_id is empty after defaulting")

    tanks_url = f"{BASE_URL}/v1/location/{location_id_str}/tanks"
    tanks_resp = requests.get(tanks_url, headers=_headers(token), timeout=timeout_seconds)
    if tanks_resp.status_code == 404:
        raise ValueError(f"Location not found in Waltr: {location_id_str}")
    if tanks_resp.status_code == 401:
        raise PermissionError("Waltr token is unauthorized/expired. Refresh the token and retry.")
    if tanks_resp.status_code == 403:
        raise PermissionError(
            "Waltr access forbidden (403). Token may be valid but lacks access to this location."
        )
    tanks_resp.raise_for_status()

    tanks_body = tanks_resp.json()
    if isinstance(tanks_body, dict) and "data" in tanks_body:
        all_tanks = tanks_body["data"]
    elif isinstance(tanks_body, list):
        all_tanks = tanks_body
    else:
        all_tanks = []

    flow_tanks = [t for t in all_tanks if t.get("show_flow_page", False)]

    # Map Waltr tank names to existing dataset tank directories using a normalized key.
    existing_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    dir_map: dict[str, Path] = {_normalize_tank_key(p.name): p for p in existing_dirs}

    results: list[TankSyncResult] = []
    total_downloaded = 0
    total_skipped = 0
    total_no_data = 0
    total_errors = 0
    downloaded_files_sample: list[str] = []
    max_downloaded_files_sample = 200

    for tank in flow_tanks:
        tank_name = str(tank.get("name", "UNKNOWN_TANK"))
        tank_id = str(tank.get("id", "")).strip()
        if not tank_id:
            total_errors += 1
            continue
        key = _normalize_tank_key(tank_name)

        tank_dir = dir_map.get(key)
        if tank_dir is None:
            tank_dir = dataset_dir / key
            tank_dir.mkdir(parents=True, exist_ok=True)
            dir_map[key] = tank_dir

        existing_dates = _parse_existing_dates(tank_dir)
        tank_start = (existing_dates[-1] + timedelta(days=1)) if existing_dates else start_date
        if tank_start > end_date:
            results.append(
                TankSyncResult(
                    tank_name=tank_name,
                    tank_id=tank_id,
                    dataset_dir=tank_dir.name,
                    start_date=str(tank_start),
                    end_date=str(end_date),
                    downloaded=0,
                    skipped_existing=0,
                    no_data_days=0,
                    errors=0,
                )
            )
            continue

        downloaded = 0
        tank_downloaded_files_sample: list[str] = []
        skipped_existing = 0
        no_data_days = 0
        errors = 0

        current = tank_start
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            out_file = tank_dir / f"{date_str}.json"

            if out_file.exists() and out_file.stat().st_size > 10:
                skipped_existing += 1
                current += timedelta(days=1)
                continue

            daily_url = f"{BASE_URL}/v1/tank/{tank_id}/flow/daily/csv/{date_str}"
            try:
                resp = requests.get(daily_url, headers=_headers(token), timeout=timeout_seconds)
            except requests.RequestException:
                errors += 1
                current += timedelta(days=1)
                time.sleep(sleep_seconds)
                continue

            if resp.status_code == 404:
                no_data_days += 1
                current += timedelta(days=1)
                time.sleep(sleep_seconds)
                continue

            if resp.status_code == 401:
                raise PermissionError("Waltr token is unauthorized/expired. Refresh the token and retry.")

            try:
                resp.raise_for_status()
            except requests.HTTPError:
                errors += 1
                current += timedelta(days=1)
                time.sleep(sleep_seconds)
                continue

            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "json" in content_type:
                body = resp.json()
                if isinstance(body, dict) and "data" in body:
                    payload = body
                else:
                    rows = body if isinstance(body, list) else []
                    payload = _build_daily_json_payload(tank, rows)
            else:
                rows = _parse_csv_flow(resp.text)
                payload = _build_daily_json_payload(tank, rows)

            out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            downloaded += 1
            rel_path = str(out_file.relative_to(dataset_dir))
            if len(tank_downloaded_files_sample) < 20:
                tank_downloaded_files_sample.append(rel_path)
            if len(downloaded_files_sample) < max_downloaded_files_sample:
                downloaded_files_sample.append(rel_path)
            current += timedelta(days=1)
            time.sleep(sleep_seconds)

        results.append(
            TankSyncResult(
                tank_name=tank_name,
                tank_id=tank_id,
                dataset_dir=tank_dir.name,
                start_date=str(tank_start),
                end_date=str(end_date),
                downloaded=downloaded,
                downloaded_files_sample=tank_downloaded_files_sample,
                skipped_existing=skipped_existing,
                no_data_days=no_data_days,
                errors=errors,
            )
        )

        total_downloaded += downloaded
        total_skipped += skipped_existing
        total_no_data += no_data_days
        total_errors += errors

    return {
        "location_id": location_id_str,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "tanks_total": len(flow_tanks),
        "downloaded_total": total_downloaded,
        "downloaded_files_sample": downloaded_files_sample,
        "skipped_existing_total": total_skipped,
        "no_data_days_total": total_no_data,
        "errors_total": total_errors,
        "per_tank": [r.__dict__ for r in results],
    }
