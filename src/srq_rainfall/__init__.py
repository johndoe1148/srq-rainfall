"""Print a rainfall report for Sarasota County Water Atlas gauges.

The 1-, 2-, and 8-hour figures are calculated by summing the timestamped
precipitation increments returned by the Data Mapper graph API.  The 24-hour
and 7-day figures come directly from the Water Atlas rainfall summary API.

Usage:
    uv run srq-rainfall
    uv run srq-rainfall --all-stations
    uv run srq-rainfall --readme README.md --no-files
    uv run srq-rainfall --format markdown --no-files
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

API_BASE = "https://api.wateratlas.usf.edu"
DEFAULT_SITE_ID = 8  # Sarasota County Water Atlas
RAINFALL_PARAMETER = "Rainfall_IN"
LOOKBACK_HOURS = (1, 2, 8)
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
MAX_WORKERS = 8
WATER_ATLAS_HOME = "https://www.sarasota.wateratlas.usf.edu/"
EASTERN = ZoneInfo("America/New_York")

# The default field-check circuit, in the user's preferred order.
STATION_CHECKS = {
    "416": "Sod farm and surrounding area",
    "501": "Lido Beach",
    "502": "Siesta Key",
    "251": "Route 72 East of MSP",
    "818": "Old Myakka Bridge",
    "580": "Venice Airport and surrounding area",
}


@dataclass
class MorningRainfall:
    station_id: str
    station_name: str
    check_area: str | None
    latitude: float | None
    longitude: float | None
    rain_1h_in: float | None
    rain_2h_in: float | None
    rain_8h_in: float | None
    rain_24h_in: float | None
    rain_7d_in: float | None
    last_updated: str | None
    station_url: str | None
    datasource_id: str | None


def get_json(url: str, *, params: dict[str, object] | None = None) -> object:
    """Fetch JSON with short retries and a useful final error."""
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS)
    raise RuntimeError(f"{url}: {last_error}")


def fetch_latest_rainfall(site_id: int) -> list[dict]:
    data = get_json(f"{API_BASE}/rainfall/latest", params={"s": site_id})
    if not isinstance(data, list):
        raise SystemExit(f"Unexpected rainfall summary response: {type(data)}")
    return data


def value(record: dict, *keys: str, default=None):
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return default


def parse_timestamp(timestamp: str) -> datetime:
    """The API currently sends timezone-less timestamps in local station time.

    We only compare samples from the same station, so a naive datetime is
    intentional and avoids shifting the requested windows.
    """
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(tzinfo=None)


def fetch_recent_totals(datasource: str, station_id: str) -> dict[int, float]:
    """Sum graph-data precipitation increments against the newest sample.

    Anchoring windows to the newest reading, instead of the computer clock,
    makes the report accurate when a gauge has a modest reporting delay.
    """
    endpoint = (
        f"{API_BASE}/DataMapper/Agency/{datasource}/Station/{station_id}"
        f"/Parameter/{RAINFALL_PARAMETER}/GraphData"
    )
    data = get_json(endpoint, params={"numberOfDays": 1, "endDate": ""})
    if not isinstance(data, list) or not data:
        raise RuntimeError("no precipitation graph samples returned")

    samples: list[tuple[datetime, float]] = []
    for item in data:
        if not isinstance(item, dict) or not item.get("sampleDate"):
            continue
        reading = item.get("resultValue")
        if reading is None:
            continue
        samples.append((parse_timestamp(str(item["sampleDate"])), float(reading)))
    if not samples:
        raise RuntimeError("graph response contained no usable precipitation samples")

    samples.sort(key=lambda sample: sample[0])
    newest = samples[-1][0]
    return {
        hours: round(sum(reading for stamp, reading in samples if stamp > newest - timedelta(hours=hours)), 3)
        for hours in LOOKBACK_HOURS
    }


def build_station(record: dict) -> MorningRainfall:
    location = value(record, "location", "Location", default={}) or {}
    datasource = value(record, "datasource", "Datasource", default={}) or {}
    datasource_id = value(datasource, "id", "Id")
    station_id = str(value(record, "id", "Id", default=""))

    recent: dict[int, float] | None = None
    if datasource_id and station_id:
        try:
            recent = fetch_recent_totals(str(datasource_id), station_id)
        except RuntimeError as exc:
            print(f"Warning: unable to get graph data for {station_id}: {exc}", file=sys.stderr)

    return MorningRainfall(
        station_id=station_id,
        station_name=str(value(record, "name", "Name", default="")).strip(),
        check_area=STATION_CHECKS.get(station_id),
        latitude=location.get("latitude"),
        longitude=location.get("longitude"),
        rain_1h_in=recent.get(1) if recent else None,
        rain_2h_in=recent.get(2) if recent else None,
        rain_8h_in=recent.get(8) if recent else None,
        rain_24h_in=_as_float(value(record, "total24h", "Total24h")),
        rain_7d_in=_as_float(value(record, "total7d", "Total7d")),
        last_updated=value(record, "lastUpdated", "LastUpdated"),
        station_url=value(record, "stationUrl", "StationUrl"),
        datasource_id=str(datasource_id) if datasource_id else None,
    )


def _as_float(number: object) -> float | None:
    return float(number) if number is not None else None


def fetch_report(site_id: int, include_all_stations: bool) -> list[MorningRainfall]:
    raw_stations = fetch_latest_rainfall(site_id)
    if not include_all_stations:
        records_by_id = {str(value(record, "id", "Id", default="")): record for record in raw_stations}
        missing = [station_id for station_id in STATION_CHECKS if station_id not in records_by_id]
        if missing:
            print(f"Warning: requested stations not returned by the API: {', '.join(missing)}", file=sys.stderr)
        raw_stations = [records_by_id[station_id] for station_id in STATION_CHECKS if station_id in records_by_id]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # map preserves the selected field-check order while still fetching in parallel.
        return list(executor.map(build_station, raw_stations))


def number(amount: float | None) -> str:
    return f"{amount:.2f}" if amount is not None else "--"


def _station_label(station: MorningRainfall) -> str:
    name = station.station_name or station.station_id or "Unknown station"
    if station.station_url:
        return f"[{name}]({station.station_url})"
    return name


def format_eastern(moment: datetime) -> str:
    """Format a timestamp in Eastern Time with EDT or EST from the date."""
    if moment.tzinfo is None:
        # Water Atlas last-updated stamps are local station time with no offset.
        moment = moment.replace(tzinfo=EASTERN)
    else:
        moment = moment.astimezone(EASTERN)
    return moment.strftime("%Y-%m-%d %H:%M %Z")


def format_last_updated(raw: str | None) -> str:
    if not raw:
        return "--"
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return str(raw)
    return format_eastern(parsed)


def render_markdown(stations: list[MorningRainfall], *, generated_at: datetime, top: int | None) -> str:
    shown = stations
    if top:
        shown = sorted(stations, key=lambda station: station.rain_8h_in or -1, reverse=True)[:top]

    generated = format_eastern(generated_at)
    lines = [
        "# Sarasota County rainfall",
        "",
        f"Updated **{generated}** from the [Sarasota County Water Atlas]({WATER_ATLAS_HOME}).",
        "",
        "1-, 2-, and 8-hour totals are summed precipitation increments from the Data Mapper graph API, anchored to each gauge's newest sample. 24-hour and 7-day totals come from the Water Atlas rainfall summary.",
        "",
    ]
    if not shown:
        lines.append("No stations to display.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Station | Check area | 1h | 2h | 8h | 24h | 7d | Last updated |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for station in shown:
        lines.append(
            "| "
            + " | ".join(
                [
                    _station_label(station),
                    station.check_area or "n/a",
                    number(station.rain_1h_in),
                    number(station.rain_2h_in),
                    number(station.rain_8h_in),
                    number(station.rain_24h_in),
                    number(station.rain_7d_in),
                    format_last_updated(station.last_updated),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "All rainfall amounts are inches.",
            "",
        ]
    )
    return "\n".join(lines)


def print_table(stations: list[MorningRainfall], top: int | None) -> None:
    shown = stations
    if top:
        shown = sorted(stations, key=lambda station: station.rain_8h_in or -1, reverse=True)[:top]
    if not shown:
        print("No stations to display.")
        return

    header = f"{'Station':25} {'Check area':38} {'1h':>6} {'2h':>6} {'8h':>6} {'24h':>6} {'7d':>6}"
    print(header)
    print("-" * len(header))
    for station in shown:
        print(
            f"{station.station_name[:25]:25} {(station.check_area or 'n/a')[:38]:38} {number(station.rain_1h_in):>6} "
            f"{number(station.rain_2h_in):>6} {number(station.rain_8h_in):>6} "
            f"{number(station.rain_24h_in):>6} {number(station.rain_7d_in):>6}"
        )
    print("\nAll rainfall amounts are inches. Short windows are summed graph-data increments.")


def write_outputs(stations: list[MorningRainfall], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"sarasota_morning_rainfall_{timestamp}.json"
    csv_path = out_dir / f"sarasota_morning_rainfall_{timestamp}.csv"
    rows = [asdict(station) for station in stations]
    json_path.write_text(json.dumps(rows, indent=2) + "\n")
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]) if rows else list(MorningRainfall.__annotations__))
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-id", type=int, default=DEFAULT_SITE_ID)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--top", type=int, default=None, help="Only display the wettest N stations in the last 8 hours")
    parser.add_argument("--all-stations", action="store_true", help="Include every Sarasota County Water Atlas gauge")
    parser.add_argument("--no-files", action="store_true", help="Print only; do not write CSV/JSON")
    parser.add_argument(
        "--format",
        choices=("table", "markdown"),
        default="table",
        help="Stdout format when --readme is not set (default: table)",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=None,
        help="Write the full report as markdown to this path (used by GitHub Actions)",
    )
    args = parser.parse_args()

    stations = fetch_report(args.site_id, args.all_stations)
    generated_at = datetime.now(EASTERN)
    markdown = render_markdown(stations, generated_at=generated_at, top=args.top)

    if args.readme:
        args.readme.write_text(markdown)
        print(f"Wrote markdown report to {args.readme}", file=sys.stderr)
    elif args.format == "markdown":
        print(markdown, end="")
    else:
        print_table(stations, args.top)

    if not args.no_files:
        json_path, csv_path = write_outputs(stations, args.out)
        print(f"\nWrote {len(stations)} stations to:\n  {json_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
