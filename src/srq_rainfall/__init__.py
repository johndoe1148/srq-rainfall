"""Print a rainfall report for Sarasota County Water Atlas gauges.

The 1-, 2-, and 8-hour figures are calculated by summing the timestamped
precipitation increments returned by the Data Mapper graph API.  The 24-hour
and 7-day figures come directly from the Water Atlas rainfall summary API.
NWS 24/48/72-hour columns are National Weather Service quantitative precipitation
forecasts at each gauge's Water Atlas coordinates.

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
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

import requests

API_BASE = "https://api.wateratlas.usf.edu"
DEFAULT_SITE_ID = 8  # Sarasota County Water Atlas
RAINFALL_PARAMETER = "Rainfall_IN"
LOOKBACK_HOURS = (1, 2, 8)
FORECAST_HOURS = (24, 48, 72)
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
MAX_WORKERS = 8
WATER_ATLAS_HOME = "https://www.sarasota.wateratlas.usf.edu/"
EASTERN = ZoneInfo("America/New_York")
NWS_API_BASE = "https://api.weather.gov"
NWS_USER_AGENT = "srq-rainfall (suntzu1079@gmail.com)"
NWS_HEADERS = {
    "User-Agent": NWS_USER_AGENT,
    "Accept": "application/geo+json",
}
MM_PER_INCH = 25.4
ISO8601_DURATION = re.compile(
    r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?"
)

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
    fcst_24h_in: float | None
    fcst_48h_in: float | None
    fcst_72h_in: float | None
    rain_1h_in: float | None
    rain_2h_in: float | None
    rain_8h_in: float | None
    rain_24h_in: float | None
    rain_7d_in: float | None
    last_updated: str | None
    station_url: str | None
    datasource_id: str | None


def get_json(
    url: str,
    *,
    params: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
) -> object:
    """Fetch JSON with short retries and a useful final error."""
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=request_headers,
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
        fcst_24h_in=None,
        fcst_48h_in=None,
        fcst_72h_in=None,
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
        stations = list(executor.map(build_station, raw_stations))
    attach_nws_forecasts(stations)
    return stations


def parse_iso8601_duration(duration: str) -> timedelta:
    match = ISO8601_DURATION.fullmatch(duration)
    if not match:
        raise ValueError(f"unsupported ISO 8601 duration: {duration}")
    days, hours, minutes, seconds = match.groups()
    return timedelta(
        days=int(days or 0),
        hours=int(hours or 0),
        minutes=int(minutes or 0),
        seconds=float(seconds or 0),
    )


def parse_valid_time(valid_time: str) -> tuple[datetime, timedelta]:
    start_s, duration_s = valid_time.split("/", 1)
    start = datetime.fromisoformat(start_s.replace("Z", "+00:00"))
    return start, parse_iso8601_duration(duration_s)


def qpf_to_inches(amount_mm_or_in: float, uom: str | None) -> float:
    unit = (uom or "").lower()
    if "in" in unit and "min" not in unit:
        return amount_mm_or_in
    return amount_mm_or_in / MM_PER_INCH


def sum_qpf_windows(values: list[dict], *, now: datetime, uom: str | None) -> dict[int, float]:
    totals = {hours: 0.0 for hours in FORECAST_HOURS}
    for item in values:
        if not isinstance(item, dict) or item.get("value") is None or not item.get("validTime"):
            continue
        start, duration = parse_valid_time(str(item["validTime"]))
        duration_seconds = duration.total_seconds()
        if duration_seconds <= 0:
            continue
        end = start + duration
        inches = qpf_to_inches(float(item["value"]), uom)
        for hours in FORECAST_HOURS:
            window_end = now + timedelta(hours=hours)
            overlap = (min(end, window_end) - max(start, now)).total_seconds()
            if overlap > 0:
                totals[hours] += inches * (overlap / duration_seconds)
    return {hours: round(total, 2) for hours, total in totals.items()}


def fetch_nws_qpf_inches(
    lat: float,
    lon: float,
    *,
    grid_cache: dict[str, dict[int, float]],
    grid_lock: Lock,
) -> dict[int, float]:
    points = get_json(f"{NWS_API_BASE}/points/{lat:.4f},{lon:.4f}", headers=NWS_HEADERS)
    if not isinstance(points, dict):
        raise RuntimeError("unexpected NWS points response")
    properties = points.get("properties") or {}
    grid_url = properties.get("forecastGridData")
    if not grid_url:
        raise RuntimeError("NWS points response missing forecastGridData")
    grid_url = str(grid_url)
    with grid_lock:
        cached = grid_cache.get(grid_url)
    if cached is not None:
        return cached
    grid = get_json(grid_url, headers=NWS_HEADERS)
    if not isinstance(grid, dict):
        raise RuntimeError("unexpected NWS gridpoint response")
    qpf = (grid.get("properties") or {}).get("quantitativePrecipitation") or {}
    values = qpf.get("values")
    if not isinstance(values, list) or not values:
        raise RuntimeError("NWS gridpoint response missing quantitativePrecipitation")
    now = datetime.now(timezone.utc)
    totals = sum_qpf_windows(values, now=now, uom=qpf.get("uom"))
    with grid_lock:
        grid_cache[grid_url] = totals
    return totals


def _apply_nws_forecast(station: MorningRainfall, grid_cache: dict[str, dict[int, float]], grid_lock: Lock) -> None:
    if station.latitude is None or station.longitude is None:
        print(f"Warning: no coordinates for station {station.station_id}; skipping NWS forecast", file=sys.stderr)
        return
    try:
        totals = fetch_nws_qpf_inches(
            float(station.latitude),
            float(station.longitude),
            grid_cache=grid_cache,
            grid_lock=grid_lock,
        )
    except (RuntimeError, ValueError, TypeError) as exc:
        print(f"Warning: NWS forecast unavailable for {station.station_id}: {exc}", file=sys.stderr)
        return
    station.fcst_24h_in = totals[24]
    station.fcst_48h_in = totals[48]
    station.fcst_72h_in = totals[72]


def attach_nws_forecasts(stations: list[MorningRainfall]) -> None:
    grid_cache: dict[str, dict[int, float]] = {}
    grid_lock = Lock()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        list(executor.map(lambda station: _apply_nws_forecast(station, grid_cache, grid_lock), stations))


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
        "1-, 2-, and 8-hour totals are summed precipitation increments from the Data Mapper graph API, anchored to each gauge's newest sample. 24-hour and 7-day totals come from the Water Atlas rainfall summary. NWS 24h/48h/72h columns are National Weather Service quantitative precipitation forecasts at each gauge.",
        "",
    ]
    if not shown:
        lines.append("No stations to display.")
        lines.append("")
        return "\n".join(lines)

    lines.extend(
        [
            "| Station | Check area | 24h QPF | 48h QPF | 72h QPF | 1h | 2h | 8h | 24h | 7d | Last updated |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for station in shown:
        lines.append(
            "| "
            + " | ".join(
                [
                    _station_label(station),
                    station.check_area or "n/a",
                    number(station.fcst_24h_in),
                    number(station.fcst_48h_in),
                    number(station.fcst_72h_in),
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

    header = (
        f"{'Station':25} {'Check area':38} {'24h QPF':>6} {'48h QPF':>6} {'72h QPF':>6} "
        f"{'1h':>6} {'2h':>6} {'8h':>6} {'24h':>6} {'7d':>6}"
    )
    print(header)
    print("-" * len(header))
    for station in shown:
        print(
            f"{station.station_name[:25]:25} {(station.check_area or 'n/a')[:38]:38} "
            f"{number(station.fcst_24h_in):>6} {number(station.fcst_48h_in):>6} {number(station.fcst_72h_in):>6} "
            f"{number(station.rain_1h_in):>6} "
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
