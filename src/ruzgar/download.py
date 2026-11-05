"""Ham veriyi indirir: LTBJ METAR ölçümleri ve GFS gün öncesi tahminleri.

Çalıştırma:  uv run python -m ruzgar.download
Çıktı:       data/raw/metar_ltbj.parquet, data/raw/gfs_ltbj.parquet

İkisi de UTC. Hizalama ve temizlik burada YAPILMAZ; o karar notebook'ta.
"""

from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

STATION = "LTBJ"
LAT, LON = 38.2924, 27.1570  # İzmir Adnan Menderes
START = date(2024, 3, 1)  # Previous Runs API'de GFS gün öncesi bundan önce boş
END = date.today() - timedelta(days=2)

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"


def year_chunks(start: date, end: date):
    """Aralığı yıllık parçalara böler; iki API de dev istekleri sevmiyor."""
    cur = start
    while cur <= end:
        chunk_end = min(date(cur.year, 12, 31), end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def fetch_metar(start: date, end: date) -> pd.DataFrame:
    # IEM'de bitiş günü dahil değil, o yüzden +1 gün
    end_excl = end + timedelta(days=1)
    params = {
        "station": STATION,
        "data": ["sknt", "drct", "gust"],
        "year1": start.year, "month1": start.month, "day1": start.day,
        "year2": end_excl.year, "month2": end_excl.month, "day2": end_excl.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "M",
        "report_type": [3, 4],  # 3: rutin METAR, 4: SPECI (ara rapor)
    }
    r = requests.get(
        "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",
        params=params, timeout=300,
    )
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), na_values="M")
    df["valid"] = pd.to_datetime(df["valid"], utc=True)
    return df.rename(columns={"valid": "time", "sknt": "speed_kn", "drct": "dir_deg", "gust": "gust_kn"})


def fetch_gfs(start: date, end: date) -> pd.DataFrame:
    # previous_day1 = hedef saatten ~24 saat önce yapılmış koşunun tahmini
    variables = [
        "wind_speed_10m_previous_day1",
        "wind_direction_10m_previous_day1",
        "wind_gusts_10m_previous_day1",
    ]
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(variables),
        "models": "gfs_seamless",
        "wind_speed_unit": "kn",
        "timezone": "GMT",
    }
    r = requests.get(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        params=params, timeout=300,
    )
    r.raise_for_status()
    hourly = r.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.rename(columns={
        "wind_speed_10m_previous_day1": "speed_kn",
        "wind_direction_10m_previous_day1": "dir_deg",
        "wind_gusts_10m_previous_day1": "gust_kn",
    })


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, fetch in [("metar", fetch_metar), ("gfs", fetch_gfs)]:
        parts = []
        for s, e in year_chunks(START, END):
            print(f"{name}: {s} → {e}")
            parts.append(fetch(s, e))
        df = pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
        out = RAW / f"{name}_ltbj.parquet"
        df.to_parquet(out, index=False)
        print(f"  {len(df)} satır → {out.name}")


if __name__ == "__main__":
    main()
