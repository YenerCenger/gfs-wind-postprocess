"""Hizalanmış veri ve zaman bölmesi.

01-veri.ipynb'de Cengo'nun verdiği kararların (en yakın ölçüm ±15 dk,
kesim 2025-11-01) tekrar kullanılabilir hali. Karar değişirse önce notebook'ta
ölç, sonra burayı güncelle.
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"
KESIM = pd.Timestamp("2025-11-01", tz="UTC")


def load_aligned(station: str = "ltbj") -> pd.DataFrame:
    """GFS saatlerine en yakın METAR ölçümü (±15 dk). Ölçümü olmayan saat atılır.

    Sütunlar: time, *_gfs (tahmin), *_obs (ölçüm), hata = gfs - obs.
    """
    metar = pd.read_parquet(RAW / f"metar_{station}.parquet")
    gfs = pd.read_parquet(RAW / f"gfs_{station}.parquet")
    aligned = pd.merge_asof(
        gfs, metar,
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("15min"),
        suffixes=("_gfs", "_obs"),
    )
    aligned = aligned.dropna(subset=["speed_kn_obs"]).reset_index(drop=True)
    aligned["hata"] = aligned["speed_kn_gfs"] - aligned["speed_kn_obs"]
    return aligned


def split(aligned: pd.DataFrame, kesim: pd.Timestamp = KESIM):
    """Zaman bölmesi: kesimden önce train, sonrası test."""
    return aligned[aligned["time"] < kesim], aligned[aligned["time"] >= kesim]
