"""Build the small bundled dataset used by the deployed dashboard.

Reads your yearly NASA FIRMS CSV files (whole-India files are fine), keeps only vegetation-fire
detections of non-low confidence that fall inside Chhattisgarh, tags each one with its district,
and writes a single compressed file that is small enough for GitHub.

Usage (PowerShell, from this folder):
    py prepare_data.py --csv-dir "C:\\path\\to\\your\\csv\\files" --boundary "C:\\path\\to\\Chhattisgarh.geojson"

Output:
    data/cg_fires.csv.gz
    data/Chhattisgarh.geojson
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import pandas as pd

import fire_core as fc

KEEP = {"latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type", "daynight"}
OUT_COLS = ["latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "daynight", "district"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True, help="Folder containing the yearly FIRMS CSV files")
    ap.add_argument("--pattern", default="*.csv", help="File name pattern (default *.csv)")
    ap.add_argument("--boundary", required=True, help="Chhattisgarh district boundary GeoJSON")
    ap.add_argument("--out", default="data", help="Output folder (default: data)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.csv_dir, args.pattern)))
    if not files:
        raise SystemExit(f"No files matching {args.pattern} in {args.csv_dir}")

    with open(args.boundary, encoding="utf-8") as fh:
        geojson = json.load(fh)
    districts, _ = fc.parse_geojson_districts(geojson)
    print(f"Boundary: {len(districts)} districts")

    frames = []
    for path in files:
        part = pd.read_csv(path, usecols=lambda c: c.strip().lower() in KEEP, dtype={"acq_time": str})
        part.columns = [c.strip().lower() for c in part.columns]
        kept = fc.load_fires(part, districts=districts)
        print(f"  {os.path.basename(path)}: {len(part):,} rows -> {len(kept):,} in Chhattisgarh")
        frames.append(kept)

    df = pd.concat(frames, ignore_index=True)
    dedupe = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in df.columns]
    before = len(df)
    df = df.drop_duplicates(subset=dedupe).reset_index(drop=True)
    if before != len(df):
        print(f"Removed {before - len(df):,} duplicate rows")

    for col in OUT_COLS:
        if col not in df.columns:
            df[col] = None
    out = df[OUT_COLS].copy()
    out["latitude"] = out["latitude"].round(4)
    out["longitude"] = out["longitude"].round(4)
    out["acq_date"] = pd.to_datetime(out["acq_date"]).dt.strftime("%Y-%m-%d")
    out = out.sort_values(["acq_date", "district"]).reset_index(drop=True)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "cg_fires.csv.gz")
    out.to_csv(csv_path, index=False, compression="gzip")
    shutil.copyfile(args.boundary, os.path.join(args.out, "Chhattisgarh.geojson"))

    years = pd.to_datetime(out["acq_date"]).dt.year
    size_mb = os.path.getsize(csv_path) / 1e6
    print(f"\nWrote {csv_path}: {len(out):,} detections, years {years.min()}-{years.max()}, {size_mb:.1f} MB")
    if size_mb > 90:
        print("WARNING: file is over 90 MB; GitHub rejects files over 100 MB.")


if __name__ == "__main__":
    main()
