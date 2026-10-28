"""Core logic for the Chhattisgarh forest fire risk dashboard.

Input: a NASA FIRMS active-fire CSV (VIIRS or MODIS).
Required columns: latitude, longitude, acq_date
Optional columns: confidence, frp, daynight
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

# Approximate bounding box of Chhattisgarh (lat/lon). Coarse filter only:
# it also catches small parts of neighbouring states. Use a district
# boundary file for an exact clip.
CG_BOUNDS = dict(lat_min=17.78, lat_max=24.10, lon_min=80.25, lon_max=84.40)
GRID_DEG = 0.25  # grid cell size in degrees (~27 km)


def parse_geojson_rings(geojson: dict, name_filter: str | None = None):
    """Extract polygon rings (as Nx2 arrays of lon, lat) from a GeoJSON dict.

    name_filter: if given, only features with a property value containing this
    text (case-insensitive) are used. Returns (rings, n_features_used).
    """
    if geojson.get("type") == "FeatureCollection":
        feats = geojson.get("features", [])
    elif geojson.get("type") == "Feature":
        feats = [geojson]
    else:
        feats = [{"type": "Feature", "properties": {}, "geometry": geojson}]

    rings, used = [], 0
    for f in feats:
        if name_filter:
            props = " ".join(str(v) for v in (f.get("properties") or {}).values()).lower()
            if name_filter.lower() not in props:
                continue
        geom = f.get("geometry") or {}
        if geom.get("type") == "Polygon":
            polys = [geom["coordinates"]]
        elif geom.get("type") == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        used += 1
        for poly in polys:
            for ring in poly:
                rings.append(np.asarray(ring, dtype=float)[:, :2])
    return rings, used


def points_in_rings(lons: np.ndarray, lats: np.ndarray, rings) -> np.ndarray:
    """Vectorised even-odd ray casting; works for polygons with holes."""
    inside = np.zeros(len(lons), dtype=bool)
    for ring in rings:
        x1, y1 = ring[:, 0], ring[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        m = (lons >= x1.min()) & (lons <= x1.max()) & (lats >= y1.min()) & (lats <= y1.max())
        idx = np.where(m)[0]
        if len(idx) == 0:
            continue
        for chunk in np.array_split(idx, max(1, len(idx) // 4000)):
            px = lons[chunk][:, None]
            py = lats[chunk][:, None]
            cond = ((y1 > py) != (y2 > py)) & (
                px < (x2 - x1) * (py - y1) / np.where(y2 == y1, 1e-300, y2 - y1) + x1
            )
            inside[chunk] ^= (cond.sum(axis=1) % 2 == 1)
    return inside


def load_fires(df: pd.DataFrame, rings=None, districts=None) -> pd.DataFrame:
    """Clean a FIRMS dataframe and clip to Chhattisgarh.

    Always applies the bounding box. If `districts` ({name: rings}) is given, detections
    are clipped to the union of the districts and tagged with a `district` column.
    If only `rings` is given, detections are clipped to those polygons."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"latitude", "longitude", "acq_date"} - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date", "latitude", "longitude"])

    b = CG_BOUNDS
    df = df[
        df["latitude"].between(b["lat_min"], b["lat_max"])
        & df["longitude"].between(b["lon_min"], b["lon_max"])
    ].copy()

    # Keep vegetation fires only: FIRMS 'type' 0 = vegetation fire,
    # 2 = other static land source (industry, e.g. steel plants), 3 = offshore.
    if "type" in df.columns:
        t = pd.to_numeric(df["type"], errors="coerce")
        df = df[t.isna() | (t == 0)].copy()

    # Drop low-confidence detections. VIIRS uses l/n/h; MODIS uses 0-100.
    if "confidence" in df.columns:
        conf = df["confidence"].astype(str).str.lower().str.strip()
        is_low = conf.isin(["l", "low"])
        numeric = pd.to_numeric(df["confidence"], errors="coerce")
        is_low |= numeric.notna() & (numeric < 30)
        df = df[~is_low].copy()

    lon = df["longitude"].to_numpy(float)
    lat = df["latitude"].to_numpy(float)
    if districts:
        label = np.full(len(df), None, dtype=object)
        for name, rr in districts.items():
            todo = np.where(pd.isna(label))[0]
            if len(todo) == 0:
                break
            hit = points_in_rings(lon[todo], lat[todo], rr)
            label[todo[hit]] = name
        keep = pd.notna(label)
        df = df[keep].copy()
        df["district"] = label[keep]
    elif rings:
        df = df[points_in_rings(lon, lat, rings)].copy()

    df["year"] = df["acq_date"].dt.year
    df["month"] = df["acq_date"].dt.month
    df = add_grid(df)
    return df.reset_index(drop=True)


def add_grid(df: pd.DataFrame) -> pd.DataFrame:
    b = CG_BOUNDS
    df = df.copy()
    df["gy"] = ((df["latitude"] - b["lat_min"]) // GRID_DEG).astype(int)
    df["gx"] = ((df["longitude"] - b["lon_min"]) // GRID_DEG).astype(int)
    df["cell"] = df["gy"].astype(str) + "_" + df["gx"].astype(str)
    df["cell_lat"] = b["lat_min"] + (df["gy"] + 0.5) * GRID_DEG
    df["cell_lon"] = b["lon_min"] + (df["gx"] + 0.5) * GRID_DEG
    return df


def monthly_counts(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["year", "month"]).size().rename("fires").reset_index()


def cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["cell", "cell_lat", "cell_lon"]).agg(fires=("cell", "size")).reset_index()
    if "frp" in df.columns:
        frp = df.groupby("cell")["frp"].mean().rename("mean_frp")
        g = g.merge(frp, on="cell", how="left")
    return g.sort_values("fires", ascending=False).reset_index(drop=True)


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Cell x month x year panel with fire counts and leak-free features.

    Every feature for (cell, t) only uses information from before t.
    """
    b = CG_BOUNDS
    ny = int((b["lat_max"] - b["lat_min"]) // GRID_DEG) + 1
    nx = int((b["lon_max"] - b["lon_min"]) // GRID_DEG) + 1
    years = np.arange(df["year"].min(), df["year"].max() + 1)

    idx = pd.MultiIndex.from_product(
        [range(ny), range(nx), years, range(1, 13)], names=["gy", "gx", "year", "month"]
    )
    counts = df.groupby(["gy", "gx", "year", "month"]).size().rename("fires")
    panel = counts.reindex(idx, fill_value=0).reset_index()

    # Drop the (partial) first and last months of the data window if incomplete.
    first = df["acq_date"].min()
    last = df["acq_date"].max()
    panel["t"] = panel["year"] * 12 + panel["month"]
    t_first = first.year * 12 + first.month + (1 if first.day > 1 else 0)
    t_last = last.year * 12 + last.month - (1 if last.day < last.days_in_month else 0)
    panel = panel[(panel["t"] >= t_first) & (panel["t"] <= t_last)].copy()

    panel["cell"] = panel["gy"].astype(str) + "_" + panel["gx"].astype(str)
    panel = panel.sort_values(["cell", "t"]).reset_index(drop=True)

    g = panel.groupby("cell")["fires"]
    panel["prev_month"] = g.shift(1)
    panel["prior_total"] = g.cumsum() - panel["fires"]

    gm = panel.groupby(["cell", "month"])["fires"]
    prior_same_sum = gm.cumsum() - panel["fires"]
    prior_same_n = gm.cumcount()
    panel["same_month_mean"] = np.where(prior_same_n > 0, prior_same_sum / prior_same_n.clip(lower=1), np.nan)

    panel["lat"] = b["lat_min"] + (panel["gy"] + 0.5) * GRID_DEG
    panel["lon"] = b["lon_min"] + (panel["gx"] + 0.5) * GRID_DEG
    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12)
    panel["fire"] = (panel["fires"] > 0).astype(int)
    return panel


FEATURES = ["lat", "lon", "month_sin", "month_cos", "prev_month", "prior_total", "same_month_mean"]


def train_and_evaluate(panel: pd.DataFrame, test_year: int | None = None, min_fires: int = 1):
    """Train on years before test_year, test on test_year (time-based split).

    A cell-month counts as a fire event when it has at least `min_fires` detections."""
    panel = panel.assign(fire=(panel["fires"] >= min_fires).astype(int))
    years = sorted(panel["year"].unique())
    if len(years) < 3:
        raise ValueError("Need at least 3 years of data to train and test a model.")
    if test_year is None:
        test_year = years[-1]

    train = panel[panel["year"] < test_year].dropna(subset=["prev_month"])
    test = panel[panel["year"] == test_year].dropna(subset=["prev_month"])
    if train["fire"].nunique() < 2 or test["fire"].nunique() < 2:
        raise ValueError("Not enough variation in the data to evaluate the model.")

    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, random_state=42)
    model.fit(train[FEATURES], train["fire"])

    proba = model.predict_proba(test[FEATURES])[:, 1]
    baseline = test["same_month_mean"].fillna(0).values

    metrics = {
        "test_year": int(test_year),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "fire_rate_test": float(test["fire"].mean()),
        "auc_model": float(roc_auc_score(test["fire"], proba)),
        "auc_baseline": float(roc_auc_score(test["fire"], baseline)),
        "ap_model": float(average_precision_score(test["fire"], proba)),
        "ap_baseline": float(average_precision_score(test["fire"], baseline)),
    }
    test = test.assign(risk=proba)
    return model, metrics, test


def permutation_importance_table(model, test: pd.DataFrame, n_repeats: int = 3) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        model, test[FEATURES], test["fire"], scoring="roc_auc",
        n_repeats=n_repeats, random_state=0,
    )
    return (
        pd.DataFrame({"feature": FEATURES, "importance": r.importances_mean})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def risk_map_for_month(model, panel: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    sub = panel[(panel["year"] == year) & (panel["month"] == month)].dropna(subset=["prev_month"])
    out = sub[["cell", "lat", "lon"]].copy()
    out["risk"] = model.predict_proba(sub[FEATURES])[:, 1]
    return out


def make_demo_data(seed: int = 0, years=range(2018, 2025)) -> pd.DataFrame:
    """Synthetic data with a Feb-May fire season. NOT real data - for demo/testing only."""
    rng = np.random.default_rng(seed)
    b = CG_BOUNDS
    hot = [(19.0, 81.5), (21.2, 82.0), (22.5, 83.3), (20.0, 82.2)]
    rows = []
    for y in years:
        for m in range(1, 13):
            season = {2: 0.5, 3: 1.0, 4: 0.8, 5: 0.4}.get(m, 0.03)
            n = int(1500 * season * rng.uniform(0.7, 1.3))
            for _ in range(n):
                c = hot[rng.integers(len(hot))]
                lat = np.clip(rng.normal(c[0], 0.8), b["lat_min"], b["lat_max"])
                lon = np.clip(rng.normal(c[1], 0.8), b["lon_min"], b["lon_max"])
                day = int(rng.integers(1, 28))
                rows.append((lat, lon, f"{y}-{m:02d}-{day:02d}", "n", float(rng.gamma(2, 4))))
    return pd.DataFrame(rows, columns=["latitude", "longitude", "acq_date", "confidence", "frp"])


# ---------------------------------------------------------------------------
# District level
# ---------------------------------------------------------------------------
_NAME_KEYS = ["dist_name", "district", "dtname", "district_name", "name_2", "dist_nm", "dist", "name"]


def parse_geojson_districts(geojson: dict, name_filter: str | None = None):
    """Return ({district name: rings}, geojson for plotting with property `_district`)."""
    if geojson.get("type") == "FeatureCollection":
        feats = geojson.get("features", [])
    elif geojson.get("type") == "Feature":
        feats = [geojson]
    else:
        feats = [{"type": "Feature", "properties": {}, "geometry": geojson}]

    districts, out = {}, []
    for i, f in enumerate(feats):
        props = f.get("properties") or {}
        if name_filter:
            blob = " ".join(str(v) for v in props.values()).lower()
            if name_filter.lower() not in blob:
                continue
        geom = f.get("geometry") or {}
        if geom.get("type") == "Polygon":
            polys = [geom["coordinates"]]
        elif geom.get("type") == "MultiPolygon":
            polys = geom["coordinates"]
        else:
            continue
        lower = {str(k).lower(): v for k, v in props.items()}
        name = next((str(lower[c]).strip() for c in _NAME_KEYS if lower.get(c) not in (None, "")),
                    f"District {i + 1}")
        rings = [np.asarray(r, dtype=float)[:, :2] for poly in polys for r in poly]
        districts.setdefault(name, []).extend(rings)
        out.append({"type": "Feature", "properties": {"_district": name}, "geometry": geom})
    return districts, {"type": "FeatureCollection", "features": out}


def district_counts(df: pd.DataFrame, district_names) -> pd.DataFrame:
    c = df.groupby("district").size().reindex(list(district_names), fill_value=0)
    return c.rename("fires").rename_axis("district").reset_index().sort_values("fires", ascending=False)


def build_district_panel(df: pd.DataFrame, district_names) -> pd.DataFrame:
    """District x month panel with leak-free features (only past information)."""
    years = np.arange(df["year"].min(), df["year"].max() + 1)
    idx = pd.MultiIndex.from_product([list(district_names), years, range(1, 13)],
                                     names=["district", "year", "month"])
    counts = df.groupby(["district", "year", "month"]).size().rename("fires")
    p = counts.reindex(idx, fill_value=0).reset_index()

    first, last = df["acq_date"].min(), df["acq_date"].max()
    p["t"] = p["year"] * 12 + p["month"]
    t_first = first.year * 12 + first.month + (1 if first.day > 1 else 0)
    t_last = last.year * 12 + last.month - (1 if last.day < last.days_in_month else 0)
    p = p[(p["t"] >= t_first) & (p["t"] <= t_last)].copy()
    p = p.sort_values(["district", "t"]).reset_index(drop=True)

    g = p.groupby("district")["fires"]
    p["prev_month"] = g.shift(1)
    p["prior_total"] = g.cumsum() - p["fires"]
    n_prior = p.groupby("district").cumcount()
    p["prior_avg"] = np.where(n_prior > 0, p["prior_total"] / n_prior.clip(lower=1), np.nan)

    gm = p.groupby(["district", "month"])["fires"]
    same_sum = gm.cumsum() - p["fires"]
    same_n = gm.cumcount()
    p["same_month_mean"] = np.where(same_n > 0, same_sum / same_n.clip(lower=1), np.nan)

    p["last_year_same_month"] = g.shift(12)
    p["log_last_year"] = np.log1p(p["last_year_same_month"])

    p["month_sin"] = np.sin(2 * np.pi * p["month"] / 12)
    p["month_cos"] = np.cos(2 * np.pi * p["month"] / 12)
    p["log_prev"] = np.log1p(p["prev_month"])
    p["log_prior_avg"] = np.log1p(p["prior_avg"])
    p["log_same_month"] = np.log1p(p["same_month_mean"])
    return p


D_FEATURES = ["month_sin", "month_cos", "log_prev", "log_prior_avg", "log_same_month", "log_last_year"]


def _top_k_overlap(test: pd.DataFrame, score_col: str, k: int = 5) -> float:
    """Mean share of the true top-k fire districts that a score ranks in its top k (per month)."""
    vals = []
    for _, g in test.groupby("month"):
        if (g["fires"] > 0).sum() < k:
            continue
        actual = set(g.nlargest(k, "fires")["district"])
        pred = set(g.nlargest(k, score_col)["district"])
        vals.append(len(actual & pred) / k)
    return float(np.mean(vals)) if vals else float("nan")


def _district_candidates():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import PoissonRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "gradient boosting (Poisson loss)": lambda: HistGradientBoostingRegressor(
            loss="poisson", max_depth=3, learning_rate=0.06, max_iter=200,
            min_samples_leaf=10, random_state=42),
        "Poisson regression": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            PoissonRegressor(alpha=1.0, max_iter=2000)),
    }


def train_district_model(dpanel: pd.DataFrame, test_year: int | None = None):
    """Forecast monthly fire counts per district (count models with Poisson loss).

    Train before test_year, test on test_year. The model type is chosen on the year
    before the test year (a validation year), never on the test year itself.
    """
    years = sorted(dpanel["year"].unique())
    if len(years) < 3:
        raise ValueError("Need at least 3 years of data to train and test the district model.")
    if test_year is None:
        test_year = years[-1]

    usable = dpanel.dropna(subset=["prev_month"])
    train = usable[usable["year"] < test_year]
    test = usable[usable["year"] == test_year].copy()
    if len(train) < 50 or test.empty:
        raise ValueError("Not enough district-month rows to evaluate the model.")

    candidates = _district_candidates()
    chosen = next(iter(candidates))
    val_scores = {}
    val_year = test_year - 1
    inner = usable[usable["year"] < val_year]
    val = usable[usable["year"] == val_year]
    if len(inner) >= 50 and not val.empty:
        for name, make in candidates.items():
            m = make().fit(inner[D_FEATURES], inner["fires"])
            val_scores[name] = float((np.clip(m.predict(val[D_FEATURES]), 0, None) - val["fires"]).abs().mean())
        chosen = min(val_scores, key=val_scores.get)

    model = candidates[chosen]().fit(train[D_FEATURES], train["fires"])
    test["pred"] = np.clip(model.predict(test[D_FEATURES]), 0, None)
    test["baseline"] = test["same_month_mean"].fillna(0)

    metrics = {
        "test_year": int(test_year),
        "model_name": chosen,
        "val_scores": val_scores,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "spearman_model": float(test["pred"].corr(test["fires"], method="spearman")),
        "spearman_baseline": float(test["baseline"].corr(test["fires"], method="spearman")),
        "mae_model": float((test["pred"] - test["fires"]).abs().mean()),
        "mae_baseline": float((test["baseline"] - test["fires"]).abs().mean()),
        "top5_model": _top_k_overlap(test, "pred"),
        "top5_baseline": _top_k_overlap(test, "baseline"),
        "bias_model": float(test["pred"].sum() / max(test["fires"].sum(), 1)),
        "bias_baseline": float(test["baseline"].sum() / max(test["fires"].sum(), 1)),
    }
    return model, metrics, test


def district_importance_table(model, test: pd.DataFrame, n_repeats: int = 3) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, test[D_FEATURES], test["fires"],
                               scoring="neg_mean_absolute_error", n_repeats=n_repeats, random_state=0)
    return (pd.DataFrame({"feature": D_FEATURES, "importance": r.importances_mean})
            .sort_values("importance", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------------------
# Year-ahead forecast (uses only information from before the target year)
# ---------------------------------------------------------------------------
FC_FEATURES = ["lat", "lon", "month_sin", "month_cos", "same_month_mean",
               "last_year_same_month", "prior_years_total", "last_year_total"]


def build_forecast_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Training rows for the year-ahead model: every feature only uses years before the row's year."""
    p = panel.sort_values(["cell", "t"]).reset_index(drop=True).copy()
    yearly = (p.groupby(["cell", "year"])["fires"].sum().rename("year_total")
              .reset_index().sort_values(["cell", "year"]))
    gy = yearly.groupby("cell")["year_total"]
    yearly["prior_years_total"] = gy.cumsum() - yearly["year_total"]
    yearly["last_year_total"] = gy.shift(1)
    p = p.merge(yearly[["cell", "year", "prior_years_total", "last_year_total"]], on=["cell", "year"], how="left")
    p["last_year_same_month"] = p.groupby("cell")["fires"].shift(12)
    return p


def make_future_frame(panel: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """One row per cell and month of target_year, with features from years before it only."""
    hist = panel[panel["year"] < target_year]
    cells = panel[["cell", "lat", "lon"]].drop_duplicates("cell")
    months = pd.DataFrame({"month": range(1, 13)})
    rows = cells.merge(months, how="cross")
    rows["year"] = target_year

    same = hist.groupby(["cell", "month"])["fires"].mean().rename("same_month_mean").reset_index()
    prev = hist[hist["year"] == target_year - 1][["cell", "month", "fires"]].rename(
        columns={"fires": "last_year_same_month"})
    tot = hist.groupby("cell")["fires"].sum().rename("prior_years_total").reset_index()
    lyt = (hist[hist["year"] == target_year - 1].groupby("cell")["fires"].sum()
           .rename("last_year_total").reset_index())

    out = (rows.merge(same, on=["cell", "month"], how="left")
               .merge(prev, on=["cell", "month"], how="left")
               .merge(tot, on="cell", how="left")
               .merge(lyt, on="cell", how="left"))
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def forecast_year(panel: pd.DataFrame, target_year: int, min_fires: int = 1):
    """Forecast fire-event risk for every cell-month of target_year.

    The model is trained only on years before target_year, so a past year gives an honest backtest
    and a future year gives a genuine forecast. Returns (model, frame, metrics or None).
    """
    fp = build_forecast_panel(panel)
    fp["fire"] = (fp["fires"] >= min_fires).astype(int)
    train = fp[fp["year"] < target_year].dropna(subset=["same_month_mean"])
    if train["year"].nunique() < 2:
        raise ValueError("Need at least 3 years of earlier data to forecast this year.")
    if train["fire"].nunique() < 2:
        raise ValueError("Not enough fire events at this threshold to train. Try a smaller N.")

    model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.06, max_iter=150,
                                           min_samples_leaf=20, random_state=42)
    model.fit(train[FC_FEATURES], train["fire"])

    frame = make_future_frame(panel, target_year)
    frame["risk"] = model.predict_proba(frame[FC_FEATURES])[:, 1]

    obs = panel[panel["year"] == target_year][["cell", "month", "fires"]]
    frame = frame.merge(obs, on=["cell", "month"], how="left")

    metrics = None
    ev = frame.dropna(subset=["fires"]).copy()
    ev = ev[ev["prior_years_total"].fillna(0) >= 0]
    if len(ev):
        ev["fire"] = (ev["fires"] >= min_fires).astype(int)
        if ev["fire"].nunique() == 2:
            base = ev["same_month_mean"].fillna(0)
            metrics = {
                "n": int(len(ev)),
                "auc_model": float(roc_auc_score(ev["fire"], ev["risk"])),
                "auc_baseline": float(roc_auc_score(ev["fire"], base)),
                "ap_model": float(average_precision_score(ev["fire"], ev["risk"])),
                "ap_baseline": float(average_precision_score(ev["fire"], base)),
            }
    return model, frame, metrics
