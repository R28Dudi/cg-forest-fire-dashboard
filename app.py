from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

import fire_core as fc

st.set_page_config(page_title="Chhattisgarh Forest Fire Risk", layout="wide")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CENTER = {"lat": 21.25, "lon": 82.3}

DATA_DIR = Path(__file__).parent / "data"
BUNDLE = DATA_DIR / "cg_fires.csv.gz"
BUNDLE_BOUNDARY = DATA_DIR / "Chhattisgarh.geojson"
HAS_BUNDLE = BUNDLE.exists() and BUNDLE_BOUNDARY.exists()


def map_fig(df, color, size, hover, title, zoom=5.6, range_color=None):
    fn = getattr(px, "scatter_map", None)
    kwargs = dict(
        lat=df.columns[df.columns.str.contains("lat")][0],
        lon=df.columns[df.columns.str.contains("lon")][0],
        color=color, size=size, hover_name=hover,
        color_continuous_scale="YlOrRd", size_max=9, zoom=zoom,
        range_color=range_color,
        center=CENTER, height=560, title=title,
    )
    if fn is not None:
        fig = fn(df, map_style="open-street-map", **kwargs)
    else:  # older plotly
        fig = px.scatter_mapbox(df, mapbox_style="open-street-map", **kwargs)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


KEEP_COLS = {"latitude", "longitude", "acq_date", "acq_time", "confidence", "frp", "type", "daynight"}


@st.cache_data(show_spinner="Reading, cleaning and clipping the data (first run only)...")
def load_and_prepare(_files, files_key: str, boundary_text: str | None, name_filter: str):
    """Read every CSV, clip each to Chhattisgarh as it is read (keeps memory low), then build panels.

    `_files` is not hashed (leading underscore); `files_key` identifies the uploaded files instead,
    so the heavy work runs once and is cached across reruns.
    """
    import json

    districts = geo = names = None
    if boundary_text:
        districts, geo = fc.parse_geojson_districts(json.loads(boundary_text), name_filter or None)
        if not districts:
            raise ValueError("No polygons found in the boundary file (check the name filter).")
        names = sorted(districts)

    if _files is None:  # demo mode
        parts = [(fc.make_demo_data(), "synthetic demo data")]
    else:
        parts = []
        for u in _files:
            u.seek(0)
            part = pd.read_csv(u, usecols=lambda c: c.strip().lower() in KEEP_COLS)
            parts.append((part, u.name))

    frames, info, n_raw = [], [], 0
    for part, name in parts:
        part.columns = [c.strip().lower() for c in part.columns]
        dates = pd.to_datetime(part["acq_date"], errors="coerce") if "acq_date" in part.columns else None
        yr = (f"{dates.min().year}-{dates.max().year}" if dates is not None and dates.notna().any() else "?")
        kept = fc.load_fires(part, districts=districts)
        n_raw += len(part)
        info.append({"file": name, "rows": len(part), "kept in Chhattisgarh": len(kept), "years": yr})
        frames.append(kept)

    df = pd.concat(frames, ignore_index=True)
    dedupe = [c for c in ["latitude", "longitude", "acq_date", "acq_time"] if c in df.columns]
    n_before = len(df)
    df = df.drop_duplicates(subset=dedupe).reset_index(drop=True)
    if df.empty:
        raise ValueError("No detections fall inside Chhattisgarh (check the data and boundary files).")

    panel = fc.build_panel(df)
    dpanel = fc.build_district_panel(df, names) if districts else None
    return df, panel, dpanel, geo, names, pd.DataFrame(info), n_raw, n_before - len(df)


@st.cache_data(show_spinner="Loading the bundled Chhattisgarh dataset...")
def load_bundled(path: str, boundary_path: str, mtime: float):
    """Load the precomputed dataset made by prepare_data.py (already clipped and tagged by district)."""
    import json

    with open(boundary_path, encoding="utf-8") as fh:
        districts, geo = fc.parse_geojson_districts(json.load(fh))
    names = sorted(districts)
    df = pd.read_csv(path, dtype={"acq_time": str}, parse_dates=["acq_date"])
    df["year"] = df["acq_date"].dt.year
    df["month"] = df["acq_date"].dt.month
    df = fc.add_grid(df)
    panel = fc.build_panel(df)
    dpanel = fc.build_district_panel(df, names)
    info = pd.DataFrame([{"file": Path(path).name, "rows": len(df), "kept in Chhattisgarh": len(df),
                          "years": f"{df['year'].min()}-{df['year'].max()}"}])
    return df, panel, dpanel, geo, names, info, len(df), 0


@st.cache_resource(show_spinner="Training model...")
def train(_panel: pd.DataFrame, key: str, min_fires: int = 1):
    return fc.train_and_evaluate(_panel, min_fires=min_fires)


@st.cache_resource(show_spinner="Training district model...")
def train_district(_dpanel: pd.DataFrame, key: str):
    return fc.train_district_model(_dpanel)


def district_map(counts, geo, value_col, title):
    fn = getattr(px, "choropleth_map", None)
    kw = dict(
        geojson=geo, locations="district", featureidkey="properties._district", color=value_col,
        color_continuous_scale="YlOrRd", opacity=0.75, zoom=5.6, center=CENTER, height=560,
        title=title, hover_name="district",
    )
    if fn is not None:
        fig = fn(counts, map_style="open-street-map", **kw)
    else:  # older plotly
        fig = px.choropleth_mapbox(counts, mapbox_style="open-street-map", **kw)
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


def points_map(sub, title):
    pts = sub.sample(30000, random_state=0) if len(sub) > 30000 else sub
    fn = getattr(px, "scatter_map", None)
    kw = dict(lat="latitude", lon="longitude", zoom=5.6, center=CENTER, height=560, opacity=0.6, title=title)
    if "frp" in pts.columns:
        kw.update(color="frp", color_continuous_scale="YlOrRd",
                  range_color=(0, float(pts["frp"].quantile(0.95))))
    if fn is not None:
        fig = fn(pts, map_style="open-street-map", **kw)
    else:  # older plotly
        fig = px.scatter_mapbox(pts, mapbox_style="open-street-map", **kw)
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    return fig


@st.cache_resource(show_spinner="Training forecast model...")
def forecast_cached(_panel: pd.DataFrame, target_year: int, min_fires: int, key: str):
    return fc.forecast_year(_panel, target_year, min_fires)


st.title("Chhattisgarh Forest Fire Risk Dashboard")
st.caption(
    "Historical satellite fire detections (NASA FIRMS), a grid-cell fire-risk model and district-level forecasts. "
    "Grid cells are 0.25 degrees (~27 km)."
)

with st.sidebar:
    st.header("Data")
    source_options = (["Bundled Chhattisgarh data"] if HAS_BUNDLE else []) + ["Upload FIRMS files", "Synthetic demo"]
    source = st.radio("Data source", source_options, key="data_source")
    ups, boundary, name_filter = [], None, ""
    if source == "Upload FIRMS files":
        ups = st.file_uploader(
            "FIRMS CSV file(s) (VIIRS or MODIS). You can select several files, e.g. one per year.",
            type="csv", accept_multiple_files=True,
        )
        boundary = st.file_uploader(
            "Optional: Chhattisgarh boundary (GeoJSON) for exact clipping", type=["geojson", "json"]
        )
        name_filter = st.text_input(
            "Boundary name filter (optional)", value="",
            help="If your GeoJSON has many states/districts, type e.g. Chhattisgarh to keep only matching features.",
        )
    elif source == "Synthetic demo":
        st.warning("Showing SYNTHETIC data for demonstration. Not real fire records.")
    else:
        st.caption("Precomputed from NASA FIRMS VIIRS (S-NPP) active-fire detections, clipped to Chhattisgarh.")

try:
    if source == "Bundled Chhattisgarh data":
        files_key = "bundled"
        result = load_bundled(str(BUNDLE), str(BUNDLE_BOUNDARY), BUNDLE.stat().st_mtime)
    elif source == "Synthetic demo":
        files_key = "demo"
        result = load_and_prepare(None, "demo-nob", None, "")
    else:
        if not ups:
            st.info("Upload one or more FIRMS CSV files in the sidebar to begin.")
            st.stop()
        files_key = "upload-" + "-".join(f"{u.name}:{u.size}" for u in ups)
        boundary_text = boundary.getvalue().decode("utf-8") if boundary is not None else None
        result = load_and_prepare(
            ups, files_key + ("-b" if boundary_text else "-nob"), boundary_text, name_filter.strip())
except ValueError as e:
    st.error(str(e))
    st.stop()

df, panel, dpanel, geo, dnames, file_info, n_raw, n_dups = result

if source == "Upload FIRMS files":
    st.sidebar.success(
        f"{len(ups)} file(s) loaded, {n_raw:,} rows read, {len(df):,} detections kept in Chhattisgarh"
        + (f" ({n_dups:,} duplicates removed)" if n_dups else "")
    )
    with st.sidebar.expander("Files loaded"):
        st.dataframe(file_info, hide_index=True, use_container_width=True)
elif source == "Bundled Chhattisgarh data":
    st.sidebar.success(f"{len(df):,} detections, {df['year'].min()}-{df['year'].max()}, 27 districts")

source_key = files_key
data_key = f"{source_key}-{len(df)}-{'b' if dpanel is not None else 'nob'}"

with st.sidebar:
    st.header("Filters")
    yrs = sorted(df["year"].unique())
    yr_sel = st.select_slider("Years", options=yrs, value=(yrs[0], yrs[-1]))
    mo_sel = st.multiselect("Months", MONTHS, default=MONTHS)
    months_idx = [MONTHS.index(m) + 1 for m in mo_sel]

f = df[df["year"].between(*yr_sel) & df["month"].isin(months_idx)]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Fire detections", f"{len(f):,}")
k2.metric("Years covered", f"{yr_sel[0]}-{yr_sel[1]}")
peak = f.groupby("month").size().idxmax() if len(f) else None
k3.metric("Peak month", MONTHS[peak - 1] if peak else "-")
k4.metric("Active grid cells", f"{f['cell'].nunique():,}")

tabO, tabF, tab1, tabD, tab3, tab4 = st.tabs(["Observed map", "Forecast", "Trends", "Districts", "Model details", "About"])

with tabO:
    yrs_i = [int(y) for y in yrs]
    oc1, oc2, oc3 = st.columns(3)
    obs_year = oc1.selectbox("Year", ["All years"] + yrs_i, index=len(yrs_i), key="obs_year")
    obs_month = oc2.selectbox("Month", ["All months"] + MONTHS, index=0, key="obs_month")
    views = ["Grid cells", "Detection points"] + (["Districts"] if dpanel is not None else [])
    obs_view = oc3.radio("Map layer", views, horizontal=True, key="obs_view")

    sub = df
    if obs_year != "All years":
        sub = sub[sub["year"] == obs_year]
    if obs_month != "All months":
        sub = sub[sub["month"] == MONTHS.index(obs_month) + 1]
    label = f"{obs_month if obs_month != 'All months' else 'all months'}, {obs_year if obs_year != 'All years' else 'all years'}"

    if sub.empty:
        st.info("No detections for this selection.")
    else:
        st.caption("Observed satellite detections only. No model predictions on this tab.")
        o1, o2, o3 = st.columns(3)
        o1.metric("Fire detections", f"{len(sub):,}")
        o2.metric("Active grid cells", f"{sub['cell'].nunique():,}")
        if "frp" in sub.columns:
            o3.metric("Mean fire power (MW)", f"{sub['frp'].mean():.1f}")
        if obs_view == "Grid cells":
            cs = fc.cell_summary(sub)
            st.plotly_chart(map_fig(cs, "fires", "fires", "cell", f"Observed detections per grid cell, {label}"),
                            use_container_width=True)
            st.subheader("Top 15 hotspot cells")
            st.dataframe(cs.head(15), use_container_width=True, hide_index=True)
        elif obs_view == "Districts":
            st.plotly_chart(district_map(fc.district_counts(sub, dnames), geo, "fires",
                                         f"Observed detections by district, {label}"),
                            use_container_width=True)
        else:
            if len(sub) > 30000:
                st.caption("Showing a random sample of 30,000 detections for speed.")
            st.plotly_chart(points_map(sub, f"Individual detections, {label}"), use_container_width=True)

with tabF:
    st.markdown(
        "Forecast fire risk for a **whole year** using only earlier years (seasonality and each area's fire "
        "history; no weather). Pick a past year to backtest against what really happened, or the next year for a "
        "genuine forecast."
    )
    yrs_i = [int(y) for y in yrs]
    next_year = yrs_i[-1] + 1
    options = yrs_i[3:] + [next_year]
    f1, f2, f3 = st.columns(3)
    t_year = f1.selectbox(
        "Year to forecast", options, index=len(options) - 1, key="fc_year",
        format_func=lambda y: f"{y} (future)" if y == next_year else f"{y} (backtest)")
    f_month = f2.selectbox("Month", MONTHS, index=2, key="fc_month")
    f_min = f3.select_slider("Fire event = at least N detections", options=[1, 3, 5, 10, 20], value=1, key="fc_min")
    is_future = t_year == next_year

    try:
        fmodel, fframe, fmet = forecast_cached(panel, int(t_year), int(f_min), f"{data_key}-fy{t_year}-{f_min}")
    except ValueError as e:
        st.warning(str(e))
    else:
        active = fframe[fframe["prior_years_total"].fillna(0) > 0]
        midx = MONTHS.index(f_month) + 1
        mm = active[active["month"] == midx]

        if is_future:
            st.info(f"{t_year} has no observations yet. This is a forecast made from {yrs_i[0]}-{yrs_i[-1]} patterns.")
        elif fmet:
            st.success(
                f"Backtest for {t_year}: trained only on earlier years. Whole-year ROC AUC {fmet['auc_model']:.3f} "
                f"(baseline {fmet['auc_baseline']:.3f}), average precision {fmet['ap_model']:.3f} "
                f"(baseline {fmet['ap_baseline']:.3f})."
            )

        pred_df = mm[["cell", "lat", "lon", "risk"]].assign(risk=mm["risk"].round(3))
        if is_future:
            st.plotly_chart(map_fig(pred_df, "risk", "risk", "cell", f"Forecast fire risk: {f_month} {t_year}",
                                    range_color=(0, 1)), use_container_width=True)
        else:
            left, right = st.columns(2)
            obs_df = mm.dropna(subset=["fires"])
            obs_df = obs_df[obs_df["fires"] > 0][["cell", "lat", "lon", "fires"]]
            with left:
                if obs_df.empty:
                    st.info("No observed detections for this month.")
                else:
                    st.plotly_chart(map_fig(obs_df, "fires", "fires", "cell",
                                            f"Observed detections: {f_month} {t_year}"), use_container_width=True)
            with right:
                st.plotly_chart(map_fig(pred_df, "risk", "risk", "cell", f"Predicted risk: {f_month} {t_year}",
                                        range_color=(0, 1)), use_container_width=True)

        monthly = active.groupby("month").agg(expected=("risk", "sum")).reset_index()
        if not is_future:
            ev = active.dropna(subset=["fires"])
            observed = ev.assign(hit=(ev["fires"] >= f_min).astype(int)).groupby("month")["hit"].sum()
            monthly["observed"] = monthly["month"].map(observed)
        monthly["month"] = monthly["month"].map(lambda m: MONTHS[m - 1])
        long = monthly.melt(id_vars="month", var_name="series", value_name="cells")
        long = long.dropna(subset=["cells"])
        st.plotly_chart(
            px.bar(long, x="month", y="cells", color="series", barmode="group",
                   title=f"Grid cells with a fire event per month, {t_year}: "
                         f"{'forecast' if is_future else 'expected vs observed'}"),
            use_container_width=True)

        out = active[["cell", "lat", "lon", "year", "month", "risk"]].rename(columns={"lat": "cell_lat", "lon": "cell_lon"})
        st.download_button("Download this forecast (CSV)", out.to_csv(index=False).encode("utf-8"),
                           file_name=f"fire_risk_forecast_{t_year}.csv", mime="text/csv")
        st.caption(
            "Risk = estimated probability that the cell has at least "
            f"{f_min} detection{'s' if f_min > 1 else ''} in that month. This is a seasonal outlook from history, "
            "not a day-to-day fire warning, and it does not know about the weather."
        )

with tab1:
    c1, c2 = st.columns(2)
    by_month = f.groupby("month").size().reindex(range(1, 13), fill_value=0).reset_index()
    by_month.columns = ["month", "fires"]
    by_month["month"] = by_month["month"].map(lambda m: MONTHS[m - 1])
    c1.plotly_chart(px.bar(by_month, x="month", y="fires", title="Seasonality: detections by month"),
                    use_container_width=True)
    by_year = f.groupby("year").size().reset_index(name="fires")
    c2.plotly_chart(px.bar(by_year, x="year", y="fires", title="Detections by year"),
                    use_container_width=True)
    heat = f.groupby(["year", "month"]).size().reset_index(name="fires")
    heat["month"] = heat["month"].map(lambda m: MONTHS[m - 1])
    pivot = heat.pivot(index="year", columns="month", values="fires").reindex(columns=MONTHS)
    st.plotly_chart(px.imshow(pivot, color_continuous_scale="YlOrRd", aspect="auto",
                              title="Year x month heatmap"), use_container_width=True)

with tabD:
    if dpanel is None:
        st.info("Upload the Chhattisgarh boundary (GeoJSON) in the sidebar to unlock district-level analysis.")
    else:
        counts = fc.district_counts(f, dnames)
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(district_map(counts, geo, "fires", "Fire detections by district"),
                        use_container_width=True)
        top = counts.head(15).sort_values("fires")
        c2.plotly_chart(px.bar(top, x="fires", y="district", orientation="h", title="Top 15 districts"),
                        use_container_width=True)

        st.subheader("District fire forecast")
        st.markdown(
            "**Task:** forecast the number of fire detections in each district for a month, using only "
            "past information. Trained on earlier years, tested on the latest year. "
            "**Baseline:** the district's average for the same month in earlier years."
        )
        try:
            dmodel, dm, dtest = train_district(dpanel, f"{data_key}-d{len(dpanel)}")
        except ValueError as e:
            st.warning(str(e))
        else:
            d1, d2, d3 = st.columns(3)
            d1.metric("Rank correlation (model)", f"{dm['spearman_model']:.3f}",
                      delta=f"{dm['spearman_model'] - dm['spearman_baseline']:+.3f} vs baseline")
            d2.metric("Mean abs. error, detections (model)", f"{dm['mae_model']:.1f}",
                      delta=f"{dm['mae_model'] - dm['mae_baseline']:+.1f} vs baseline", delta_color="inverse")
            d3.metric("Top-5 districts correctly named", f"{dm['top5_model']:.0%}",
                      delta=f"{(dm['top5_model'] - dm['top5_baseline']) * 100:+.0f} pts vs baseline")
            st.caption(
                f"Model type chosen on the validation year before the test year: {dm['model_name']}. "
                f"Total predicted / actual detections in the test year: model {dm['bias_model']:.2f}, "
                f"baseline {dm['bias_baseline']:.2f} (1.00 = unbiased)."
            )
            st.caption(
                f"Test year {dm['test_year']}: {dm['n_test']} district-months. Rank correlation is Spearman between "
                "predicted and actual counts. Top-5 compares the 5 districts predicted to have the most fires "
                "with the 5 that did, averaged over months with enough fire activity."
            )
            sel_m = st.selectbox("Month to inspect", MONTHS, index=2, key="district_month")
            sub = dtest[dtest["month"] == MONTHS.index(sel_m) + 1].nlargest(10, "fires")
            if sub.empty:
                st.info("No test data for that month.")
            else:
                long = sub.rename(columns={"fires": "actual", "pred": "predicted"}).melt(
                    id_vars="district", value_vars=["actual", "predicted"],
                    var_name="series", value_name="detections")
                st.plotly_chart(
                    px.bar(long, x="district", y="detections", color="series", barmode="group",
                           title=f"Top 10 districts, {sel_m} {dm['test_year']}: actual vs predicted"),
                    use_container_width=True)

with tab4:
    st.markdown(
        """
**Data:** NASA FIRMS active-fire detections, clipped to a bounding box around Chhattisgarh
(low-confidence detections and non-vegetation sources removed). Active-fire detections are satellite hotspots, not confirmed
forest fires, and can include agricultural burning.

**Model:** gradient boosting on a grid-cell x month panel. Features: location, month seasonality,
previous-month fires, cumulative past fires, and the cell's average for the same month in earlier years.

**Attribution:** fire detections are from NASA FIRMS (Fire Information for Resource Management System),
VIIRS S-NPP active-fire product. Check the licence of your district boundary file before publishing it.

**Districts:** with a boundary file, detections are clipped to the state and tagged with their district. A second
model forecasts monthly detection counts per district and is compared with a same-month-average baseline.

**Limitations:** no weather, vegetation or terrain features yet; without a boundary file the bounding box
also includes parts of neighbouring states (e.g. Odisha, Telangana, Maharashtra); grid cells are not
administrative districts.

**Next steps:** add rainfall/temperature (IMD or ERA5), forest-cover layers, and district boundaries.
        """
    )

with tab3:
    st.subheader("Next-month risk model (evaluation)")
    st.caption("This model also uses last month's actual fire counts, so it is a model check, not a long-range forecast. "
               "For forecasts of a whole year, use the Forecast tab.")
    min_fires = st.select_slider(
        "Count a cell-month as a fire event when it has at least this many detections",
        options=[1, 3, 5, 10, 20], value=1,
        help="1 = any detection. Higher values target heavier fire activity, so the risk map shows more contrast.",
    )
    try:
        model, metrics, test = train(panel, f"{data_key}-m{min_fires}", min_fires)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    st.markdown(
        f"**Task:** predict whether a grid cell has at least {min_fires} fire detection{'s' if min_fires > 1 else ''} in a given month. "
        f"Trained on years before {metrics['test_year']}, tested on {metrics['test_year']} "
        f"(time-based split, features use only past data)."
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC AUC (model)", f"{metrics['auc_model']:.3f}")
    m2.metric("ROC AUC (baseline)", f"{metrics['auc_baseline']:.3f}",
              help="Baseline = average fires in the same cell and month in earlier years.")
    m3.metric("Avg precision (model)", f"{metrics['ap_model']:.3f}")
    m4.metric("Avg precision (baseline)", f"{metrics['ap_baseline']:.3f}")

    c1, c2 = st.columns([2, 1])
    with c1:
        ys = sorted(panel["year"].unique())
        sel_y = st.selectbox("Year", ys, index=len(ys) - 1)
        sel_m = st.selectbox("Month", MONTHS, index=2)
        rm = fc.risk_map_for_month(model, panel, sel_y, MONTHS.index(sel_m) + 1)
        rm["risk"] = rm["risk"].round(3)
        if rm.empty:
            st.info("No data for that month.")
        else:
            st.plotly_chart(map_fig(rm, "risk", "risk", "cell", f"Predicted fire risk: {sel_m} {sel_y}", range_color=(0, 1)),
                            use_container_width=True)
    with c2:
        imp = fc.permutation_importance_table(model, test)
        st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
                               title="Permutation importance (AUC drop)"),
                        use_container_width=True)
    st.caption(
        "Note: for months inside the training period, the map shows in-sample predictions. "
        "Use the test year for an honest view."
    )
