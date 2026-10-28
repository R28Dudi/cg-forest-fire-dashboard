# Chhattisgarh Forest Fire Risk Dashboard

Interactive dashboard of NASA VIIRS satellite fire detections over Chhattisgarh (2012-2025), with a year-ahead
fire-risk forecast, a next-month risk model and district-level monthly forecasts.

**Live app:** _add your Streamlit link here_

## What it does
- **Observed map:** real detections only, for any year and month, as grid cells, individual points or districts.
- **Forecast:** year-ahead fire-risk outlook from earlier years only. Pick a past year to backtest against what really
  happened, or the next year for a forecast. Downloadable as CSV.
- **Trends, Districts:** seasonality, yearly counts, district map and a district-level monthly forecast.
- **Model details:** next-month risk model evaluation.

## Results (held-out 2025; models trained only on earlier years)
Data: 461,860 vegetation-fire detections inside Chhattisgarh, 2012-2025, tagged to 27 districts.

| Task | Metric | Model | Seasonal baseline |
|---|---|---|---|
| Year-ahead grid-cell risk (>=1 detection) | ROC AUC | 0.979 | 0.970 |
| | Average precision | 0.928 | 0.918 |
| Next-month grid-cell risk | ROC AUC | 0.981 | 0.970 |
| | Average precision | 0.936 | 0.918 |
| District-month detection counts | Mean absolute error | 70.4 | 79.2 |
| | Spearman rank correlation | 0.922 | 0.914 |
| | Top-5 districts named correctly | 65% | 67% |

The baseline is each cell's (or district's) average for the same calendar month in earlier years. Most of the skill is
seasonality; the models add a small, consistent improvement. There are no weather features yet.

## Method
- 0.25 degree grid (about 27 km), cell x month panel. Target: at least N detections in the cell-month (N adjustable).
- Features use only past information: location, month seasonality, same-month history, last year's counts, cumulative history.
- Gradient boosting (grid cells); Poisson-loss count models for districts, with the model type chosen on the year before
  the test year. Time-based train/test split throughout.
- Data cleaning: vegetation fires only (FIRMS type 0), low-confidence detections removed, exact clip to the district boundary.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app offers three data sources in the sidebar: the bundled dataset (if `data/` exists), your own FIRMS CSV files,
or synthetic demo data.

## Build the bundled dataset
Download yearly VIIRS CSVs from NASA FIRMS (whole-India files are fine), then:
```bash
python prepare_data.py --csv-dir "path/to/csv/folder" --boundary "path/to/Chhattisgarh.geojson"
```
This writes `data/cg_fires.csv.gz` and `data/Chhattisgarh.geojson`. Commit both to the repository.

## Deploy (Streamlit Community Cloud)
1. Push this folder (including `data/`) to a GitHub repository.
2. At share.streamlit.io choose "New app", select the repository, branch `main`, main file `app.py`, then deploy.

## Limitations
- Satellite hotspots are not confirmed forest fires; crop burning is included.
- No weather, vegetation or terrain features, so the forecast is a seasonal outlook, not a day-to-day warning.
- Detections are clipped to a district boundary of 27 districts, an older district map.

## Data and attribution
Fire detections: NASA FIRMS (Fire Information for Resource Management System), VIIRS S-NPP active-fire product.
Check the licence of your district boundary file before publishing it.
