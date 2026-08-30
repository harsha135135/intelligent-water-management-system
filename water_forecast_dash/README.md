# Water Forecast Dashboard (Flask Extension)

This folder contains a self-contained Flask dashboard that can run as a standalone app now and be mounted into your deployed web app later as an extension.

## What it uses
- Dataset: `Capstone/dataset/*/*.json`
- Trained models:
  - `Capstone/results/autogluon/hourly_forecasting`
  - `Capstone/results/patchtst/hourly_forecasting`

## Features
- Tank selector (all tanks from dataset)
- Model selector (AutoGluon, PatchTST)
- Next 24-hour forecast (configurable horizon)
- Recent history + forecast line chart
- Forecast table and summary metrics
- Incremental Waltr sync to update dataset till latest available date
- API endpoints for deployment integration
- Auto-refresh tank discovery when new JSON files are added

## Run locally
1. Install dependencies:
   ```bash
   pip install -r water_forecast_dash/requirements.txt
   ```
2. Start app:
   ```bash
   python water_forecast_dash/app.py
   ```
3. Open:
   - `http://127.0.0.1:5050`

## API endpoints
- `GET /api/health`
- `GET /api/tanks`
- `POST /api/refresh`
- `GET /api/retrain-status`
- `POST /api/retrain-model`
- `POST /api/retrain-stop`
- `POST /api/sync-data`
- `GET /api/history?tank_id=<TANK_ID>&hours=168`
- `POST /api/forecast`

## Interactive controls
- `Retrain Forecast Model` supports `autogluon` and `patchtst` targets.
- `Stop Training` sends graceful termination for the active retrain process.
- `Sync Latest Data` calls Waltr APIs, finds the last date already stored per tank, and downloads only missing daily files from next day to selected end date.

Example forecast payload:
```json
{
  "tank_id": "BE_BLOCK_OHT",
  "prediction_length": 24,
  "model_keys": ["autogluon", "patchtst"]
}
```

## Integrate into deployed app
You can either:
1. Reverse-proxy this Flask service under a route like `/water-forecast`.
2. Import `create_app()` from `water_forecast_dash/app.py` and mount it in your Python deployment stack.

## Retrain from Dashboard
Use the dashboard retrain controls to retrain either `autogluon` or `patchtst` after syncing latest data.
