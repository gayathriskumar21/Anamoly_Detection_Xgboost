import base64
import io
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI()

# Enable CORS for all origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change "*" to your frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and encoders
model = joblib.load('models/xgboost_model.joblib')
cell_encoder = joblib.load('models/cellname_encoder.joblib')
feature_names = joblib.load('models/feature_names.joblib')

kpi_cols = ['PRBUsageUL','PRBUsageDL','meanThr_DL','meanThr_UL',
            'maxThr_DL','maxThr_UL','meanUE_DL','meanUE_UL','maxUE_DL','maxUE_UL','maxUE_UL+DL']

# Store uploaded and processed data here
stored_data = {
    "raw_df": None,
    "processed_df": None,
    "analyzed_df": None,
    "base_stations": []
}

# Function to convert numpy data types to native Python types recursively
def convert_numpy_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(i) for i in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj

# Anomaly and preprocessing utility functions
def calc_iqr_thresholds(df, group_col, kpi_cols):
    thresholds = {}
    for grp in df[group_col].unique():
        grp_df = df[df[group_col] == grp]
        thresholds_grp = {}
        for col in kpi_cols:
            Q1 = grp_df[col].quantile(0.25)
            Q3 = grp_df[col].quantile(0.75)
            IQR = Q3 - Q1
            thresholds_grp[col] = {'low': Q1 - 0.1 * IQR, 'high': Q3 + 0.1 * IQR}
        thresholds[grp] = thresholds_grp
    return thresholds

def check_security_anomaly(row):
    if (row['meanThr_UL'] > 5) or (row['meanThr_DL'] > 5):
        return 3
    return None

def check_cell_level_anomaly(row, thresholds):
    cell_name = row['CellName']
    cell_thresholds_dict = thresholds.get(cell_name, {})
    for kpi in ['meanThr_DL', 'meanThr_UL', 'PRBUsageUL', 'PRBUsageDL']:
        if kpi in cell_thresholds_dict:
            val = row[kpi]
            if val > cell_thresholds_dict[kpi]['high']:
                return 2
            elif val < cell_thresholds_dict[kpi]['low']:
                return 1
    return None

def check_bs_level_anomaly(row, bs_data):
    bs_name = row['BaseStation']
    bs_cells_data = bs_data[bs_data['BaseStation'] == bs_name]
    if bs_cells_data.empty:
        return None
    high_usage_cells = bs_cells_data[bs_cells_data['PRBUsageDL'] > 40]
    low_usage_cells = bs_cells_data[bs_cells_data['PRBUsageDL'] < 10]
    if not high_usage_cells.empty and not low_usage_cells.empty:
        return 2
    return None

def assign_comprehensive_label(row, cell_thresholds, bs_data):
    sec = check_security_anomaly(row)
    if sec is not None:
        return 3
    cell_anom = check_cell_level_anomaly(row, cell_thresholds)
    bs_anom = check_bs_level_anomaly(row, bs_data)
    if cell_anom == 2 or bs_anom == 2:
        return 2
    if cell_anom == 1:
        return 1
    return 0

def preprocess_uploaded_data(df):
    df['BaseStation'] = df['CellName'].apply(lambda x: x[:-4] if x.endswith("LTE") else x)
    exclude_cols = ['Time', 'CellName', 'Hour', 'DayOfWeek', 'Unusual']
    numeric_cols = [c for c in df.columns if c not in exclude_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col].replace('#¡VALOR!', np.nan), errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)
    df['CellName_encoded'] = cell_encoder.transform(df['CellName'])
    df['Time'] = pd.to_datetime(df['Time'], format='%H:%M')
    df['Hour'] = df['Time'].dt.hour
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df_reindexed = df.reindex(columns=feature_names, fill_value=0)
    return df, df_reindexed

def analyze_data(df):
    cell_thresholds = calc_iqr_thresholds(df, 'CellName', kpi_cols)
    df['IQR_AnomalyCode'] = df.apply(assign_comprehensive_label, axis=1, cell_thresholds=cell_thresholds, bs_data=df)
    anomaly_map = {0:'Normal',1:'Low Anomaly',2:'High Anomaly',3:'Security Threats'}
    df['IQR_AnomalyType'] = df['IQR_AnomalyCode'].map(anomaly_map)
    return df

def filter_dataframe(df, base_stations: Optional[List[str]], start_time: Optional[str], end_time: Optional[str]):
    filtered = df
    if base_stations and 'ALL' not in base_stations:
        try:
            # convert incoming base_stations list strings to ints
            base_stations_int = list(map(int, base_stations))
        except ValueError:
            # fallback if conversion fails
            base_stations_int = base_stations
        filtered = filtered[df['BaseStation'].isin(base_stations_int)]
    if start_time:
        filtered = filtered[df['Time'] >= pd.to_datetime(start_time, format='%H:%M')]
    if end_time:
        filtered = filtered[df['Time'] <= pd.to_datetime(end_time, format='%H:%M')]
    return filtered

def generate_summary(df):
    label_counts = df['IQR_AnomalyType'].value_counts().to_dict()
    return {
        "total_samples": int(len(df)),
        "normal_samples": int(label_counts.get('Normal', 0)),
        "low_anomaly": int(label_counts.get('Low Anomaly', 0)),
        "high_anomaly": int(label_counts.get('High Anomaly', 0)),
        "security_threats": int(label_counts.get('Security Threats', 0)),
    }

def generate_kpi_alerts(df):
    alerts = []
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != 'IQR_AnomalyType']
    for col in numeric_cols:
        desc = df[col].describe()
        if '75%' in desc and desc['mean'] > 1.2 * desc['75%']:
            alerts.append(f"{col} has a HIGH VALUE ALERT")
    return alerts

def df_to_csv_bytes(df):
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return io.BytesIO(stream.getvalue().encode())

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files accepted")
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode('latin1')))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    processed_df, feature_df = preprocess_uploaded_data(df)
    analyzed_df = analyze_data(processed_df)

    stored_data["raw_df"] = df
    stored_data["processed_df"] = feature_df
    stored_data["analyzed_df"] = analyzed_df
    stored_data["base_stations"] = list(analyzed_df['BaseStation'].unique())

    response_data = {"summary": generate_summary(analyzed_df), "base_stations": ["ALL"] + stored_data["base_stations"]}
    return convert_numpy_to_native(response_data)

@app.get("/filters/base_stations")
async def get_base_stations():
    if stored_data["analyzed_df"] is None:
        return {"base_stations": []}
    response = {"base_stations": ["ALL"] + stored_data["base_stations"]}
    return convert_numpy_to_native(response)

@app.get("/summary")
async def get_summary(base_stations: Optional[List[str]] = Query(None),
                      start_time: Optional[str] = None,
                      end_time: Optional[str] = None):
    if stored_data["analyzed_df"] is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    filtered = filter_dataframe(stored_data["analyzed_df"], base_stations, start_time, end_time)
    return convert_numpy_to_native(generate_summary(filtered))

@app.get("/alerts")
async def get_alerts(base_stations: Optional[List[str]] = Query(None),
                     start_time: Optional[str] = None,
                     end_time: Optional[str] = None):
    if stored_data["analyzed_df"] is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    filtered = filter_dataframe(stored_data["analyzed_df"], base_stations, start_time, end_time)
    alerts = generate_kpi_alerts(filtered)
    return convert_numpy_to_native({"alerts": alerts})

@app.get("/table")
async def get_table(base_stations: Optional[List[str]] = Query(None),
                    start_time: Optional[str] = None,
                    end_time: Optional[str] = None,
                    limit: int = 100,
                    offset: int = 0):
    if stored_data["analyzed_df"] is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    filtered = filter_dataframe(stored_data["analyzed_df"], base_stations, start_time, end_time)
    paginated = filtered.iloc[offset:offset+limit]
    return convert_numpy_to_native({"total": len(filtered), "rows": paginated.to_dict(orient="records")})

@app.get("/charts/pie")
async def get_pie_chart(
    base_stations: Optional[List[str]] = Query(None),
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    if stored_data["analyzed_df"] is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    filtered = filter_dataframe(stored_data["analyzed_df"], base_stations, start_time, end_time)
    counts = filtered['IQR_AnomalyType'].value_counts().to_dict()
    data = [
        {"category": cat, "count": counts.get(cat, 0)}
        for cat in ['Normal', 'Low Anomaly', 'High Anomaly', 'Security Threats']
    ]
    return convert_numpy_to_native({"data": data})


@app.get("/charts/base_station")
async def get_base_station_chart(
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    if stored_data["analyzed_df"] is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    # Filter only by time, not by base station(s)
    filtered = filter_dataframe(stored_data["analyzed_df"], None, start_time, end_time)
    grp = filtered.groupby("BaseStation")['IQR_AnomalyCode'].apply(lambda x: (x > 0).sum())
    data = [{"base_station": str(k), "anomaly_count": int(v)} for k, v in grp.items()]
    return convert_numpy_to_native({"data": data})

@app.get("/charts/cell")
async def get_cell_chart(
    base_station: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
):
    if stored_data["analyzed_df"] is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    # Always treat base_station as string
    filtered = filter_dataframe(stored_data["analyzed_df"], [base_station], start_time, end_time)
    grp = filtered.groupby('CellName')['IQR_AnomalyCode'].apply(lambda x: (x > 0).sum())
    data = [{"cell_name": k, "anomaly_count": int(v)} for k, v in grp.items()]
    return convert_numpy_to_native({"data": data})

@app.get("/charts/time")
async def get_time_chart(base_station: str):
    if stored_data["analyzed_df"] is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    filtered = filter_dataframe(stored_data["analyzed_df"], [base_station], None, None)
    grp = filtered.groupby('Hour')['IQR_AnomalyCode'].apply(lambda x: (x > 0).sum())
    data = [{"hour": int(k), "anomaly_count": int(v)} for k, v in sorted(grp.items())]
    return convert_numpy_to_native({"data": data})

@app.get("/download")
async def download_csv(base_stations: Optional[List[str]] = Query(None),
                       start_time: Optional[str] = None,
                       end_time: Optional[str] = None):
    if stored_data["analyzed_df"] is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    filtered = filter_dataframe(stored_data["analyzed_df"], base_stations, start_time, end_time)
    stream = df_to_csv_bytes(filtered)
    return StreamingResponse(stream, media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=filtered_kpi_data.csv"})

@app.get("/")
async def health_check():
    return {"message": "Backend ready"}