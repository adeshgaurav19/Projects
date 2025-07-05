from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
import io
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import uvicorn
import base64
from io import BytesIO
import numpy as np
import asyncio
import logging
import os
from datetime import datetime
import gc

app = FastAPI()

# CORS Middleware setup
origins = [
    "http://localhost",
    "http://127.0.0.1",
    "file://",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Placeholder for uploaded/selected data
data_store = {}

def plot_to_base64(plt):
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.clf()
    plt.close('all')
    gc.collect()
    return image_base64

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Attempting to upload file: {file.filename}, size: {file.size}")
        contents = await file.read()
        if contents:
            logger.info(f"File content preview: {contents[:200].decode('utf-8', errors='ignore')}")
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        logger.info(f"File {file.filename} uploaded successfully. Shape: {df.shape}")
        
        # Check for NaN values
        if df.isna().any().any():
            logger.warning("Uploaded data contains NaN values")
        
        data_store['data'] = df
        return {"message": "File uploaded successfully", "columns": list(df.columns)}
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")

@app.post("/fetch_finance_data/")
async def fetch_finance_data(ticker: str = Form(...), start_date: str = Form(...), end_date: str = Form(...)):
    try:
        logger.info(f"Fetching data for ticker: {ticker}, from {start_date} to {end_date}")
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        loop = asyncio.get_event_loop()
        with asyncio.timeout(30):
            df = await loop.run_in_executor(None, yf.download, ticker, start, end)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {ticker} between {start_date} and {end_date}")

        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])  # Ensure date column is datetime
        data_store['data'] = df
        
        logger.info(f"Successfully fetched {len(df)} rows of data for {ticker}")
        return {
            "message": "Data fetched successfully",
            "columns": list(df.columns),
            "rows": len(df),
            "first_date": df['Date'].iloc[0].strftime('%Y-%m-%d'),
            "last_date": df['Date'].iloc[-1].strftime('%Y-%m-%d')
        }
    except asyncio.TimeoutError:
        logger.error("Request timed out while fetching data.")
        raise HTTPException(status_code=408, detail="Request timed out while fetching data.")
    except ValueError as ve:
        logger.error(f"Invalid ticker or date range: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid ticker or date range: {ve}")
    except Exception as e:
        logger.error(f"Unexpected error fetching data: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error fetching data: {e}")

@app.get("/eda_options/")
def eda_options():
    logger.info("Returning EDA options")
    return {"options": ["line_plot", "seasonality_decompose", "correlation_heatmap", "summary_statistics", "rolling_mean", "histogram", "boxplot"]}

@app.post("/eda/")
def eda(option: str = Form(...), column: Optional[str] = Form(None)):
    try:
        logger.info(f"Performing EDA with option: {option}, column: {column}")
        if 'data' not in data_store:
            raise HTTPException(status_code=400, detail="No data available. Please upload or fetch data first.")
        
        df = data_store['data']

        if option == "line_plot":
            if column is None:
                raise HTTPException(status_code=400, detail="Column name is required for line plot.")
            plt.figure(figsize=(10, 5))
            df[column].plot()
            plt.title(f"Line Plot of {column}")
            plt.xlabel("Index")
            plt.ylabel(column)
            return {"image": plot_to_base64(plt)}

        elif option == "seasonality_decompose":
            if column is None:
                raise HTTPException(status_code=400, detail="Column name is required for decomposition.")
            result = seasonal_decompose(df[column].dropna(), period=12, model="additive")
            plt.figure(figsize=(10, 5))
            result.plot()
            return {"image": plot_to_base64(plt)}

        elif option == "correlation_heatmap":
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap")
            return {"image": plot_to_base64(plt)}

        elif option == "summary_statistics":
            return {"summary": df.describe().to_dict()}

        elif option == "rolling_mean":
            if column is None:
                raise HTTPException(status_code=400, detail="Column name is required for rolling mean.")
            plt.figure(figsize=(10, 5))
            rolling_mean = df[column].rolling(window=5).mean()
            plt.plot(df[column], label="Original")
            plt.plot(rolling_mean, label="Rolling Mean", linestyle="--")
            plt.legend()
            plt.title(f"Rolling Mean of {column}")
            return {"image": plot_to_base64(plt)}

        elif option == "histogram":
            if column is None:
                raise HTTPException(status_code=400, detail="Column name is required for histogram.")
            plt.figure(figsize=(10, 5))
            df[column].plot(kind='hist', bins=20)
            plt.title(f"Histogram of {column}")
            return {"image": plot_to_base64(plt)}

        elif option == "boxplot":
            if column is None:
                raise HTTPException(status_code=400, detail="Column name is required for boxplot.")
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df, x=column)
            plt.title(f"Boxplot of {column}")
            return {"image": plot_to_base64(plt)}

        else:
            raise HTTPException(status_code=400, detail="Invalid EDA option selected.")

    except Exception as e:
        logger.error(f"Error during EDA: {e}")
        raise HTTPException(status_code=500, detail=f"Error during EDA: {e}")

@app.post("/feature_engineering/")
def feature_engineering(lags: Optional[int] = Form(None), rolling_window: Optional[int] = Form(None), ema_span: Optional[int] = Form(None), fourier_k: Optional[int] = Form(None)):
    try:
        logger.info(f"Applying feature engineering with lags: {lags}, rolling_window: {rolling_window}, ema_span: {ema_span}, fourier_k: {fourier_k}")
        if 'data' not in data_store:
            raise HTTPException(status_code=400, detail="No data available. Please upload or fetch data first.")

        df = data_store['data']
        if lags:
            for lag in range(1, lags + 1):
                df[f"lag_{lag}"] = df.iloc[:, 1].shift(lag)
        if rolling_window:
            for column in df.select_dtypes(include=['float', 'int']).columns:
                df[f"rolling_mean_{column}"] = df[column].rolling(window=rolling_window).mean()
        if ema_span:
            for column in df.select_dtypes(include=['float', 'int']).columns:
                df[f"ema_{column}"] = df[column].ewm(span=ema_span).mean()
        if fourier_k:
            t = np.arange(len(df))
            for k in range(1, fourier_k + 1):
                df[f"sin_{k}"] = np.sin(2 * np.pi * k * t / len(t))
                df[f"cos_{k}"] = np.cos(2 * np.pi * k * t / len(t))
        
        logger.info(f"New columns after feature engineering: {list(df.columns)}")
        data_store['data'] = df
        return {"message": "Feature engineering applied successfully", "columns": list(df.columns)}
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise HTTPException(status_code=500, detail=f"Error during feature engineering: {e}")

@app.post("/model_selection/")
def model_selection(model: str = Form(...), params: dict = Form(...), target_column: str = Form(...)):
    try:
        logger.info(f"Running model selection for model: {model}, target_column: {target_column}")
        if 'data' not in data_store:
            raise HTTPException(status_code=400, detail="No data available. Please upload or fetch data first.")

        df = data_store['data']
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail="Invalid target column selected.")

        y = df[target_column].dropna()

        if model == "ARIMA":
            order = tuple(params.get("order", [1, 1, 1]))
            model_instance = ARIMA(y, order=order)
            model_fit = model_instance.fit()
            plt.figure(figsize=(10, 5))
            plt.plot(y, label="Original")
            plt.plot(model_fit.fittedvalues, label="Fitted", linestyle="--")
            plt.legend()
            plt.title("ARIMA Model Fit")
            return {"summary": model_fit.summary().as_text(), "image": plot_to_base64(plt)}

        elif model == "SARIMA":
            order = tuple(params.get("order", [1, 1, 1]))
            seasonal_order = tuple(params.get("seasonal_order", [1, 1, 1, 12]))
            model_instance = SARIMAX(y, order=order, seasonal_order=seasonal_order)
            model_fit = model_instance.fit()
            plt.figure(figsize=(10, 5))
            plt.plot(y, label="Original")
            plt.plot(model_fit.fittedvalues, label="Fitted", linestyle="--")
            plt.legend()
            plt.title("SARIMA Model Fit")
            return {"summary": model_fit.summary().as_text(), "image": plot_to_base64(plt)}

        elif model == "Prophet":
            df_prophet = df[["Date", target_column]].rename(columns={"Date": "ds", target_column: "y"})
            prophet_model = Prophet()
            prophet_model.fit(df_prophet)
            future = prophet_model.make_future_dataframe(periods=30)
            forecast = prophet_model.predict(future)
            fig = prophet_model.plot(forecast)
            return {"forecast": forecast.to_dict(), "image": plot_to_base64(fig.gcf())}

        elif model == "GradientBoosting":
            X = df.drop(columns=[target_column]).select_dtypes(include=['float', 'int']).fillna(0)
            y = df[target_column].fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            gb_model = GradientBoostingRegressor(**params)
            gb_model.fit(X_train, y_train)
            predictions = gb_model.predict(X_test)
            plt.figure(figsize=(10, 5))
            plt.plot(y_test.values, label="True Values")
            plt.plot(predictions, label="Predictions", linestyle="--")
            plt.legend()
            plt.title("Gradient Boosting Model Predictions")
            return {"predictions": predictions.tolist(), "image": plot_to_base64(plt)}

        else:
            raise HTTPException(status_code=400, detail="Invalid model selected.")

    except Exception as e:
        logger.error(f"An error occurred while running the model: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while running the model: {e}")

@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting the application in debug mode")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, log_level="debug")