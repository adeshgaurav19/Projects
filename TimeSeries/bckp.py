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

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost",       # Allow frontend running on localhost
    "http://127.0.0.1",       # Allow frontend running on localhost IP
    "file://",                 # Allow file:// (local files)
    "*",                       # Allow all origins (this is more permissive, but useful for debugging)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
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
    return image_base64

@app.post("/upload_file/")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info(f"Attempting to upload file: {file.filename}, size: {file.size}")
        contents = await file.read()
        # Log the first few lines of the file to check its content
        if contents:
            logger.info(f"File content preview: {contents[:200].decode('utf-8', errors='ignore')}")
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        data_store['data'] = df
        logger.info(f"File {file.filename} uploaded successfully. Columns: {list(df.columns)}")
        return {"message": "File uploaded successfully", "columns": list(df.columns)}
    except Exception as e:
        # Check if the error is related to file reading or parsing
        if "CSV" in str(e):
            logger.error(f"CSV Parsing Error for file {file.filename}: {e}")
        else:
            logger.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {e}")
    
@app.post("/fetch_finance_data/")
async def fetch_finance_data(ticker: str = Form(...), start_date: str = Form(...), end_date: str = Form(...)):
    try:
        logger.info(f"Fetching data for ticker: {ticker}, from {start_date} to {end_date}")
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Run yfinance download function in a separate thread
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, yf.download, ticker, start, end)
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {ticker} between {start_date} and {end_date}")

        df.reset_index(inplace=True)
        # data_store['data'] = df   # If you need to store data globally
        
        logger.info(f"Successfully fetched {len(df)} rows of data for {ticker}")
        return {
            "message": "Data fetched successfully",
            "columns": list(df.columns),
            "rows": len(df) if not df.empty else 0,
            "first_date": df['Date'].iloc[0].strftime('%Y-%m-%d') if not df.empty else None,
            "last_date": df['Date'].iloc[-1].strftime('%Y-%m-%d') if not df.empty else None
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
            image = plot_to_base64(plt)
            logger.info(f"Line plot for column {column} generated")
            return {"image": image}

        elif option == "seasonality_decompose":
            if column is None:
                raise HTTPException(status_code=400, detail="Column name is required for decomposition.")
            result = seasonal_decompose(df[column].dropna(), period=12, model="additive")
            plt.figure(figsize=(10, 5))
            result.plot()
            image = plot_to_base64(plt)
            logger.info(f"Seasonality decomposition for column {column} completed")
            return {"image": image}

        elif option == "correlation_heatmap":
            plt.figure(figsize=(10, 8))
            sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
            plt.title("Correlation Heatmap")
            image = plot_to_base64(plt)
            logger.info("Correlation heatmap generated")
            return {"image": image}

        elif option == "summary_statistics":
            summary = df.describe().to_dict()
            logger.info("Summary statistics calculated")
            return {"summary": summary}

        elif option == "rolling_mean":
            if column is None:
                raise HTTPException(status_code=400, detail="Column name is required for rolling mean.")
            plt.figure(figsize=(10, 5))
            rolling_mean = df[column].rolling(window=5).mean()
            plt.plot(df[column], label="Original")
            plt.plot(rolling_mean, label="Rolling Mean", linestyle="--")
            plt.legend()
            plt.title(f"Rolling Mean of {column}")
            image = plot_to_base64(plt)
            logger.info(f"Rolling mean plot for column {column} generated")
            return {"image": image}

        elif option == "histogram":
            if column is None:
                raise HTTPException(status_code=400, detail="Column name is required for histogram.")
            plt.figure(figsize=(10, 5))
            df[column].plot(kind='hist', bins=20)
            plt.title(f"Histogram of {column}")
            image = plot_to_base64(plt)
            logger.info(f"Histogram for column {column} generated")
            return {"image": image}

        elif option == "boxplot":
            if column is None:
                raise HTTPException(status_code=400, detail="Column name is required for boxplot.")
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=df, x=column)
            plt.title(f"Boxplot of {column}")
            image = plot_to_base64(plt)
            logger.info(f"Boxplot for column {column} generated")
            return {"image": image}

        else:
            logger.error(f"Invalid EDA option selected: {option}")
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
        data_store['data'] = df
        logger.info("Feature engineering applied successfully")
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
            image = plot_to_base64(plt)
            return {"summary": model_fit.summary().as_text(), "image": image}

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
            image = plot_to_base64(plt)
            return {"summary": model_fit.summary().as_text(), "image": image}

        elif model == "Prophet":
            df_prophet = df[["date", target_column]].rename(columns={"date": "ds", target_column: "y"})
            prophet_model = Prophet()
            prophet_model.fit(df_prophet)
            future = prophet_model.make_future_dataframe(periods=30)
            forecast = prophet_model.predict(future)
            plt.figure(figsize=(10, 5))
            prophet_model.plot(forecast)
            image = plot_to_base64(plt)
            return {"forecast": forecast.to_dict(), "image": image}

        elif model == "Boosting":
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
            plt.title("Boosting Model Predictions")
            image = plot_to_base64(plt)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            return {"predictions": predictions.tolist(), "image": image, "rmse": rmse, "r2": r2, "feature_importances": gb_model.feature_importances_.tolist()}

        elif model == "Random Forest":
            X = df.drop(columns=[target_column]).select_dtypes(include=['float', 'int']).fillna(0)
            y = df[target_column].fillna(0)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf_model = RandomForestRegressor(**params)
            rf_model.fit(X_train, y_train)
            predictions = rf_model.predict(X_test)
            plt.figure(figsize=(10, 5))
            plt.plot(y_test.values, label="True Values")
            plt.plot(predictions, label="Predictions", linestyle="--")
            plt.legend()
            plt.title("Random Forest Model Predictions")
            image = plot_to_base64(plt)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            return {"predictions": predictions.tolist(), "image": image, "rmse": rmse, "r2": r2, "feature_importances": rf_model.feature_importances_.tolist()}

        elif model == "LSTM":
            # LSTM requires reshaping the data for time-series modeling
            X = df.drop(columns=[target_column]).select_dtypes(include=['float', 'int']).fillna(0).values
            y = df[target_column].fillna(0).values
            X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshaping for LSTM
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            lstm_model = Sequential()
            lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
            lstm_model.add(Dense(1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
            predictions = lstm_model.predict(X_test)

            plt.figure(figsize=(10, 5))
            plt.plot(y_test, label="True Values")
            plt.plot(predictions, label="Predictions", linestyle="--")
            plt.legend()
            plt.title("LSTM Model Predictions")
            image = plot_to_base64(plt)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            return {"predictions": predictions.tolist(), "image": image, "rmse": rmse}

        else:
            raise HTTPException(status_code=400, detail="Invalid model selected.")

    except ValueError as ve:
        logger.error(f"Invalid input for model parameters: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input for model parameters: {ve}")
    except KeyError as ke:
        logger.error(f"Missing or invalid column: {ke}")
        raise HTTPException(status_code=400, detail=f"Missing or invalid column: {ke}")
    except Exception as e:
        logger.error(f"An error occurred while running the model: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while running the model: {e}")


@app.get("/health")
def health_check():
    logger.info("Health check endpoint called")
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting the application")
    uvicorn.run(app, host="0.0.0.0", port=8000)