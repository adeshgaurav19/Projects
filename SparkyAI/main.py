from fastapi import FastAPI
import os
import io
import time
import logging
import numpy as np
import pandas as pd
import joblib
import rasterio
import geopandas as gpd
import plotly.express as px
import plotly.graph_objs as go
from pmdarima import auto_arima
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from prophet import Prophet
from fastapi.staticfiles import StaticFiles

# Setup FastAPI app
app = FastAPI()

# Serve static files (e.g., plot images, HTML) from a folder
app.mount("/get-plot", StaticFiles(directory="plots"), name="get-plot")

# Setup logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log"),
    ]
)

# Create logger
logger = logging.getLogger(__name__)

# Enable CORS for frontend-backend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to save plots and processed data
PLOT_DIR = "plots"
DATA_DIR = "data"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


import os
import io
import time
import logging
import numpy as np
import pandas as pd
import joblib
import rasterio
import geopandas as gpd
import plotly.express as px
import plotly.graph_objs as go
from pmdarima import auto_arima
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Setup FastAPI app
app = FastAPI(title="Solar Energy Prediction Platform")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("solar_app.log"),
    ]
)
logger = logging.getLogger(__name__)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories for storage
PLOT_DIR = "plots"
DATA_DIR = "data"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

import pandas as pd
import numpy as np
from pmdarima import auto_arima
import plotly.graph_objs as go
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assume PLOT_DIR is defined
PLOT_DIR = './plots'
os.makedirs(PLOT_DIR, exist_ok=True)

import pandas as pd
import numpy as np
from pmdarima import auto_arima
import plotly.graph_objs as go
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Assume PLOT_DIR is defined
PLOT_DIR = './plots'
os.makedirs(PLOT_DIR, exist_ok=True)

class CustomJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder to handle Timestamp and other non-serializable types
    """
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


from prophet import Prophet
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import os

class SolarEnergyAnalyzer:
    def __init__(self):
        self.geospatial_model = None
        self.demand_forecast_model = None
        self.generation_forecast_model = None
        self.data = None

    def preprocess_time_series_data(self, df):
        """
        Preprocess time series data for forecasting
        """
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            if df['date'].isnull().any():
                print(f"Warning: There are {df['date'].isnull().sum()} invalid dates after conversion.")
            df.set_index('date', inplace=True)
        else:
            raise ValueError("'date' column is missing in the DataFrame")

        df.fillna(method='ffill', inplace=True)
        print(f"Processed DataFrame:\n{df.head()}")
        return df

    def train_demand_forecast_model(self, df):
        """
        Train demand forecast model using Prophet
        """
        df_prophet = df[['demand']].reset_index().rename(columns={'date': 'ds', 'demand': 'y'})

        demand_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.5,  # Increase changepoint prior scale
            seasonality_prior_scale=15.0,  # Increase seasonality flexibility
            holidays_prior_scale=15.0,
            seasonality_mode='multiplicative'
        )

        # Add custom seasonality if needed
        demand_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Add holidays if applicable
        # holidays = make_holidays_df(df_prophet, ['US'])
        # demand_model.add_country_holidays(country_name='US')

        demand_model.fit(df_prophet)
        return demand_model

    def train_generation_forecast_model(self, df):
        """
        Train solar generation forecast model using Prophet
        """
        df_prophet = df[['solar_generation']].reset_index().rename(columns={'date': 'ds', 'solar_generation': 'y'})

        generation_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.5,
            seasonality_prior_scale=15.0,
            holidays_prior_scale=15.0,
            seasonality_mode='multiplicative'
        )

        # Add custom seasonality if needed
        generation_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # Add holidays if applicable
        # holidays = make_holidays_df(df_prophet, ['US'])
        # generation_model.add_country_holidays(country_name='US')

        generation_model.fit(df_prophet)
        return generation_model

    def generate_comprehensive_forecast(self, demand_model, generation_model, forecast_periods=60):
        """
        Generate comprehensive forecast for demand and solar generation
        """
        future_demand = demand_model.make_future_dataframe(periods=forecast_periods, freq='M')
        demand_forecast = demand_model.predict(future_demand)

        future_generation = generation_model.make_future_dataframe(periods=forecast_periods, freq='M')
        generation_forecast = generation_model.predict(future_generation)

        forecast_dates = future_demand['ds'][-forecast_periods:]

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'demand_forecast': demand_forecast['yhat'][-forecast_periods:].values,
            'generation_forecast': generation_forecast['yhat'][-forecast_periods:].values
        })

        forecast_data = []
        for _, row in forecast_df.iterrows():
            forecast_data.append({
                'date': row['date'].isoformat(),
                'demand_forecast': float(row['demand_forecast']),
                'generation_forecast': float(row['generation_forecast'])
            })

        return forecast_data

    def create_interactive_forecast_plot(self, original_data, forecast_data):
        """
        Create interactive Plotly plot for demand and generation forecast
        """
        forecast_dates = [pd.Timestamp(item['date']) for item in forecast_data]
        forecast_demand = [item['demand_forecast'] for item in forecast_data]
        forecast_generation = [item['generation_forecast'] for item in forecast_data]

        historical_demand = go.Scatter(
            x=original_data.index,
            y=original_data['demand'],
            mode='lines',
            name='Historical Demand'
        )

        historical_generation = go.Scatter(
            x=original_data.index,
            y=original_data['solar_generation'],
            mode='lines',
            name='Historical Solar Generation'
        )

        forecast_demand_trace = go.Scatter(
            x=forecast_dates,
            y=forecast_demand,
            mode='lines',
            name='Demand Forecast',
            line=dict(dash='dot')
        )

        forecast_generation_trace = go.Scatter(
            x=forecast_dates,
            y=forecast_generation,
            mode='lines',
            name='Solar Generation Forecast',
            line=dict(dash='dot')
        )

        layout = go.Layout(
            title='Energy Demand and Solar Generation Forecast',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Energy (MWh)'),
            height=600
        )

        fig = go.Figure(data=[
            historical_demand,
            historical_generation,
            forecast_demand_trace,
            forecast_generation_trace
        ], layout=layout)

        plot_path = os.path.join('plots', 'energy_forecast_plot.html')
        fig.write_html(plot_path)
        return plot_path

    def analyze_energy_balance(self, forecast_data):
        """
        Analyze energy balance between demand and generation
        """
        demand_forecasts = [item['demand_forecast'] for item in forecast_data]
        generation_forecasts = [item['generation_forecast'] for item in forecast_data]

        surplus_deficit = np.array(generation_forecasts) - np.array(demand_forecasts)

        return {
            'total_surplus_mwh': float(np.sum(surplus_deficit)),
            'avg_monthly_surplus_mwh': float(np.mean(surplus_deficit)),
            'months_with_surplus': int(np.sum(surplus_deficit > 0)),
            'months_with_deficit': int(np.sum(surplus_deficit < 0))
        }

    def process_energy_forecasting(self, file_path):
        """
        Process energy forecasting from a file
        """
        df = pd.read_csv(file_path)
        self.data = self.preprocess_time_series_data(df)
        self.demand_forecast_model = self.train_demand_forecast_model(self.data)
        self.generation_forecast_model = self.train_generation_forecast_model(self.data)
        forecast_data = self.generate_comprehensive_forecast(self.demand_forecast_model, self.generation_forecast_model)
        plot_path = self.create_interactive_forecast_plot(self.data, forecast_data)
        energy_balance = self.analyze_energy_balance(forecast_data)

        return {
            'forecast_data': forecast_data,
            'plot_path': plot_path,
            'energy_balance': energy_balance
        }



# Route for energy forecasting
@app.post("/energy-forecast/")
async def energy_forecast(file: UploadFile):
    """
    Process energy demand and solar generation forecast
    """
    try:
        # Save uploaded file
        file_path = os.path.join(DATA_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Initialize analyzer and process data
        analyzer = SolarEnergyAnalyzer()
        result = analyzer.process_energy_forecasting(file_path)
        
        return JSONResponse(content={
            "message": "Energy forecast completed successfully",
            "forecast_data": result['forecast_data'],
            "plot_url": f"/get-plot/{os.path.basename(result['plot_path'])}",
            "energy_balance": result['energy_balance']
        })
    
    except Exception as e:
        logger.error(f"Energy forecast processing error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"error": str(e)}
        )

@app.get("/get-plot/{plot_name}")
async def get_plot(plot_name: str):
    """
    Serve interactive plot
    """
    plot_path = os.path.join(PLOT_DIR, plot_name)
    if os.path.exists(plot_path):
        return FileResponse(plot_path, media_type="text/html")
    return JSONResponse(status_code=404, content={"error": "Plot not found"})

import os
import io
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import plotly.express as px
import plotly.graph_objs as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import json
from shapely.geometry import Point
import logging
import traceback

# Set up logging configuration
logging.basicConfig(level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class SolarFeasibilityAnalyzer:
    def __init__(self, landsat_file: str, geojson_file: str):
        """
        Initialize the solar feasibility analyzer
        """
        self.landsat_file = landsat_file
        self.geojson_file = geojson_file
        self.combined_data = None
        self.model = None

    def extract_geospatial_data(self) -> pd.DataFrame:
        """
        Extract and process geospatial data from the provided JSON structure.
        """
        try:
            # Load JSON data
            with open(self.landsat_file, 'r') as f:
                geojson_data = json.load(f)

            # Extract metadata
            metadata = geojson_data["metadata"]
            crs = metadata["crs"]
            transform = metadata["transform"]
            width = metadata["width"]
            height = metadata["height"]

            # Extract coordinates
            coordinates = geojson_data["coordinates"]
            longitude = coordinates["longitude"]
            latitude = coordinates["latitude"]

            # Extract band data
            bands = geojson_data["bands"]
            red_band_data = bands["red_band"]["data"]
            nir_band_data = bands["nir_band"]["data"]

            # Ensure arrays have the same length
            min_length = min(len(longitude), len(latitude), len(red_band_data), len(nir_band_data))
            longitude = longitude[:min_length]
            latitude = latitude[:min_length]
            red_band_data = red_band_data[:min_length]
            nir_band_data = nir_band_data[:min_length]

            # Calculate NDVI
            ndvi = (np.array(nir_band_data) - np.array(red_band_data)) / (
                np.array(nir_band_data) + np.array(red_band_data)
            )

            # Create DataFrame
            points_data = {
                "longitude": longitude,
                "latitude": latitude,
                "red_band_value": red_band_data,
                "nir_band_value": nir_band_data,
                "ndvi": ndvi.tolist(),
            }
            points = pd.DataFrame(points_data)

            # Filter valid data (NDVI > 0)
            points = points.dropna().query("ndvi > 0")

            # Create GeoDataFrame
            points["geometry"] = points.apply(lambda row: Point(row["longitude"], row["latitude"]), axis=1)
            points_gdf = gpd.GeoDataFrame(points, geometry="geometry", crs=crs)

            # Load GeoJSON for regulatory or additional information
            geo_df = gpd.read_file(self.geojson_file)

            # Perform spatial join
            combined_gdf = gpd.sjoin(points_gdf, geo_df, how="inner", predicate="intersects")

            # Convert the resulting GeoDataFrame back to DataFrame and drop the geometry column
            self.combined_data = pd.DataFrame(combined_gdf.drop(columns="geometry"))

            return self.combined_data

        except Exception as e:
            error_message = f"Error extracting geospatial data: {str(e)}"
            logging.error(error_message)
            logging.error("Traceback: %s", traceback.format_exc())
            raise ValueError(error_message)

    def generate_regulatory_features(self) -> pd.DataFrame:
        """
        Generate synthetic regulatory and economic features
        """
        if self.combined_data is None:
            raise ValueError("Geospatial data not extracted. Call extract_geospatial_data() first.")

        np.random.seed(42)
        size = len(self.combined_data)
        
        regulatory_data = pd.DataFrame({
            "solar_potential_index": np.random.uniform(0.5, 1.0, size=size),
            "land_use_efficiency": np.random.uniform(0.6, 0.95, size=size),
            "environmental_constraint_score": np.random.uniform(0.1, 0.8, size=size),
            "infrastructure_readiness": np.random.uniform(0.4, 1.0, size=size),
            "target_variable": np.random.uniform(50000, 500000, size=size)  # Solar plant value/potential
        })

        # Combine with existing data
        merged_data = pd.concat([self.combined_data.reset_index(drop=True), 
                                  regulatory_data.reset_index(drop=True)], axis=1)
        
        return merged_data

    def prepare_regression_data(self, merged_data: pd.DataFrame, target_column: str):
        """
        Prepare data for regression analysis
        """
        try:
            # Select features and target
            X = merged_data.drop(columns=[target_column])
            y = merged_data[target_column]

            # Handle categorical and numerical features
            X = pd.get_dummies(X, drop_first=True)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            # Train Random Forest Regressor model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)

            # Predictions and evaluation
            predictions = self.model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)

            return {
                "mean_squared_error": mse,
                "r2_score": r2,
                "predictions": predictions.tolist()
            }

        except Exception as e:
            error_message = f"Error preparing regression data: {str(e)}"
            logging.error(error_message)
            logging.error("Traceback: %s", traceback.format_exc())
            raise ValueError(error_message)

    def generate_interactive_plots(self, merged_data: pd.DataFrame):
        """
        Generate interactive Plotly visualizations and save them as HTML files.
        """
        try:
            plots_dir = 'plots'  # Directory for saving plots
            os.makedirs(plots_dir, exist_ok=True)  # Ensure the directory exists

            plots = {}

            # Scatter plot of latitude vs longitude colored by solar potential
            scatter_plot = px.scatter_mapbox(
                merged_data, 
                lat='latitude', 
                lon='longitude', 
                color='solar_potential_index',
                hover_name='latitude',
                zoom=7,
                height=600,
                mapbox_style="open-street-map"
            )
            scatter_plot_path = os.path.join(plots_dir, 'scatter_map.html')
            scatter_plot.write_html(scatter_plot_path)
            plots['location_potential_map'] = scatter_plot_path

            # Feature importance bar plot
            if hasattr(self, 'model'):
                # Ensure the model features match the merged_data columns
                feature_columns = merged_data.drop(columns=['target_variable', 'latitude', 'longitude']).columns
                if len(feature_columns) == len(self.model.feature_importances_):
                    feature_importance = pd.DataFrame({
                        'feature': feature_columns,
                        'importance': self.model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    bar_plot = px.bar(
                        feature_importance.head(10), 
                        x='feature', 
                        y='importance',
                        title='Top 10 Feature Importances'
                    )
                    bar_plot_path = os.path.join(plots_dir, 'feature_importance.html')
                    bar_plot.write_html(bar_plot_path)
                    plots['feature_importance'] = bar_plot_path
                else:
                    logging.warning("Feature length mismatch between model and merged data. Skipping feature importance plot.")

            # Distribution of solar potential
            hist_plot = px.histogram(
                merged_data, 
                x='solar_potential_index', 
                marginal='box',
                title='Distribution of Solar Potential Index'
            )
            hist_plot_path = os.path.join(plots_dir, 'solar_potential_distribution.html')
            hist_plot.write_html(hist_plot_path)
            plots['solar_potential_distribution'] = hist_plot_path

            logging.info("Plots saved: %s", plots)
            return plots
        except Exception as e:
            error_message = f"Error generating interactive plots: {str(e)}"
            logging.error(error_message)
            logging.error("Traceback: %s", traceback.format_exc())
            return {}


def process_solar_feasibility(landsat_file: str, geojson_file: str):
    """
    Main processing function to tie everything together.
    """
    try:
        # Initialize analyzer
        analyzer = SolarFeasibilityAnalyzer(landsat_file, geojson_file)

        # Extract geospatial data
        geospatial_data = analyzer.extract_geospatial_data()

        # Generate regulatory features
        merged_data = analyzer.generate_regulatory_features()

        # Run regression
        regression_results = analyzer.prepare_regression_data(
            merged_data, 
            target_column='target_variable'
        )

        # Proceed with generating plots
        interactive_plots = analyzer.generate_interactive_plots(merged_data)

        # Prepare final response
        return {
            "message": "Solar feasibility analysis completed successfully",
            "data_preview": merged_data.head(10).to_dict(),
            "regression_results": regression_results,
            "interactive_plots": interactive_plots
        }

    except Exception as e:
        logging.error("Solar feasibility analysis failed: %s", str(e))
        logging.error("Traceback: %s", traceback.format_exc())
        return {"error": str(e), "details": traceback.format_exc()}


@app.post("/process-geospatial/")
async def process_geospatial(file: UploadFile, geojson: UploadFile):
    logging.info(f"Received files: {file.filename}, {geojson.filename}")
    
    if not file.filename.endswith(".json") or not geojson.filename.endswith(".geojson"):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected a .json and .geojson file.")

    try:
        # Save uploaded files
        landsat_path = os.path.join(DATA_DIR, file.filename)
        geojson_path = os.path.join(DATA_DIR, geojson.filename)
        
        logging.info(f"Saving Landsat file to: {landsat_path}")
        with open(landsat_path, "wb") as f:
            f.write(await file.read())
        
        logging.info(f"Saving GeoJSON file to: {geojson_path}")
        with open(geojson_path, "wb") as f:
            f.write(await geojson.read())

        # Process solar feasibility
        results = process_solar_feasibility(landsat_path, geojson_path)

        # Safely construct plot URLs
        plot_urls = {}
        for plot_name in ['location_potential_map', 'feature_importance', 'solar_potential_distribution']:
            if plot_name in results['interactive_plots']:
                plot_filename = os.path.basename(results['interactive_plots'][plot_name])
                plot_urls[plot_name] = f"/get-plot/{plot_filename}"
                logging.info(f"Plot URL for {plot_name}: {plot_urls[plot_name]}")


        # Return the response with plot URLs
        return JSONResponse(
            content={
                "message": "Geospatial data processed successfully",
                "data_preview": results['data_preview'],
                "regression_results": results['regression_results'],
                "interactive_plots": plot_urls  # Use plot URLs in the response
            }
        )
    
    except Exception as e:
        logger.error(f"Geospatial processing error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

    
def cleanup_old_files():
    """
    Remove files older than 24 hours from plots directory
    """
    current_time = time.time()
    for filename in os.listdir(PLOT_DIR):
        file_path = os.path.join(PLOT_DIR, filename)
        file_age = current_time - os.path.getmtime(file_path)
        if file_age > 24 * 60 * 60:  # Older than 24 hours
            os.remove(file_path)
            logger.info(f"Deleted old file: {filename}")

@app.on_event("startup")
async def startup_event():
    """
    Perform cleanup on application startup
    """
    logger.info("Cleaning up old files on startup...")
    cleanup_old_files()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
