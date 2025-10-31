# Importar Librerías
import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
from dotenv import load_dotenv
import os

# ============================
# Startup and Loading
# ============================

# Carga las variables del archivo .env
load_dotenv(override=True)  

# Set databricks
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

EXPERIMENT_NAME = "/Users/viviana.toledo@iteso.mx/nyc-taxi-experiment-prefect"

# Search for stored runs in Databricks by RMSE
run_ = mlflow.search_runs(order_by=['metrics.rmse ASC'],
                          output_format="list",
                          experiment_names= [EXPERIMENT_NAME]
                          )[0]

# Search for run IDs
run_id = run_.info.run_id
run_uri = f"runs:/{run_id}/preprocessor"

# Download artifacts (preprocessor)
client.download_artifacts(
    run_id=run_id,
    path='preprocessor',
    dst_path='.'
)

# Load preprocessor
with open("preprocessor/preprocessor.b", "rb") as f_in:
    dv = pickle.load(f_in)

# Select model
model_name = "workspace.default.nyc-taxi-model"
alias = "champion"
model_uri = f"models:/{model_name}@{alias}"

# Load the model
champion_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)

# ============================
# Preprocessing & Prediction
# ============================

def preprocess(input_data):
    input_dict = {
        'PU_DO': input_data.PULocationID + "_" + input_data.DOLocationID,
        'trip_distance': input_data.trip_distance,
    }
    return dv.transform(input_dict)

# Prediction
def predict(input_data):
    X_val = preprocess(input_data)

    return champion_model.predict(X_val)

# ============================
# API
# ============================

app = FastAPI()

# Body Parameter with Input Features
class InputData(BaseModel):
    PULocationID: str
    DOLocationID: str
    trip_distance: float

# Endpoint to return predictions
@app.post("/api/v1/predict")
def predict_endpoint(input_data: InputData):
    result = predict(input_data)[0]
    return {"prediction": float(result)}

