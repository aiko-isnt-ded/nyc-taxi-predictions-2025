# General Libraries
import os
import pandas as pd
# Databricks Env
import pathlib
import pickle
from dotenv import load_dotenv
# Feature Engineering
from sklearn.feature_extraction import DictVectorizer
# Optimization
import math
import optuna
from optuna.samplers import TPESampler
# MLFlow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow import MlflowClient
# Modeling
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Evaluation Metrics
from sklearn.metrics import root_mean_squared_error
# Pipeline
from prefect import flow, task

# =======================
# Tasks
# =======================

@task(name="Read Data")
def read_data(file_path: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    df = pd.read_parquet(file_path)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task(name="Add Features")
def add_features(df_train: pd.DataFrame, df_val: pd.DataFrame):
    """Add features to the model"""
    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]  #['PULocationID', 'DOLocationID'] Combined
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    # Train
    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)
    y_train = df_train["duration"].values

    # Test
    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)
    y_val = df_val["duration"].values

    return X_train, X_val, y_train, y_val, dv

@task(name="RandomForest Tunning and Training")

@task(name="GradientBoosting Tunning and Training")

# =======================
# Pipeline Flow
# =======================

@flow(name="Main Challenger Flow")
def main_flow(year: int, month_train: str, month_val: str) -> None:
    """Main training pipeline for Challenger Models (RandomForest and Gradient Boosting)"""

    train_path = f"../data/green_tripdata_{year}-{month_train}.parquet"
    val_path = f"../data/green_tripdata_{year}-{month_val}.parquet"
    
    # Load .env file and experiment for Databricks
    load_dotenv(override=True)  
    EXPERIMENT_NAME = "/Users/viviana.toledo@iteso.mx/nyc-taxi-experiment-prefect"

    # Set MLFlow tracking to Databricks
    mlflow.set_tracking_uri("databricks")
    experiment = mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    # Load Data
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    # Transform Data
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)
    
    # Hyper-parameter Tunning
    best_params = hyper_parameter_tunning(X_train, X_val, y_train, y_val, dv)
    
    # Train
    train_best_model(X_train, X_val, y_train, y_val, dv, best_params)

    # Model Registry
    model_registry(EXPERIMENT_NAME)