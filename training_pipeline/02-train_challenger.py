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
def run_random_forest(X_train, X_val, y_train, y_val, dv):
    """Tune Hyperparameters and Train RandomForest model"""

    # Definir la función objetivo para Optuna
    def objective(trial: optuna.trial.Trial):
        # Hiperparámetros MUESTREADOS por Optuna en CADA trial.
        # Nota: usamos log=True para emular rangos log-uniformes (similar a loguniform).
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 150),
            "max_depth": trial.suggest_int("max_depth", 4, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 40, 200),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 4, 100),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", math.exp(-3), math.exp(-1), log=True),
            "max_features": trial.suggest_int("max_features", 30, 120),
            "ccp_alpha": trial.suggest_float("ccp_alpha",   math.exp(-4), math.exp(-3), log=True),
            "random_state": 42,                      
        }

        # Run anidado para dejar rastro de cada trial en MLflow
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "randomforest")          # etiqueta informativa
            mlflow.log_params(params)                               # registra hiperparámetros del trial

            # Entrenamiento con el conjunto de validación
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)

            # Predicción y métrica en validación
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            # Registrar la métrica principal
            mlflow.log_metric("rmse", rmse)

            # La "signature" describe la estructura esperada de entrada y salida del modelo:
            # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
            # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
            signature = infer_signature(X_val, y_pred)

            # Guardar el modelo del trial como artefacto en MLflow.
            mlflow.sklearn.log_model(
                sk_model = rf,
                name="model",
                input_example=X_val[:5],
                signature=signature,
            )

        # Optuna minimiza el valor retornado
        return rmse

    # Log the model
    mlflow.sklearn.autolog(log_models=False)
    
    # Create Optuna Study
    sampler = TPESampler(seed=42)                                           # Sampler
    study = optuna.create_study(direction="minimize", sampler=sampler)      # direction="minimize" to minimize the RMSE

    # Execute the optimizer
    with mlflow.start_run(run_name="RandomForest Hyperparameter Optimization (Optuna)", nested=True):           # Nested runs inside father model (RandomForest)
        study.optimize(objective, n_trials=10)      # 10 tries for optimization

        # Get and register the best hyperparameters
        best_params = study.best_params
        # Asegurar tipos/campos fijos (por claridad y consistencia)
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["seed"] = 42
        best_params["objective"] = "reg:squarederror"

        # Log the parameters 
        mlflow.log_params(best_params)

        # Father run labels (experiment metadata)
        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "randomforest",
            "feature_set_version": 1,
        })

        # Train a Final Model with the selected hyperparameters
        rf = RandomForestRegressor(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
            max_features=best_params["max_features"],
            ccp_alpha=best_params["ccp_alpha"],
            random_state=42
        )
        # Fit the model
        rf.fit(X_train, y_train)

        # Evaluar y registrar la métrica final en validación
        y_pred = rf.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Store additional artifacts (e.g. the preprocessor)
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # La "signature" describe la estructura esperada de entrada y salida del modelo:
        # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
        # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
        # Si X_val es la matriz dispersa (scipy.sparse) salida de DictVectorizer:
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)

        # Para que las longitudes coincidan, usa el mismo slice en y_pred
        signature = infer_signature(input_example, y_val[:5])

        # Guardar el modelo del trial como artefacto en MLflow.
        mlflow.sklearn.log_model(
        sk_model=rf,                    # Trained RandomForestRegressor
        name="model",                   # Folder inside MLflow run to store the model
        input_example= input_example,   # First few rows of validation data
        signature=signature,            
    )
        
    return None

@task(name="GradientBoosting Tunning and Training")
def run_gradient_boosting(X_train, y_train, X_val, y_val, dv):
    """Tune Hyperparameters and Train GradientBoosting model"""

    # Definir la función objetivo para Optuna
    def objective(trial: optuna.trial.Trial):
        # Hiperparámetros MUESTREADOS por Optuna en CADA trial.
        # Nota: usamos log=True para emular rangos log-uniformes (similar a loguniform).
        params = {
            "learning_rate": trial.suggest_float("learning_rate", math.exp(-2), math.exp(2), log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 250),
            "max_depth": trial.suggest_int("max_depth", 4, 100),
            "min_samples_split": trial.suggest_int("min_samples_split", 30, 150),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 4, 100),
            "min_weight_fraction_leaf": trial.suggest_float("min_weight_fraction_leaf", math.exp(-3), math.exp(-1), log=True),
            "max_features": trial.suggest_int("max_features", 20, 150),
            "alpha": trial.suggest_float("alpha",   math.exp(-4), math.exp(-3), log=True),
            "random_state": 42,                      
        }

        # Run anidado para dejar rastro de cada trial en MLflow
        with mlflow.start_run(nested=True):
            mlflow.set_tag("model_family", "randomforest")  # etiqueta informativa
            mlflow.log_params(params)                       # registra hiperparámetros del trial

            # Entrenamiento con el conjunto de validación
            rf = GradientBoostingRegressor(**params)
            rf.fit(X_train, y_train)

            # Predicción y métrica en validación
            y_pred = rf.predict(X_val)
            rmse = root_mean_squared_error(y_val, y_pred)

            # Registrar la métrica principal
            mlflow.log_metric("rmse", rmse)

            # La "signature" describe la estructura esperada de entrada y salida del modelo:
            # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
            # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
            signature = infer_signature(X_val, y_pred)

            # Guardar el modelo del trial como artefacto en MLflow.
            mlflow.sklearn.log_model(
                sk_model = rf,
                name="model",
                input_example=X_val[:5],
                signature=signature,
            )

        # Optuna minimiza el valor retornado
        return rmse
    # Log the model
    mlflow.sklearn.autolog(log_models=False)

    # Create Optuna Study
    sampler = TPESampler(seed=42)                                           # Sampler
    study = optuna.create_study(direction="minimize", sampler=sampler)      # direction="minimize" to minimize the RMSE

    # Execute the optimizer
    with mlflow.start_run(run_name="GradientBoosting Hyperparameter Optimization (Optuna)", nested=True):
        study.optimize(objective, n_trials=10)

        # Get and register the best hyperparameters
        best_params = study.best_params
        # Asegurar tipos/campos fijos (por claridad y consistencia)
        best_params["max_depth"] = int(best_params["max_depth"])
        best_params["seed"] = 42
        best_params["objective"] = "reg:squarederror"

        # Log the parameters
        mlflow.log_params(best_params)

        # Father run labels (experiment metadata)
        mlflow.set_tags({
            "project": "NYC Taxi Time Prediction Project",
            "optimizer_engine": "optuna",
            "model_family": "gradientboosting",
            "feature_set_version": 1,
        })

        # Train a Final Model with the selected hyperparameters
        gb = GradientBoostingRegressor(
            learning_rate=best_params["learning_rate"],
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            min_samples_split=best_params["min_samples_split"],
            min_samples_leaf=best_params["min_samples_leaf"],
            min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
            max_features=best_params["max_features"],
            alpha=best_params["alpha"],
            random_state=42
        )

        # Fit the model
        gb.fit(X_train, y_train)

        # Evaluar y registrar la métrica final en validación
        y_pred = gb.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)

        # Store additional artifacts (e.g. the preprocessor)
        pathlib.Path("preprocessor").mkdir(exist_ok=True)
        with open("preprocessor/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocessor/preprocessor.b", artifact_path="preprocessor")

        # La "signature" describe la estructura esperada de entrada y salida del modelo:
        # incluye los nombres, tipos y forma (shape) de las variables de entrada y el tipo de salida.
        # MLflow la usa para validar datos en inferencia y documentar el modelo en el Model Registry.
        # Si X_val es la matriz dispersa (scipy.sparse) salida de DictVectorizer:
        feature_names = dv.get_feature_names_out()
        input_example = pd.DataFrame(X_val[:5].toarray(), columns=feature_names)

        # Para que las longitudes coincidan, usa el mismo slice en y_pred
        signature = infer_signature(input_example, y_val[:5])

        # Guardar el modelo del trial como artefacto en MLflow.
        mlflow.sklearn.log_model(
        sk_model=gb,                    # Trained GradientBoostingRegressor
        name="model",                   # Folder inside MLflow run to store the model
        input_example= input_example,   # First few rows of validation data
        signature=signature,            
    )
    return None

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
    
    # Tune and Train RandomForest
    run_random_forest(X_train, X_val, y_train, y_val, dv)
    
    # Tune and Train Gradient Boosting
    run_gradient_boosting(X_train, X_val, y_train, y_val, dv)

    # Model Registry
    model_registry(EXPERIMENT_NAME)