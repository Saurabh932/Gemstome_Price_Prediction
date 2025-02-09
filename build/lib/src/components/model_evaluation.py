import os
import sys
import numpy as np

from src.logger import logging
from src.exception import customexception
from src.utils import load_object

from dataclasses import dataclass
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn

@dataclass
class ModelEvalutionConfig:
    pass

class ModelEvaluation():
    def __init__(self):
        logging.info("Evalutation has started")

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        logging.info("Evalution metrics has captured")
        return rmse, mae, r2

    def initiate_model_evalution(self, test_array):
        try:
            X_test, y_test = (test_array[:,:-1], test_array[:,-1])
            
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            # mlflow.set_registry_uri("")
            logging.info("Model has register")

            tracking_url_type_store = urlparse(mlflow.get_registry_uri()).scheme

            print(tracking_url_type_store)

            with mlflow.start_run():
                prediction = model.predict(X_test)

                (rmse, mae, r2) = self.eval_metrics(y_test, prediction)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                # Model registry does not work with file store
                if tracking_url_type_store != "file":

                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow

                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")

                else:
                    mlflow.sklearn.log_model(model, "model")

            return rmse, mae, r2


        except Exception as e:
            logging.info()
            raise customexception(e, sys)