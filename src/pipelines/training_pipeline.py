import os
import sys
from dataclasses import dataclass
import numpy as np
from src.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.logger import logging
from src.exception import customexception

@dataclass
class TrainingPipelineConfig:
    pass

class TrainingPipeline:
    def __init__(self):
        logging.info("Training Pipeline has started.")
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()

    def start_data_ingestion(self):
        logging.info("Started data ingestion process.")
        try:
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logging.info("Completed data ingestion process.")
            return train_data_path, test_data_path
        except Exception as e:
            logging.error(f"Error in data ingestion process: {e}")
            raise customexception(e, sys)

    def start_data_transformation(self, train_data_path, test_data_path):
        logging.info("Started data transformation process.")
        try:
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
            logging.info("Completed data transformation process.")
            return train_arr, test_arr
        except Exception as e:
            logging.error(f"Error in data transformation process: {e}")
            raise customexception(e, sys)

    def start_model_training(self, train_arr, test_arr):
        logging.info("Started model training process.")
        try:
            model_trainer = ModelTrainer()
            best_model_name, best_r2_score = model_trainer.initate_model_training(train_arr, test_arr)
            logging.info(f"Completed model training process. Best Model: {best_model_name}, R2 Score: {best_r2_score}")
            return best_model_name, best_r2_score
        except Exception as e:
            logging.error(f"Error in model training process: {e}")
            raise customexception(e, sys)

    def start_model_evaluation(self, test_arr):
        logging.info("Started model evaluation process.")
        try:
            model_evaluation = ModelEvaluation()
            evaluation_results = model_evaluation.initiate_model_evalution(test_arr)
            logging.info(f"Completed model evaluation process. Evaluation Results: {evaluation_results}")
            return evaluation_results
        except Exception as e:
            logging.error(f"Error in model evaluation process: {e}")
            raise customexception(e, sys)

    def run_pipeline(self):
        try:
            train_data_path, test_data_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(train_data_path, test_data_path)
            best_model_name, best_r2_score = self.start_model_training(train_arr, test_arr)
            evaluation_results = self.start_model_evaluation(test_arr)
            logging.info(f"Pipeline execution completed successfully. Best Model: {best_model_name} with R2 Score: {best_r2_score}")
            logging.info(f"Evaluation Results: {evaluation_results}")
        except Exception as e:
            logging.error(f"Exception occurred in pipeline execution: {e}")
            raise customexception(e, sys)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
