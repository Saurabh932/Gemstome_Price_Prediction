import os
import sys
import numpy as np
import pandas as pd
import pymongo
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from configuration.configure import MongoDBConfig
from src.logger import logging
from src.exception import customexception


@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts", "raw.csv")
    train_data_path:str = os.path.join("artifacts", "train.csv")
    test_data_path:str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            logging.info("Reading Data from MongoDb")

            client = pymongo.MongoClient(MongoDBConfig.CLIENT)
            db = client["gemstone_data"] 
            collection = db["raw_data"]

            mongo_data = collection.find({})
            
            data = pd.DataFrame(list(mongo_data))
            pd.set_option('display.max_columns', None)
            
            os.makedirs(os.path.dirname(os.path.join(self.data_ingestion_config.raw_data_path)), exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info("Raw dataset imported and saved in artifact folder.")

            logging.info("Performing train_test_split")
            train_data, test_data = train_test_split(data, test_size=0.30)
            logging.info("Successfully performed train test split")

            train_data.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.data_ingestion_config.test_data_path, index=False)

            logging.info("Data Ingestion has Completed.")

            return self.data_ingestion_config.train_data_path, self.data_ingestion_config.test_data_path

        except Exception as e:
            raise customexception(e, sys)
        

if __name__ == "__main__":
    
    obj=DataIngestion()
    obj.initiate_data_ingestion()