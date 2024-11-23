from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import train_test_split
from src.Components.DataTransformation import DataTransformation
from src.Components.DataTransformation import DataTransformationConfig
from src.Pipeline.Model_Training_pipeline import ModelTraining
from src.Pipeline.Model_Training_pipeline import ModelConfig
import os
import sys 
import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_path = os.path.join('artifacts','train.csv')
    test_path = os.path.join('artifacts','test.csv')
    raw_path = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self) :
        self.DataIngestionConfig = DataIngestionConfig()
    
    def intiate_data_ingestion(self):
        logging.info("Data Reading as DataFrame")
        try:
            data = pd.read_csv('D:\Heart_disease_Classification\Raw dataset of heart disease.csv')

            logging.info("Data Readed as DataFrame")

            logging.info('Train Test Split Intiated')

            train_data,test_data = train_test_split(data,test_size=0.2,random_state=42)

            os.makedirs(os.path.dirname(self.DataIngestionConfig.train_path),exist_ok=True)

            train_data.to_csv(self.DataIngestionConfig.train_path)

            test_data.to_csv(self.DataIngestionConfig.test_path)

            data.to_csv(self.DataIngestionConfig.raw_path)

            logging.info("Data Ingestion Completed")

            return (self.DataIngestionConfig.train_path,self.DataIngestionConfig.test_path)
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ =='__main__':
    train_path,test_path = DataIngestion().intiate_data_ingestion()
    train_data,test_data = DataTransformation().initiate_data_Transformation(train_path,test_path)
    model_path = ModelTraining().initiate_model_training(train_data,test_data)