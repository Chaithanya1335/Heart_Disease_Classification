from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from src.Utils import save_object
import pandas as pd
import numpy as np
import sys
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self) -> None:
        self.DataTransformationConfig = DataTransformationConfig()
    def initiate_data_Transformation(self,train_path,test_path):
        logging.info("Data Transformation started")

        try:
            
            logging.info("Train and Test Data Reading")
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

         
            logging.info("Data Transformation started")

            

            train_data['sex'] = train_data['sex'].map({'Male':0,'Female':1})
            train_data['target'] = train_data['target'].map({'No':0,'Yes':1})
            train_data['exercise induced angina'] = train_data['exercise induced angina'].map({'No':0,'Yes':1})
            test_data['fasting blood sugar'] = test_data['fasting blood sugar'].map({False:0,True:1})

            test_data['sex'] = test_data['sex'].map({'Male':0,'Female':1})
            test_data['target'] = test_data['target'].map({'No':0,'Yes':1})
            test_data['exercise induced angina'] = test_data['exercise induced angina'].map({'No':0,'Yes':1})
            test_data['fasting blood sugar'] = test_data['fasting blood sugar'].map({False:0,True:1})

            input_train_data = train_data.drop(columns=['target'],axis=1)
            target_train_data = train_data['target']

            input_test_data = test_data.drop(columns=['target'],axis=1)
            target_test_data = test_data['target']



            numerical_columns = input_train_data.select_dtypes(exclude = 'object').columns.to_list()
            categorical_columns = input_train_data.select_dtypes(include = 'object').columns.to_list()
            
            num_pipeline = Pipeline(steps=(
                ('imputer',SimpleImputer(strategy='mean')),
                ('scaler',StandardScaler())
            ))

            cat_pipeline = Pipeline(steps=(
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ))

            preprocessor = ColumnTransformer((
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
            ))

            
            input_train_data_new = preprocessor.fit_transform(input_train_data)

            input_test_data = test_data.drop(columns=['target'],axis=1)
            target_test_data = test_data['target']
            input_test_data_new = preprocessor.transform(input_test_data)

            train_data_new = np.c_[input_train_data_new,np.array(target_train_data)]

            test_data_new = np.c_[input_test_data_new,np.array(target_test_data)]

            logging.info("Data Transformation Completed")

            logging.info("Saving the preprocessor object")
            save_object(
                file_path=self.DataTransformationConfig.preprocessor_obj_path,
                object=preprocessor
            )

            

            return train_data_new,test_data_new
        except Exception as e:
            raise CustomException(e,sys)
    
