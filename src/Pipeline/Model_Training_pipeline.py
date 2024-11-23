import os
import sys
from src.exception import CustomException
from src.logger import logging
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from src.Utils import evaluate_model
from dataclasses import dataclass
from src.Utils import save_object

@dataclass
class ModelConfig:
    model_obj_path = os.path.join('artifacts','model.pkl')

class ModelTraining:
    
    def __init__(self) -> None:
        self.modelconfig = ModelConfig()
    def initiate_model_training(self,train_data,test_data):
        try:
            
            

            logging.info("Data splitting Initiated")
            

            x_train,y_train,x_test,y_test = (train_data[:,:-1],
                                            train_data[:,-1],
                                            test_data[:,:-1],
                                           test_data[:,-1])
            logging.info("Model training started")
            models = {
                     'Logistic Regression': LogisticRegression(),
                     'K-Nearest Neighbors': KNeighborsClassifier(),
                     'Decision Tree': DecisionTreeClassifier(),
                     'Random Forest': RandomForestClassifier(),
                     'AdaBoost': AdaBoostClassifier(),
                     'Gradient Boosting': GradientBoostingClassifier(),
                     'Support Vector Machine': SVC(),
                     'XGBoost': XGBClassifier(),
                     'CatBoost': CatBoostClassifier(verbose=False)
                  }

            params = {'Logistic Regression':{
                     'penalty':['l2'],'C':[0.1,1,10]},
                     'K-Nearest Neighbors':{
                     'n_neighbors':[3,5,7,9],
                     'weights':['uniform', 'distance'],'p':[1,2]},
                     'Decision Tree':{
                     'criterion':['gini', 'entropy'],
                     'max_depth':[None, 5, 10,15,20]},'Random Forest':{
                     'n_estimators':[100,200,300],'max_depth':[None,5,10]},
                     'AdaBoost':{
                     'n_estimators':[50,100,200],
                     'learning_rate':[0.01,0.05,0.1]},
                     'Gradient Boosting':{
                     'n_estimators':[100,200,300],
                     'learning_rate':[0.01,0.05,0.1]},
                     'Support Vector Machine':{
                     'C':[0.1,0.2,0.5],
                     'kernel':['linear', 'rbf', 'poly']},
                     'XGBoost':{
                     'n_estimators':[100,200,300],
                     'learning_rate':[0.01,0.05,0.1]},
                     'CatBoost':{
                     'iterations':[100,200,300],
                     'learning_rate':[0.01,0.05,0.1]}}
            
            
            best_model,report = evaluate_model(x_train,x_test,y_train,y_test,models,params)
            print("=================== After Hyperparameter tuning =================== ")
            
            for model_name,score in report.items():

                print(f"{model_name}:{score}")


            logging.info("Saving The model")

            save_object(
                self.modelconfig.model_obj_path,
                best_model
            )
            return self.modelconfig.model_obj_path
        except Exception as e:
            raise CustomException(e,sys)
   
    
   



 
       
                  
              
      
         
         

