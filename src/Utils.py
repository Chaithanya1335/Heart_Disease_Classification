import os 
import sys
import pickle
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score

def save_object(file_path,object):
    try:
        with open(file_path,'wb') as file:
            pickle.dump(object,file)
        logging.info("Object saved ")
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_model(x_train,x_test,y_train,y_test,models,params):
    try:
        report = {}

        print("=================== Before Hyperparameter tuning =================== ")
        for model_name,model in models.items():
            model.fit(x_train,y_train)
            predict = model.predict(x_test)
            accurracy = accuracy_score(predict,y_test)
            
            print(f"{model_name}:{accurracy}")
            gs = GridSearchCV(model,params[model_name],cv=5,scoring='accuracy')
            gs.fit(x_train,y_train)
            report[model_name] = gs.best_score_
            
        logging.info("Model Trainig completed")
        best_score = max(score for _,score in report.items())
        best_model = [model_name for model_name,score in report.items() if score==best_score]

        logging.info(f"Found Best model {best_model} with accuracy score :{best_score} ")

        best_model = gs.best_estimator_
        

        return (best_model,report)
    except Exception as e:
        raise CustomException(e,sys)

