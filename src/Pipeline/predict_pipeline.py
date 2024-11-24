from src.exception import CustomException
from src.logger import logging
from src.Utils import load_object
import pandas as pd
import os
import sys


class Predict:
    def __init__(self) -> None:
        pass
    def predict(self,features):
        try:
            logging.info("Loading model and preprocessor objects")

            model_path = os.path.join("artifacts",'model.pkl')

            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")

            model = load_object(model_path)

            preprocessor = load_object(preprocessor_path)

            scaled_data = preprocessor.transform(features)
            predictions = model.predict(scaled_data)[0]

            logging.info("Prediction Completed")

            logging.info(f"Predicted value is {predictions}")

            print(predictions)

            return predictions
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,age, sex, chest_pain_type,resting_blood_pressure,
           serum_cholestoral, fasting_blood_sugar,
           resting_electrocardiographic_results,maximum_heart_rate_achieved,
           exercise_induced_angina, oldpeak, slope_of_the_peak,
           colored_by_flourosopy,thal,):
        
        self.age = age
        self.sex = sex
        self.chest_pain_type = chest_pain_type
        self.resting_blood_pressure = resting_blood_pressure
        self.serum_cholestoral = serum_cholestoral
        self.fasting_blood_sugar = fasting_blood_sugar
        self.resting_electrocardiographic_results = resting_electrocardiographic_results
        self.maximum_heart_rate_achieved = maximum_heart_rate_achieved
        self.exercise_induced_angina = exercise_induced_angina
        self.oldpeak = oldpeak
        self.slope_of_the_peak = slope_of_the_peak
        self.colored_by_flourosopy = colored_by_flourosopy
        self.thal = thal
    
    def get_data_as_dataframe(self):
        try:
            data_dictionary = {
            'age':self.age, 'sex':self.sex,'chest pain type':self.chest_pain_type, 
            'resting blood pressure':self.resting_blood_pressure,
           'serum cholestoral':self.serum_cholestoral, 'fasting blood sugar':self.fasting_blood_sugar,
           'resting electrocardiographic results ':self.resting_electrocardiographic_results, 'maximum heart rate achieved':self.maximum_heart_rate_achieved,
           'exercise induced angina':self.exercise_induced_angina, 'oldpeak':self.oldpeak, ' slope of the peak':self.slope_of_the_peak,
           'colored by flourosopy':self.colored_by_flourosopy, 'thal':self.thal

            }

            
            return pd.DataFrame(data_dictionary)
        except Exception as e :
            raise CustomException(e,sys)




    

        
        