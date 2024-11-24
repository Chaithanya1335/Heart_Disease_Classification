from flask import Flask, render_template, request, redirect, url_for
from src.Pipeline.predict_pipeline import Predict, CustomData
from src.exception import CustomException
import os
import sys

app = Flask(__name__)



@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Collect form data
                age = float(request.form.get('age')),
                sex = float(request.form.get('sex')),
                chest_pain_type =  request.form.get('chest pain type'),
                resting_blood_pressure = float(request.form.get('resting blood pressure')),
                serum_cholestoral = float(request.form.get('serum cholestoral')),
                fasting_blood_sugar = float(request.form.get('fasting blood sugar')),
                resting_electrocardiographic_results = request.form.get('resting electrocardiographic results '),
                maximum_heart_rate_achieved = float(request.form.get('maximum heart rate achieved')),
                exercise_induced_angina= float(request.form.get('exercise induced angina')),
                oldpeak = float(request.form.get('oldpeak')),
                slope_of_the_peak = request.form.get('slope of the peak'),
                colored_by_flourosopy= float(request.form.get('colored by flourosopy')),
                thal = request.form.get('thal')
            
            # Convert data into DataFrame format for the model
                input_data_df = CustomData(age,sex,chest_pain_type,resting_blood_pressure,
                                      serum_cholestoral,fasting_blood_sugar,resting_electrocardiographic_results,
                                      maximum_heart_rate_achieved,exercise_induced_angina,oldpeak,slope_of_the_peak,colored_by_flourosopy,thal).get_data_as_dataframe()

                

            # Make prediction
                model_prediction = Predict().predict(input_data_df)
                prediction = "Yes" if model_prediction == 1 else "No"

                return redirect(url_for('display', res=prediction))

        except Exception as e:
           
            raise CustomException(e,sys)

    return render_template('index.html', prediction=prediction)

@app.route('/display/<res>', methods=['GET'])
def display(res):
    return render_template('display.html', res=res)

if __name__ == '__main__':
    app.run(debug=True)
