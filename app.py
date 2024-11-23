from flask import Flask, render_template, request,redirect,url_for
import pandas as pd
import pickle
import os

app = Flask(__name__)    

# Example: Columns expected by the model
columns = ['age', 'sex', 'chest pain type', 'resting blood pressure',
           'serum cholestoral', 'fasting blood sugar',
           'resting electrocardiographic results ', 'maximum heart rate achieved',
           'exercise induced angina', 'oldpeak', ' slope of the peak',
           'colored by flourosopy', 'thal']

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        try:
            # Retrieve form data
            age = float(request.form.get('age'))
            sex = float(request.form.get('sex'))
            chest_pain_type = request.form.get('chest pain type')
            resting_blood_pressure = float(request.form.get('resting blood pressure'))
            serum_cholestoral = float(request.form.get('serum cholestoral'))
            fasting_blood_sugar = float(request.form.get('fasting blood sugar'))
            resting_electrocardiographic_results = request.form.get('resting electrocardiographic results ')
            maximum_heart_rate_achieved = float(request.form.get('maximum heart rate achieved'))
            exercise_induced_angina = float(request.form.get('exercise induced angina'))
            oldpeak = float(request.form.get('oldpeak'))
            slope_of_the_peak = request.form.get('slope of the peak')
            colored_by_flourosopy = float(request.form.get('colored by flourosopy'))
            thal = request.form.get('thal')

            form_data = {
            'age':age, 'sex':sex,'chest pain type':chest_pain_type, 
            'resting blood pressure':resting_blood_pressure,
           'serum cholestoral':serum_cholestoral, 'fasting blood sugar':fasting_blood_sugar,
           'resting electrocardiographic results ':resting_electrocardiographic_results, 'maximum heart rate achieved':maximum_heart_rate_achieved,
           'exercise induced angina':exercise_induced_angina, 'oldpeak':oldpeak, ' slope of the peak':slope_of_the_peak,
           'colored by flourosopy':colored_by_flourosopy, 'thal':thal

            }

            # Convert to DataFrame
            input_data_df = pd.DataFrame([form_data])

            print("Form Data:", form_data)


            # Log DataFrame for debugging
            print("Input DataFrame:\n", input_data_df)

            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            model_path = os.path.join("artifacts","xgboost.pkl")
            with open(preprocessor_path,'rb') as file:
                preprocessor = pickle.load(file)
            
            with open(model_path,'rb') as file:
                xgb_model = pickle.load(file)

            # Preprocess data
            input_data_processed = preprocessor.transform(input_data_df)

            # Make prediction
            model_prediction = xgb_model.predict(input_data_processed)[0]
            prediction = "Yes" if model_prediction == 1 else "No"

            return redirect(url_for('display',res=prediction))

        except Exception as e:
            print("Error during prediction:", e)
            prediction = "Error: Unable to process the request."

    return render_template('index.html', prediction=prediction)

@app.route('/display/<res>',methods = ['GET'])
def display(res):
    return render_template('display.html',res=res)


if __name__ == '__main__':
    app.run(debug=True)
