"# End to End Pipeline For Heart_disease_Classification  " 

"How to Run ??"

"""
=> First create a virtual Environment and activate environment by following commands:

"replace name of the environment with your name what ever you want"

        conda create -p name_of_the_environment python==3.8 -y 
        conda activate name_of_the_environment/ 


=> second execute this command 

                pip install -r requirements.txt

=> Third execute This command 
                
                python DataIngestion.py 


"# You need to give entire path of DataIngestion.py file # "
 
"# This will Train the model and save data Preprocessing and model objects later used for prediction." 

"## IF u want to test model predictions with custom Data Try the below comand "

                python app.py
