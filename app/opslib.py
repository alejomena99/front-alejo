import joblib
import os
import numpy as np
import pandas as pd
import logging
import json

logging.basicConfig(level=logging.INFO)
cwd = os.path.abspath(os.path.dirname(__file__))

def load_model():
    given_path = "model"
    model_path = os.path.abspath(os.path.join(cwd, given_path))
    
    model = joblib.load(os.path.abspath(
        os.path.join(model_path,
                     'model.joblib')))
    return model

def formatting(prediction):
    return np.around(prediction, 0) 

def predict(instances): 
    model = load_model()    
    predictions = []
    for instance in instances:
        df = pd.DataFrame.from_dict([json.loads(instance)])
        prediction = model.predict(df)
        prediction = formatting(prediction)
        predictions.append(prediction[0])
        
    return predictions