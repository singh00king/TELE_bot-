import pickle
import pandas as pd

def predict_kyphosis(age:int, number:int, start:int)-> int:
    """ This function takes age, number, and start as input and returns the predicted kyphosis class (0 or 1) using the trained model. """
    
    model = pickle.load(open("model.pkl", "rb"))
    preprocessing = pickle.load(open("preprocessing.pkl", "rb"))
    patient = pd.DataFrame({"Age": [age], "Number": [number], "Start": [start]})
    patient[['Age']] = preprocessing["age_scaler"].transform(patient[['Age']])
    prediction = model.predict(patient)
    return preprocessing["label_encoder"].inverse_transform(prediction)[0]