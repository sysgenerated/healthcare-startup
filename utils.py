import joblib
import pandas as pd


def csv_to_json(csv):
    pd.read_csv(csv).to_json('train.json', orient='index')


def predict():
    # hard coding names because this isn't Prod
    model = joblib.load('saved_model.joblib')
    df = pd.read_csv('train.zip')
    return model.predict(df)
