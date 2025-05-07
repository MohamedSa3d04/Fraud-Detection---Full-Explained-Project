from joblib import load
from credit_fraud_utilites_eval import basic_metrics, evaluate_model_with_curves
from credit_fraud_utilites_data import UnderSampling_data, get_classes_dist
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_model_dict(file_path):
    model_dict = load(file_path)
    scaler = model_dict['scaler']
    threshold = model_dict['threshold']
    model = model_dict['model']
    return model_dict, scaler, threshold, model

def test(X, y, model, scaler, threshold):
    get_classes_dist(y)
    # Scaling the data
    X_scaled = scaler.transform(X)
    
    # Train the model
    predicted_proba = model.predict_proba(X_scaled)[:, 1]
    plt.figure()
    plt.scatter(x=range(len(predicted_proba)), y = predicted_proba)

    # print(sorted(predicted_proba))
    predicted = (predicted_proba >= threshold).astype(int)
    # Evaluate The model
    basic_metrics(y, predicted)
    plt.show()
    evaluate_model_with_curves(model, X_scaled, y)

# load the model
model_dict, scaler, threshold, model = get_model_dict('./models/vlcf_model_dict.joblib')

# load the data
test_df = pd.read_csv('./data/un_seen.csv')
X = test_df.drop('Class', axis=1)
y = test_df['Class']
# X, y = UnderSampling_data(X, y)
# test the model
test(X, y, model, scaler, threshold)



