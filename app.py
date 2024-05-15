#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import tensorflow.keras
import wfdb
import os
from keras.models import load_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


# Load the models
cnn_model = load_model('best_model.h5')
cnn_lstm_model = load_model('best_model_1.h5')



# In[3]:


# Load test data
test_data = pd.read_csv('test_data.csv')


# In[4]:


label_dict = {
    0: 'Bundle branch block',
    1: 'Cardiomyopathy',
    2: 'Dysrhythmia',
    3: 'Healthy control',
    4: 'Heart failure (NYHA 2)',
    5: 'Heart failure (NYHA 3)',
    6: 'Heart failure (NYHA 4)',
    7: 'Hypertrophy',
    8: 'Myocardial infarction',
    9: 'Myocarditis',
    10: 'Palpitation',
    11: 'Stable angina',
    12: 'Unstable angina',
    13: 'Valvular heart disease',
}


# In[5]:


import pickle


# In[6]:

with open('model_rf.pkl', 'rb') as file:
    rf_model = pickle.load(file)


from tensorflow.keras.models import load_model,Model

app = Flask(__name__)



# Define the index route
@app.route('/')
def index():
    patients = test_data['Participant'].unique().tolist()
    models = ['CNN', 'CNN-LSTM', 'Random Forest']  # Model names
    return render_template('index.html', patients=patients, models=models)

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    patient = request.form['patient']
    model_name = request.form['model']
    
    # Filter test data for the selected patient
    patient_data = test_data[test_data['Participant'] == patient].drop(['Participant', 'Label'], axis=1)

    if model_name == 'CNN':
        #CNN model
        print(patient_data)
        X = np.array(patient_data).reshape(len(patient_data), patient_data.shape[1], 1)
        prediction_proba = cnn_model.predict(X)
        prediction = np.argmax(prediction_proba, axis=1)
    elif model_name == 'CNN-LSTM':
        #  CNN-LSTM model
        X = np.array(patient_data).reshape(len(patient_data), patient_data.shape[1], 1)
        prediction_proba = cnn_lstm_model.predict(X)
        prediction = np.argmax(prediction_proba, axis=1)
    elif model_name == 'Random Forest':
        # RF model
        X = np.array(patient_data).reshape(len(patient_data), patient_data.shape[1], 1)
        prediction_proba = cnn_model.predict(X)
        prediction = np.argmax(prediction_proba, axis=1) 
    prediction_class = label_dict[prediction[0]]  # Assuming prediction is a single value

    patients = test_data['Participant'].unique().tolist()
    
    return render_template('index.html', patients=patients, prediction=prediction_class, patient=patient, model=model_name)


# In[7]:


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)



# In[ ]:




