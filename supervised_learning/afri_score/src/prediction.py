#!/usr/bin/env python3
import streamlit as st
from sklearn import preprocessing
import pandas as pd 
import numpy as np
import pickle

model = pickle.load(open('models/model.pkl', 'rb'))
encoder_dict = pickle.load(open('models/encoder_dict.pkl', 'rb'))
columns = ['age','job','balance', 'day', 'month', 'duration']

def main():
    st.title('Credit Allocation predictor')
    html_temp =""""
    <div style = "background:#02524; padding:10px">
    <h2 style ="color:white;text-align:center;">Credit Allocation Predictor </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    age = st.text_input("Age", "0")
    job = st.selectbox("unemployed", "services", "management", "blue-collar", "self-employed", "technician", "entreprenuer", "admin.", "student", "housemaid", "retired", "unknown")
    month = st.selectbox("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    day = st.text_input("day", "0")
    balance = st.text_input("balance", "0")
    duration = st.text_input("duration", "0")

    if st.button("Predict"):
        features = [[age, job, month, day, balance, duration]]
        data = {'age': int(age), 'job':job, 'month': month, 'day':int(day), 'balance': int(balance), 'duration':int(duration)}
        print(data)
        df = pd.DataFrame([list(data.values())], columns=['age','job','month','day','balance','duration'])
        
        category_col = ['job', 'month']
        for cat in encoder_dict:
            for col in df.columns:
                le = preprocessing.LabelEncoder()
                if cat == col:
                    le.classes_ = encoder_dict[cat]
                    for unique_item in df[col].unique():
                        if unique_item not in le.classes_:
                            df[col] = ['Unknown' if x == unique_item else x for x in df[col]]
                        df[col] = le.transform(df[col])
        features_list = df.values.tolist()
        prediction = model.predict(features_list)
        output = int(prediction[0])
        if output == 1:
            text ="Yes"
        else:
            text = "No"
        st.success(f"{text}, to customer credit approval")

if __name__=='__main__': 
    main()
