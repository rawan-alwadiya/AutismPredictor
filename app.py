import streamlit as st
import pandas as pd
import joblib
import numpy as np


le = joblib.load("LabelEncoder.pkl")  
model = joblib.load("ANN.pkl")


st.set_page_config(page_title="Autism Predictor", page_icon="🧠", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 Autism Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Machine Learning Based Screening Tool</h3>", unsafe_allow_html=True)
st.write("Please provide the information below to predict the likelihood of Autism Spectrum Disorder (ASD).")


a_scores = [f"A{i}_Score" for i in range(1, 11)]


binary_cols = {
    "jundice": "Jaundice at Birth",
    "austim": "Family History of Autism"
}


categorical_cols = {"gender": "Gender"}

col1, col2, col3 = st.columns(3)


a_values = {}
for idx, feature in enumerate(a_scores):
    col = [col1, col2, col3][idx % 3]
    a_values[feature] = col.selectbox(f"{feature}", ["No", "Yes"])


age = col1.number_input("Age", min_value=1, max_value=120, value=30)
result = col2.number_input("Screening Result", min_value=0, max_value=10, value=5)


binary_values = {}
for idx, (colname, label) in enumerate(binary_cols.items()):
    col = [col1, col2, col3][(idx+10) % 3] 
    binary_values[colname] = col.selectbox(label, ["No", "Yes"])


gender = col3.selectbox("Gender", ["Male", "Female"])
gender_raw = "m" if gender == "Male" else "f"

if st.button("Predict ASD"):
    
    a_scores_input = [1 if a_values[feat] == "Yes" else 0 for feat in a_scores]

    
    binary_input = [1 if binary_values[b] == "Yes" else 0 for b in binary_cols.keys()]

    
    gender_encoded = le.transform([gender_raw])[0]

    
    input_data = pd.DataFrame([a_scores_input + [age, gender_encoded] + binary_input + [result]],
                              columns=a_scores + ["age", "gender", "jundice", "austim", "result"])
    
    prob = model.predict(input_data)[0][0]   # if shape is (1,1)
    
    prediction = 1 if prob > 0.5 else 0

    # prediction = model.predict(input_data)[0]

    
    if prediction == 1:
        st.success("🔹 The model predicts: **High likelihood of Autism Spectrum Disorder (ASD)**.")
    else:
        st.info("🔹 The model predicts: **Low likelihood of Autism Spectrum Disorder (ASD)**.")
