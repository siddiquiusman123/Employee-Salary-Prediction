import pandas as pd
import streamlit as st
import joblib

st.title("Employee Salary Prediction App")

# load The dataset , model , scalar , encoder

dataset = pd.read_csv('salary_data.csv')
model = joblib.load("Salary_prd_model.pkl")

# User Input
age = st.number_input("Age",min_value=10 , max_value=100)
gender = st.radio("Gender",['Male', 'Female', 'Other'])
education = st.selectbox("Education Level", ["Bachelor's", "Master's", 'PhD', 'High School'])
job_title = st.selectbox("Job Title",dataset["Job Title"].unique())
year_of_exp = st.number_input("Years of Experience",min_value=0.0 , max_value=50.0 , step=0.5)

columnss = ["Age","Gender","Education Level","Job Title","Years of Experience"]
input_data = [[age , gender , education , job_title , year_of_exp]]
input_df = pd.DataFrame(input_data, columns=columnss)


# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.success(f"Estimated Salary : {prediction[0]:.2f} ₹")