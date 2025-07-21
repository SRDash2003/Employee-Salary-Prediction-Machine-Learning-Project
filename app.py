import streamlit as st
import pandas as pd
import joblib

# Load trained regression model
model = joblib.load("best_model_regression.pkl")

st.set_page_config(page_title="Employee Salary Predictor 💰", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Prediction App")
st.markdown("Estimate the expected salary of an employee based on input attributes.")

# Mapping dictionaries (based on your value_counts data)
workclass_map = {
    "Private": 3,
    "Self-emp-not-inc": 5,
    "Local-gov": 1,
    "State-gov": 2,
    "Without-pay": 6,
    "Self-emp-inc": 4,
    "Federal-gov": 0
}

marital_map = {
    "Never-married": 2,
    "Married-civ-spouse": 4,
    "Divorced": 0,
    "Separated": 5,
    "Married-spouse-absent": 6,
    "Widowed": 3,
    "Married-AF-spouse": 1
}

occupation_map = {
    "Prof-specialty": 8,
    "Craft-repair": 2,
    "Exec-managerial": 1,
    "Adm-clerical": 0,
    "Sales": 9,
    "Tech-support": 6,
    "Other-service": 5,
    "Machine-op-inspct": 7,
    "Transport-moving": 11,
    "Handlers-cleaners": 4,
    "Farming-fishing": 10,
    "Priv-house-serv": 3
}

gender_map = {"Male": 1, "Female": 0}

country_map = {
    "United-States": 39,
    "Mexico": 0,
    "Philippines": 26,
    "Germany": 30,
    "Canada": 11,
    "Cuba": 2,
    "India": 19,
    "Puerto-Rico": 33,
    "El-Salvador": 9,
    "South": 35,
    "China": 3,
    "Columbia": 5,
    "Jamaica": 23,
    "Japan": 24,
    "Italy": 22,
    "Dominican-Republic": 8,
    "Vietnam": 40,
    "Haiti": 31,
    "Guatemala": 4,
    "Honduras": 6,
    "Trinadad&Tobago": 36,
    "Laos": 14,
    "Peru": 20,
    "Cambodia": 12,
    "Iran": 29,
    "Nicaragua": 13,
    "Poland": 27,
    "Portugal": 32,
    "Ecuador": 7,
    "Scotland": 21,
    "France": 10,
    "Thailand": 37,
    "England": 1,
    "Yugoslavia": 17,
    "Taiwan": 28,
    "Outlying-US(Guam-USVI-etc)": 38,
    "Hong": 41,
    "Greece": 34,
    "Hungary": 25,
    "Ireland": 18,
    "Holand-Netherlands": 15,
    "Other": 16
}

education_map = {
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-acdm": 11,
    "Assoc-voc": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Doctorate": 15,
    "Prof-school": 16
}

# Sidebar Inputs
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 80, 30)
workclass = st.sidebar.selectbox("Workclass", list(workclass_map.keys()))
educ_label = st.sidebar.selectbox("Education Level", list(education_map.keys()))
marital_status = st.sidebar.selectbox("Marital Status", list(marital_map.keys()))
occupation = st.sidebar.selectbox("Occupation", list(occupation_map.keys()))
gender = st.sidebar.radio("Gender", list(gender_map.keys()))
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
country = st.sidebar.selectbox("Country", list(country_map.keys()))

# Input DataFrame (aligned to model features)
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass_map[workclass]],
    'educational-num': [education_map[educ_label]],
    'marital-status': [marital_map[marital_status]],
    'occupation': [occupation_map[occupation]],
    'gender': [gender_map[gender]],
    'hours-per-week': [hours_per_week],
    'native-country': [country_map[country]]
})

# Display DataFrame
disp_df = pd.DataFrame({
    'age': [age],
    'workclass': workclass,
    'educational-num': educ_label,
    'marital-status': marital_status,
    'occupation': occupation,
    'gender': gender,
    'hours-per-week': hours_per_week,
    'native-country': country
})

# Show input
st.write("### 🧾 Your Selected Inputs")
st.write(disp_df)
st.write("### 🧠 Processed Inputs for Model")
st.write(input_df)

# Predict Salary
if st.button("📊 Predict Salary"):
    prediction = model.predict(input_df)
    st.success(f"🧮 Estimated Salary: ₹{int(prediction[0]):,} per year")

# Optional: Batch prediction via CSV upload
st.markdown("---")
st.markdown("### 📂 Batch Salary Estimation")
uploaded_file = st.file_uploader("Upload a CSV file with employee data", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['Estimated_Income'] = batch_preds
    st.write("✅ Salary Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Salary Predictions", csv, file_name='estimated_incomes.csv', mime='text/csv')

# 📥 Input CSV Template
st.markdown("#### 📄 Download Input CSV Template")

template_df = pd.DataFrame({
    'age': [30],
    'workclass': [3],
    'educational-num': [10],
    'marital-status': [2],
    'occupation': [8],
    'gender': [1],
    'hours-per-week': [40],
    'native-country': [39]
})

csv = template_df.to_csv(index=False).encode('utf-8')
st.download_button("Download Template CSV", csv, file_name="input_template.csv", mime='text/csv')





