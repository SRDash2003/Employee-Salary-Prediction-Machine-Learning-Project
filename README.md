# 💼 Employee-Salary-Prediction-Machine-Learning-Project

A Machine Learning-powered web application that predicts an individual's estimated income based on demographic and work-related features. Built using **Scikit-learn**, **Pandas**, and deployed via **Streamlit**, this project demonstrates the complete ML lifecycle from data preprocessing to model training and user-facing prediction.

---

## 📌 Project Overview

This project utilizes a modified version of the UCI Adult Income dataset. Unlike the original classification task (i.e., predicting income `<=50K` or `>50K`), the target column has been enhanced to contain **realistic numeric income estimates**, enabling a **regression-based approach**.

The final Streamlit web app allows users to:
- Input personal and work details via UI elements (sliders, dropdowns)
- Get estimated annual income as a prediction
- Upload batch CSV files for multiple predictions
- Download the results as a CSV

---

## 🚀 Features

- Full end-to-end pipeline: from data cleaning to model deployment
- Data preprocessing using Label Encoding, outlier removal, and feature selection
- Multiple regression models tested and compared
- Streamlit app with both **single** and **batch** prediction support
- Clean and user-friendly UI with mapped dropdown values
- Input template available for batch predictions

---

## 🛠️ Tech Stack

| Category              | Tools/Libs Used                         |
|----------------------|-----------------------------------------|
| Language             | Python 3.12                             |
| ML Libraries         | scikit-learn, pandas, numpy             |
| Visualization        | matplotlib, seaborn (for analysis)      |
| Model Deployment     | Streamlit                              |
| Serialization        | joblib                                 |

---

## 🧠 Machine Learning Pipeline

1. **Data Preprocessing**:
   - Dropped redundant or irrelevant columns
   - Removed rows with undefined or noisy values
   - Label encoded categorical columns
   - Removed statistical outliers based on domain knowledge

2. **Modeling**:
   - Trained & evaluated 5 regression models:
     - Linear Regression
     - Random Forest Regressor
     - K-Nearest Neighbors
     - Support Vector Regressor
     - Gradient Boosting Regressor
   - Compared models based on:
     - Mean Absolute Error (MAE)
     - R² Score

3. **Best Model**:
   - Gradient Boosting was selected (R² ≈ 0.31, MAE ≈ ₹11,962)
   - Serialized as `best_model_regression.pkl` using joblib

---

## 📊 Visualization & Analysis

Some data insights were obtained via charts like:
- Boxplot (for age/outlier detection)
- Bar charts (for feature value counts)
- Heatmaps (for correlation analysis)
- MAE/R² comparison graphs across models

---

## 🖥️ Streamlit Web App

### To launch the app locally:

```bash
streamlit run app.py
```

---

## 🔧 App Features:

### Dropdowns for workclass, marital status, occupation, etc., mapped to encoded values
- Shows input preview with human-readable labels
- Predicts estimated income based on inputs
- CSV batch upload for bulk predictions
- Output CSV download functionality

---

## 📁 Input CSV Template
To use the batch prediction feature, structure your CSV as follows:
```
age,workclass,educational-num,marital-status,occupation,gender,hours-per-week,native-country
29,3,13,2,8,1,40,39
```
### ⚠️ Ensure all values are numeric and aligned with the model's encoded mappings.

---

## 📈 Future Enhancements
- Add toggle to switch between models for comparison
- Incorporate SHAP or LIME for interpretability
- Enable dynamic model retraining from UI
- Use Docker for containerized deployment
- Add login/auth for user-specific dashboards

---

## 📚 References
- UCI Adult Income Dataset
- Scikit-learn Documentation
- Streamlit Documentation
- Pandas Documentation
- Internship resource notebooks and project instructions

---

# 🤝 Contributors

- Soumya Ranjan Dash – Developer & Project Lead
- Internship guided by __Edunet Foundation__ in collaboration with __IBM Skillsbuild__




