# 📊 Telecom Customer Churn Prediction — Machine Learning Project

This project builds a **machine learning model to predict telecom customer churn**, helping identify users likely to leave a service provider.  
It covers **data preprocessing, feature engineering, model training, hyperparameter tuning, and ensemble stacking** for improved accuracy.

---
![sreenshot](https://github.com/user-attachments/assets/e6cbdc75-68e7-4235-903e-afc670682075)

## 🚀 Features
- 📁 Data cleaning and preprocessing (missing values, encoding, scaling)
- ⚙️ Model training with **Logistic Regression**, **Random Forest**, and **XGBoost**
- 🧠 Ensemble learning using **Voting** and **Stacking Classifiers**
- 🔍 Hyperparameter tuning with **GridSearchCV**
- 📊 Exploratory Data Analysis (EDA) and **feature importance visualization**
- 💬 Interactive churn prediction interface using **IPyWidgets**

---

## 🧩 Tech Stack
- **Language:** Python  
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `ipywidgets`  
- **Techniques:** Data preprocessing, Feature scaling, One-hot encoding, Model evaluation, Ensemble learning  

---

## 📦 How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/telecom-churn-prediction.git
cd telecom-churn-prediction

# Install required dependencies
pip install pandas numpy scikit-learn xgboost matplotlib ipywidgets

**Then open the notebook or run the script:**

python telecom_churn.py

🧠 Workflow Overview

Data Loading & Cleaning:

Handle missing values and incorrect data types

Encode categorical variables using LabelEncoder and OneHotEncoder

Feature Scaling:

Standardize numerical columns (Age, Tenure, MonthlyCharges, TotalCharges)

Model Training & Comparison:

Train multiple models and evaluate using accuracy scores

Model Optimization:

Grid search for best hyperparameters

Ensemble Modeling:

Combine models using Voting and Stacking to improve predictions

Interactive Prediction:

Simple interface for predicting churn probability for new customers

📈 Sample Results
Model	Accuracy
Logistic Regression	~0.80
Random Forest	~0.83
XGBoost	~0.85
Stacking Ensemble	~0.87 ✅
📊 Visual Insights

Churn distribution

Feature correlations with churn

Boxplots of key numeric variables

Feature importance chart using Random Forest

💡 Learning Outcomes

This project demonstrates:

End-to-end data science workflow

Ensemble learning and stacking strategies

Model evaluation and interpretability

Building interactive ML tools with widgets

🌟 Future Enhancements

Deploy the model using Streamlit or Gradio

Integrate SHAP for explainable AI

Add automated reporting dashboards

Train on larger, real-world datasets
