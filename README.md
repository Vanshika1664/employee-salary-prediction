# 💼 Employee Salary Prediction App

This is a machine learning web app built with Streamlit that predicts an employee's **annual and monthly salary in USD** based on input features like gender, education level, job title, age, and years of experience.

<p align="center">
  <img src="https://img.shields.io/badge/Built%20With-Streamlit-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Model-Random%20Forest-green?style=flat-square" />
  <img src="https://img.shields.io/badge/Language-Python-blue?style=flat-square" />
</p>

---

## 🚀 Features

- Predicts monthly and annual salary in **USD**
- Interactive form to collect input details
- Built using a **Random Forest Regression** model
- Supports deployment on Streamlit Cloud / Hugging Face

---

## 🧠 Model Training

The model was trained using the following features:

- Gender
- Education Level
- Job Title
- Age
- Years of Experience

Label encoding was used for categorical variables and the model was trained using `RandomForestRegressor` from `scikit-learn`.

---

## You can run the file code using the following command:
streamlit run app.py

## 🧾 Requirements
- Python 3.8+
- streamlit
- pandas
- scikit-learn
- joblib

## 👩‍💻 Author
Made by Vanshika Tomar as part of a AIML internship project.
