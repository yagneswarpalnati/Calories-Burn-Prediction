import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

model = joblib.load("models/best_model.pkl")

X_train = pd.read_csv("data/X_train.csv")

st.title("Calories Burn Prediction - Machine Learning Approach")

st.write("""
### Predict calories burned during exercise based on personal characteristics and workout data.
""")


gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=10, max_value=100, value=25)
height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
duration = st.number_input("Exercise Duration (minutes)", min_value=0, max_value=300, value=30)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=100)
body_temp = st.number_input("Body Temperature (°C)", min_value=30.0, max_value=45.0, value=36.5)

gender_encoded = 1 if gender == "Male" else 0


def predict_calories(gender, age, height, weight, duration, heart_rate, body_temp):
    input_data = pd.DataFrame([[gender, age, height, weight, duration, heart_rate, body_temp]], 
                              columns=X_train.columns)
    
    prediction = model.predict(input_data)
    return round(prediction[0], 2)


if st.button("Predict Calories Burned"):
    result = predict_calories(gender_encoded, age, height, weight, duration, heart_rate, body_temp)
    st.success(f"🔥 You have burned approximately **{result} kcal** 🔥")


st.subheader("📊 Calories Burn Pattern Analysis")


df = pd.read_csv("data/X_train.csv")


st.write("### 🔍 Relationship Between Features & Calories Burned")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=df["Duration"], y=df["Heart_Rate"], hue=df["Weight"], palette="coolwarm", ax=ax)
plt.xlabel("Exercise Duration (minutes)")
plt.ylabel("Heart Rate (bpm)")
st.pyplot(fig)


st.subheader("📈 Model Comparison (R2 Score)")
performance_img = "model_performance.png"
st.image(performance_img, caption="Comparison of Machine Learning Models", use_container_width=True)
