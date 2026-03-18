import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ------------------------
# App Title
# ------------------------
st.header (" OYO STATE COLLEGE OF AGRICULTURE AND TECHNOLOGY")
st.title("🌾 Intelligent Farm Management System (Nigeria)")
st.write("Predict irrigation needs based on farm conditions in selected states.")


# ------------------------
# Load Dataset
# ------------------------
@st.cache_data
def load_data():
    return pd.read_csv("nigeria_farm_data.csv")

data = load_data()

# ------------------------
# State Selection
# ------------------------
st.subheader("🌍 Select State")
states = data['state'].unique()
selected_state = st.selectbox("Choose a state", states)

# Filter data for selected state
state_data = data[data['state'] == selected_state]

# Display selected state's capital and area
capital = state_data['capital'].iloc[0]
area = state_data['area'].iloc[0]
st.write(f"**Capital:** {capital}")
st.write(f"**Area (km²):** {area}")

# ------------------------
# User Inputs
# ------------------------
st.subheader("🔮 Predict Irrigation Need")
temperature = st.slider("Temperature (°C)", int(state_data['temperature'].min()), int(state_data['temperature'].max()), int(state_data['temperature'].mean()))
humidity = st.slider("Humidity (%)", int(state_data['humidity'].min()), int(state_data['humidity'].max()), int(state_data['humidity'].mean()))
soil = st.selectbox("Soil Moisture", ["Low", "High"])
soil_moisture = 0 if soil == "Low" else 1

# ------------------------
# Train Model on Selected State
# ------------------------
X = state_data[['temperature', 'humidity', 'soil_moisture']]
y = state_data['irrigation']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# ------------------------
# Predict User Input
# ------------------------
if st.button("Predict"):
    input_df = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'soil_moisture': [soil_moisture]
    })
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠ Irrigation Recommended")
    else:
        st.success("✅ No Irrigation Needed")

# ------------------------
# Dataset Preview for Selected State
# ------------------------
st.subheader(f"📊 Dataset Preview ({selected_state})")
st.dataframe(state_data)

# ------------------------
# Model Accuracy
# ------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader("📈 Model Accuracy")
st.success(f"Accuracy: {accuracy*100:.2f}%")

# ------------------------
# Dynamic Prediction Chart
# ------------------------
st.subheader("📊 Predicted Irrigation vs Temperature")
temp_range = pd.DataFrame({
    'temperature': list(range(int(state_data['temperature'].min()), int(state_data['temperature'].max())+1)),
    'humidity': [humidity]*(int(state_data['temperature'].max()) - int(state_data['temperature'].min()) + 1),
    'soil_moisture': [soil_moisture]*(int(state_data['temperature'].max()) - int(state_data['temperature'].min()) + 1)
})
predictions = model.predict(temp_range)
temp_range['predicted_irrigation'] = predictions

st.line_chart(temp_range.set_index('temperature')['predicted_irrigation'])

# ------------------------
# Predicted Irrigation Distribution
# ------------------------
st.subheader("📊 Predicted Irrigation Distribution (0 = No, 1 = Yes)")
irrigation_count = pd.Series(predictions).value_counts().sort_index()
st.bar_chart(irrigation_count)