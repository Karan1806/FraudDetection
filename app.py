# app.py

import streamlit as st
import numpy as np
import pickle
import pandas as pd
import sklearn
# Load trained model pipeline
with open("fraud_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="🛡️ Fraud Detection", layout="centered")
st.title("🔍 Real-Time Fraud Detection App")

# ---- User Input ----
account_age = st.number_input("📅 Account Age (days)", min_value=0, value=100)
num_items = st.number_input("📦 Number of Items", min_value=1, value=1)
local_time = st.number_input("⏰ Local Time of Purchase (e.g. 13.5 = 1:30 PM)", min_value=0.0, max_value=23.99, step=0.5, value=12.0)

payment_age = st.number_input("💳 Payment Method Age (days)", min_value=0.0, step=1.0, value=90.0)
is_weekend = st.radio("📆 Is it Weekend?", ["No", "Yes"])
payment_method = st.selectbox("💰 Payment Method", ["creditcard", "paypal", "storecredit"])
category = st.selectbox("🛍️ Category", ["electronics", "food", "shopping"])

# ---- Prepare Input ----
isWeekend = 1 if is_weekend == "Yes" else 0

# Create input as a dict (same order as training features)
input_dict = {
    'accountAgeDays': account_age,
    'numItems': num_items,
    'localTime': local_time,
    'paymentMethodAgeDays': payment_age,
    'isWeekend': isWeekend,
    'paymentMethod': payment_method,
    'Category': category
}

# Convert to DataFrame
input_df = st.session_state.get("input_df", None)
input_df = st.session_state["input_df"] = pd.DataFrame([input_dict])

st.write("📊 Input Data Preview:", input_df)

# ---- Prediction ----
if st.button("🔍 Predict Fraud"):
    prediction = model.predict(input_df)[0]
    probs = model.predict_proba(input_df)[0]
    if len(probs) == 2:
        probability = probs[1]
    else:
     probability = 0 

    if prediction == 1:
        st.error(f"⚠️ Fraud Detected! (Risk: {probability*100:.2f}%)")
    else:
        st.success(f"✅ Legitimate Transaction. (Risk: {probability*100:.2f}%)")
