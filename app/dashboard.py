import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add the 'src' directory to the path to import custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from model_loader import load_rf_model
from data_processor import load_all_data, preprocess_data
from predict_classification import prepare_features_for_prediction, predict_risk


# --- Configuration ---
st.set_page_config(layout="wide", page_title="Predictive Maintenance Dashboard")


# --- 1. Load Data and Model (Cached for efficiency) ---
@st.cache_resource
def get_data_and_model():
    """Loads and preprocesses all data, and loads the model once."""
    try:
        # Resolve absolute paths relative to project root (one level up from app/)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        data_dir = os.path.join(project_root, 'data')
        model_path = os.path.join(project_root, 'models', 'predictive_maintenance_model.pkl')

        data_dict = load_all_data(data_path=data_dir + os.sep)
        df_preprocessed = preprocess_data(data_dict)
        model = load_rf_model(path=model_path)
        return df_preprocessed, model
    except Exception as e:
        st.error(
            f"Failed to load required assets. Ensure CSVs are in 'data/' and model is in 'models/'. Error: {e}"
        )
        st.stop()


df_preprocessed, model = get_data_and_model()


# --- Streamlit Application Layout ---
st.title("Predictive Maintenance Dashboard")
st.markdown("---")

# --- Sidebar for Machine Selection ---
machine_ids = sorted(df_preprocessed['machineID'].unique().tolist())
if not machine_ids:
    st.error("No machine data found. Check your data loading.")
    st.stop()

st.sidebar.header("Machine Selection")
selected_machine = st.sidebar.selectbox(
    "Select Equipment ID:",
    options=machine_ids,
)


# --- Prepare Data for Selected Machine ---
X_current, datetimes = prepare_features_for_prediction(df_preprocessed, selected_machine)
probabilities, max_risk = predict_risk(model, X_current)

machine_meta = df_preprocessed[df_preprocessed['machineID'] == selected_machine].iloc[0]
machine_age = machine_meta['age']

# --- Risk Level Logic ---
if max_risk > 15:
    risk_level = "HIGH"
    color = "red"
elif max_risk > 5:
    risk_level = "MEDIUM"
    color = "orange"
else:
    risk_level = "LOW"
    color = "green"


# --- 2. Key Metrics ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Machine Model", f"Model {machine_meta['model']}")

with col2:
    st.metric("Machine Age (Years)", f"{machine_age:.0f}")

with col3:
    st.metric(
        "Maximum Failure Risk Score",
        f"{max_risk:.2f}%",
        help="Highest predicted probability of failure in the dataset.",
    )

with col4:
    st.markdown(
        f"""
    <div style='background-color: {color}; padding: 10px; border-radius: 5px;'>
        <h3 style='color: white; text-align: center; margin: 0;'>Risk Status</h3>
        <p style='color: white; text-align: center; font-size: 24px; font-weight: bold; margin: 0;'>{risk_level}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

st.markdown("---")


# --- 3. Time Series Visualizations ---
st.header("Hourly Telemetry and Predicted Risk")

# Combine probabilities with the original data for plotting
plot_data = X_current.copy()
plot_data['datetime'] = datetimes
plot_data['Failure Risk (%)'] = probabilities * 100
plot_data = plot_data.set_index('datetime')

# Sensor Readings
st.subheader("Sensor Readings (Volt, Rotate, Pressure, Vibration)")
sensor_cols = ['volt', 'rotate', 'pressure', 'vibration']
if not plot_data.empty:
    st.line_chart(plot_data[sensor_cols])
else:
    st.info("No telemetry available for the selected machine.")

# Predicted Risk over Time
st.subheader("Predicted Failure Risk Over Time")
if not plot_data.empty:
    st.line_chart(plot_data[['Failure Risk (%)']], color="#ff4b4b")
else:
    st.info("No predictions available for the selected machine.")

# Optional: Display Raw Data Table
if st.sidebar.checkbox("Show Raw Data"):
    st.subheader("Raw Data Sample")
    st.dataframe(plot_data.head(50))


