# Predictive Maintenance System for Oil & Gas Equipment (PSU Internship Portfolio)

## Project Goal
Implement an end-to-end Predictive Maintenance (PdM) system to transition from reactive/scheduled maintenance to condition-based, predictive maintenance. This system predicts the hourly probability of equipment failure using sensor data and machine metadata, reducing unscheduled downtime.

## Methodology
- **Dataset**: Microsoft Azure Predictive Maintenance Dataset (100 machines, 1 year of hourly data).
- **Preprocessing**: Data merging (Telemetry, Errors, Maintenance, Failures, Machine Metadata) and categorical encoding.
- **Model**: Random Forest Classifier trained to predict the binary target (failure: 0/1).
- **Metrics**: High accuracy with class imbalance caveats; risk is communicated via probability instead of a hard 0/1.
- **Dashboard**: Streamlit app to visualize sensor data and failure risk over time.

## Setup and Running the Dashboard

### Prerequisites
1. Python 3.8+ installed.
2. All five CSV files must be placed in the `./data/` directory.
3. The saved model (`predictive_maintenance_model.pkl`) must be placed in the `./models/` directory.

### Installation
1. Clone the repository.
2. Install dependencies:
```bash
pip install -r app/requirements.txt
```

### Run the Dashboard
From the project root directory:
```bash
streamlit run app/dashboard.py
```
This will open the application in your web browser.


