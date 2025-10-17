import pandas as pd
import os


# --- Helper Function for Data Loading ---
def load_all_data(data_path: str = 'data/'):
    """Loads all five required CSV files from the data directory."""
    files = {
        'telemetry': 'PdM_telemetry.csv',
        'machines': 'PdM_machines.csv',  # match actual filename in workspace
        'failures': 'PdM_failures.csv',
        'maint': 'PdM_maint.csv',
        'errors': 'PdM_errors.csv',
    }

    data_dict = {}
    for name, filename in files.items():
        file_path = os.path.join(data_path, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing file: {file_path}. Ensure all 5 CSVs are in the 'data/' folder."
            )

        # Read CSV (let pandas infer types first)
        df = pd.read_csv(file_path)

        # Ensure datetime columns are parsed correctly
        if name in ['telemetry', 'failures', 'maint', 'errors']:
            # The datetime column name may be 'datetime' or 'date'; standardize to 'datetime'
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            elif 'date' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.drop(columns=['date'], errors='ignore')

            if name == 'telemetry':
                # Telemetry is hourly time series; set datetime as index for clean merges later
                df = df.set_index('datetime')

        data_dict[name] = df

    print("All raw data loaded successfully.")
    return data_dict


# --- Main Preprocessing Function ---
def preprocess_data(data_dict: dict) -> pd.DataFrame:
    """
    Replicates the data merging and feature creation steps from the training notebook.
    """
    telemetry = data_dict['telemetry'].copy().reset_index()
    machines = data_dict['machines']
    maint = data_dict['maint']
    failures = data_dict['failures']
    errors = data_dict['errors']

    # 1. Merge telemetry with machine details
    df = pd.merge(telemetry, machines, on='machineID', how='left')

    # 2. Merge maintenance, failures, and errors on ['machineID', 'datetime']
    df = pd.merge(
        df,
        maint.rename(columns={'comp': 'maint_comp'}),
        on=['machineID', 'datetime'],
        how='left',
    )

    # Failures CSV has column named 'failure' with values like 'comp1' .. 'comp4'
    # Normalize to 'failure_comp' for downstream logic
    failures_norm = failures.rename(columns={'failure': 'failure_comp'})
    df = pd.merge(
        df,
        failures_norm,
        on=['machineID', 'datetime'],
        how='left',
    )

    df = pd.merge(
        df,
        errors.rename(columns={'errorID': 'error'}),
        on=['machineID', 'datetime'],
        how='left',
    )

    # 3. Create target column BEFORE generic NaN fill: 1 if any failure component present
    if 'failure_comp' in df.columns:
        df['failure'] = df['failure_comp'].notna().astype(int)
    else:
        df['failure'] = 0

    # Now fill remaining NaNs with 0 for simpler UI/modeling
    df = df.fillna(0)

    # 5. Encode 'model' as categorical codes
    df['model'] = df['model'].astype('category').cat.codes

    # 6. Drop redundant columns created during merges
    df = df.drop(columns=['maint_comp', 'failure_comp'], errors='ignore')

    print(f"Preprocessed dataset shape: {df.shape}")
    return df


