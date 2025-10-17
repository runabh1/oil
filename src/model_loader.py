import joblib
import os


def load_rf_model(path: str = 'models/predictive_maintenance_model.pkl'):
    """Loads the pre-trained Random Forest Classification model."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at: {path}. Ensure you've placed the model there."
        )

    try:
        model = joblib.load(path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


if __name__ == '__main__':
    # Example manual test: uncomment to try local load
    # _ = load_rf_model()
    pass


