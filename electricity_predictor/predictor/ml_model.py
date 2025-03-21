import os
import joblib
import pandas as pd
from django.conf import settings

class ElectricityModel:
    model = None

    @classmethod
    def get_model(cls):
        """Load the model if it's not already loaded"""
        if cls.model is None:
            model_path = os.path.join(settings.BASE_DIR, 'model', 'electricity_consumption_model.pkl')
            cls.model = joblib.load(model_path)
        return cls.model

    @classmethod
    def predict(cls, features):
        """Make a prediction using the loaded model"""
        model = cls.get_model()

        # Create DataFrame with the input features
        df = pd.DataFrame([features])

        # Make prediction
        prediction = model.predict(df)[0]
        return round(prediction, 2)
