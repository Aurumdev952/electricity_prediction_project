import os
import json
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import PredictionInputSerializer, PredictionOutputSerializer
from .models import Prediction
from .ml_model import ElectricityModel

def index(request):
    """Main page view"""
    return render(request, 'predictor/index.html')

def visualization(request):
    """Visualization page view"""
    return render(request, 'predictor/visualization.html')

def get_sample_data(request):
    """Endpoint to return sample data for visualizations"""
    sample_data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                   'visualizations', 'sample_data.csv')

    if os.path.exists(sample_data_path):
        data = pd.read_csv(sample_data_path)
        return JsonResponse(data.to_dict(orient='records'), safe=False)
    else:
        return JsonResponse({'error': 'Sample data not found'}, status=404)

class PredictElectricity(APIView):
    """API view for making predictions"""

    def post(self, request):
        serializer = PredictionInputSerializer(data=request.data)

        if serializer.is_valid():
            features = serializer.validated_data

            # Get prediction from model
            predicted_kwh = ElectricityModel.predict(features)

            # Save prediction to database
            features['predicted_kwh'] = predicted_kwh
            prediction = Prediction.objects.create(**features)

            # Return the prediction
            output_serializer = PredictionOutputSerializer(prediction)
            return Response(output_serializer.data, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

def feature_importance(request):
    """Endpoint to return feature importance data"""
    model = ElectricityModel.get_model()

    # Extract feature names and importance
    feature_names = model.named_steps['regressor'].feature_names_in_
    importances = model.named_steps['regressor'].feature_importances_

    # Create dictionary of feature importances
    feature_importance_data = [
        {'feature': str(feature), 'importance': float(importance)}
        for feature, importance in zip(feature_names, importances)
    ]

    return JsonResponse(feature_importance_data, safe=False)
