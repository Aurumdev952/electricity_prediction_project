from rest_framework import serializers
from .models import Prediction

class PredictionInputSerializer(serializers.Serializer):
    temperature = serializers.FloatField()
    humidity = serializers.FloatField()
    season = serializers.ChoiceField(choices=['Winter', 'Spring', 'Summer', 'Fall'])
    time_of_day = serializers.ChoiceField(choices=['Morning', 'Afternoon', 'Evening', 'Night'])
    num_people = serializers.IntegerField(min_value=1, max_value=10)
    num_children = serializers.IntegerField(min_value=0, max_value=8)
    appliance_count = serializers.IntegerField(min_value=1, max_value=30)
    weekend = serializers.BooleanField()
    hour = serializers.IntegerField(min_value=0, max_value=23)
    day_of_week = serializers.IntegerField(min_value=0, max_value=6)
    month = serializers.IntegerField(min_value=1, max_value=12)

class PredictionOutputSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'
