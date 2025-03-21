from django.db import models

class Prediction(models.Model):
    temperature = models.FloatField()
    humidity = models.FloatField()
    season = models.CharField(max_length=10)
    time_of_day = models.CharField(max_length=10)
    num_people = models.IntegerField()
    num_children = models.IntegerField()
    appliance_count = models.IntegerField()
    weekend = models.BooleanField()
    hour = models.IntegerField()
    day_of_week = models.IntegerField()
    month = models.IntegerField()
    predicted_kwh = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction {self.id}: {self.predicted_kwh} kWh on {self.timestamp}"
