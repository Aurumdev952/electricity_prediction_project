import numpy as np
import pandas as pd
import datetime
import random
from sklearn.preprocessing import OneHotEncoder

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_electricity_data(num_samples=5000):
    """
    Generate synthetic electricity consumption data with realistic patterns
    """
    # Generate dates for one year
    start_date = datetime.datetime(2024, 1, 1)
    dates = [start_date + datetime.timedelta(hours=i*4) for i in range(num_samples)]

    # Extract month, day of week, and hour for seasonality features
    months = [d.month for d in dates]
    day_of_week = [d.weekday() for d in dates]  # 0-6 (Monday-Sunday)
    hours = [d.hour for d in dates]

    # Map month to season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'

    seasons = [get_season(m) for m in months]

    # Determine time of day
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 22:
            return 'Evening'
        else:
            return 'Night'

    time_of_day = [get_time_of_day(h) for h in hours]

    # Generate weekend flag
    weekend = [1 if d >= 5 else 0 for d in day_of_week]  # Saturday=5, Sunday=6

    # Generate household features
    # We'll create 200 different households with varying characteristics
    num_households = 200
    household_features = []

    for _ in range(num_households):
        # Number of adults (1-5)
        adults = random.randint(1, 5)

        # Number of children (0-4)
        children = random.randint(0, 4)

        # Number of appliances (5-20, correlated with number of people)
        base_appliances = 5
        appliance_per_person = random.uniform(1.5, 3)
        appliances = int(base_appliances + appliance_per_person * (adults + children))
        appliances = min(20, max(5, appliances))  # Clamp between 5 and 20

        household_features.append({
            'num_people': adults + children,
            'num_children': children,
            'appliance_count': appliances
        })

    # Randomly assign data points to households
    household_indices = np.random.choice(num_households, num_samples)

    num_people = [household_features[i]['num_people'] for i in household_indices]
    num_children = [household_features[i]['num_children'] for i in household_indices]
    appliance_count = [household_features[i]['appliance_count'] for i in household_indices]

    # Generate weather data
    # Temperature varies by season with some random fluctuation
    base_temp = {
        'Winter': 5,
        'Spring': 15,
        'Summer': 25,
        'Fall': 15
    }

    temperature = []
    for s in seasons:
        # Add daily fluctuation (cooler at night, warmer in daytime)
        temp = base_temp[s] + np.random.normal(0, 3)
        temperature.append(temp)

    # Adjust temperature based on time of day
    for i, tod in enumerate(time_of_day):
        if tod == 'Night':
            temperature[i] -= random.uniform(3, 6)
        elif tod == 'Afternoon':
            temperature[i] += random.uniform(3, 6)

    # Humidity (related to temperature and season)
    humidity = []
    for i, s in enumerate(seasons):
        if s == 'Summer':
            base_humidity = 70
        elif s == 'Winter':
            base_humidity = 40
        else:
            base_humidity = 55

        # Inverse relationship with temperature within a range
        temp_factor = -0.3 * (temperature[i] - base_temp[s])
        hum = base_humidity + temp_factor + np.random.normal(0, 5)
        humidity.append(max(30, min(90, hum)))  # Clamp between 30 and 90

    # Now generate the target variable: energy consumption in kWh
    energy_kWh = []

    for i in range(num_samples):
        # Base consumption related to household size and appliances
        base_consumption = 0.1 * num_people[i] + 0.1 * appliance_count[i]

        # Seasonal factor
        season_factor = {
            'Winter': 1.5,  # Highest in winter
            'Summer': 1.3,  # High in summer (A/C)
            'Spring': 0.9,  # Lower in spring
            'Fall': 1.0     # Moderate in fall
        }[seasons[i]]

        # Time of day factor
        time_factor = {
            'Morning': 1.2,  # Higher in morning
            'Afternoon': 0.9, # Lower in afternoon when people may be out
            'Evening': 1.5,  # Highest in evening
            'Night': 0.6     # Lowest at night
        }[time_of_day[i]]

        # Weekend factor
        weekend_factor = 1.2 if weekend[i] == 1 else 1.0  # Higher on weekends

        # Temperature factor - U-shaped curve
        # More energy when it's very cold (heating) or very hot (cooling)
        temp = temperature[i]
        temp_factor = 1.0 + 0.02 * (abs(temp - 18) ** 1.5)  # 18°C is most energy-efficient

        # Children factor - households with children use more energy during certain times
        children_factor = 1.0
        if num_children[i] > 0:
            if time_of_day[i] in ['Morning', 'Evening']:
                children_factor = 1.0 + 0.1 * num_children[i]

        # Combine all factors
        consumption = (base_consumption * season_factor * time_factor *
                      weekend_factor * temp_factor * children_factor)

        # Add some random noise
        noise = np.random.normal(0, consumption * 0.05)  # 5% noise

        energy_kWh.append(max(0.1, consumption + noise))

    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'temperature': np.round(temperature, 1),
        'humidity': np.round(humidity, 1),
        'season': seasons,
        'time_of_day': time_of_day,
        'num_people': num_people,
        'num_children': num_children,
        'appliance_count': appliance_count,
        'weekend': weekend,
        'energy_kWh': np.round(energy_kWh, 2)
    })

    return data

# Generate the data
data = generate_electricity_data(5000)

# Display basic statistics
print("Generated dataset with", len(data), "rows")
print("\nBasic statistics:")
print(data.describe())

# Show the relationships between features and target
print("\nAverage energy consumption by season:")
print(data.groupby('season')['energy_kWh'].mean().sort_values(ascending=False))

print("\nAverage energy consumption by time of day:")
print(data.groupby('time_of_day')['energy_kWh'].mean().sort_values(ascending=False))

print("\nAverage energy consumption on weekends vs weekdays:")
print(data.groupby('weekend')['energy_kWh'].mean())

# Save the data to a CSV file
data.to_csv('electricity_consumption_data.csv', index=False)
print("\nData saved to 'electricity_consumption_data.csv'")
