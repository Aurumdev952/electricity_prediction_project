import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


os.makedirs('model', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)


data = pd.read_csv('electricity_consumption_data.csv')


data['date'] = pd.to_datetime(data['date'])
data['hour'] = data['date'].dt.hour
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month


print("Dataset preview:")
print(data.head())


numeric_cols = ['temperature', 'humidity', 'num_people', 'num_children',
                'appliance_count', 'weekend', 'hour', 'day_of_week', 'month', 'energy_kWh']
corr = data[numeric_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png')
plt.close()


X = data.drop(['energy_kWh', 'date'], axis=1)
y = data['energy_kWh']


categorical_features = ['season', 'time_of_day']
numerical_features = ['temperature', 'humidity', 'num_people', 'num_children',
                      'appliance_count', 'weekend', 'hour', 'day_of_week', 'month']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])


param_grid = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__max_depth': [None, 10, 20, 30],
    'regressor__min_samples_split': [2, 5, 10],
}


quick_param_grid = {
    'regressor__n_estimators': [100],
    'regressor__max_depth': [20],
    'regressor__min_samples_split': [5],
}

print("Starting hyperparameter tuning...")
grid_search = GridSearchCV(model, param_grid=quick_param_grid, cv=3, n_jobs=-1, verbose=1, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Model Evaluation on Test Set:")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"Root Mean Squared Error: {rmse:.4f}")
print(f"R-squared: {r2:.4f}")


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Energy Consumption')
plt.tight_layout()
plt.savefig('visualizations/actual_vs_predicted.png')
plt.close()

feature_names = numerical_features + list(best_model.named_steps['preprocessor']
                                          .named_transformers_['cat']
                                          .get_feature_names_out(categorical_features))

importances = best_model.named_steps['regressor'].feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png')
plt.close()


fig, axs = plt.subplots(2, 2, figsize=(16, 12))


sns.scatterplot(x='temperature', y='energy_kWh', data=data, ax=axs[0, 0], alpha=0.3)
axs[0, 0].set_title('Temperature vs Energy Consumption')


sns.boxplot(x='appliance_count', y='energy_kWh', data=data, ax=axs[0, 1])
axs[0, 1].set_title('Appliance Count vs Energy Consumption')
axs[0, 1].set_xticks(axs[0, 1].get_xticks()[::2])  # Show fewer xticks


sns.boxplot(x='season', y='energy_kWh', data=data, ax=axs[1, 0])
axs[1, 0].set_title('Season vs Energy Consumption')


sns.boxplot(x='time_of_day', y='energy_kWh', data=data, ax=axs[1, 1])
axs[1, 1].set_title('Time of Day vs Energy Consumption')

plt.tight_layout()
plt.savefig('visualizations/feature_relationships.png')
plt.close()

joblib.dump(best_model, 'model/electricity_consumption_model.pkl')
print("Model saved to 'model/electricity_consumption_model.pkl'")


def predict_consumption(temperature, humidity, season, time_of_day,
                       num_people, num_children, appliance_count, weekend,
                       hour, day_of_week, month):
    """Function to generate predictions that can be used in the API"""
    data = pd.DataFrame({
        'temperature': [temperature],
        'humidity': [humidity],
        'season': [season],
        'time_of_day': [time_of_day],
        'num_people': [num_people],
        'num_children': [num_children],
        'appliance_count': [appliance_count],
        'weekend': [weekend],
        'hour': [hour],
        'day_of_week': [day_of_week],
        'month': [month]
    })

    prediction = best_model.predict(data)[0]
    return round(prediction, 2)

sample_data = data.sample(1000)
sample_data.to_csv('visualizations/sample_data.csv', index=False)

print("Training and evaluation complete!")
