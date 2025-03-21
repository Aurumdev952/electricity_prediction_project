# Electricity Consumption Predictor

## Overview
This project predicts electricity consumption based on various factors like temperature, household size, and time of day. It includes a machine learning model for predictions and interactive visualizations to analyze trends.

## Features
- **Prediction**: Predict energy consumption using a trained Random Forest model.
- **Visualizations**: Interactive charts for analyzing trends in energy usage.
- **API**: REST API for making predictions and fetching data.

## Setup

### Prerequisites
- Python 3.8+
- PostgreSQL (or any database supported by Django)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Aurumdev952/electricity_prediction_project.git
   cd electricity_prediction_project
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Generate data:
   ```bash
   python generate.py
   ```

5. Train the model:
   ```bash
   python train.py
   ```

6. Set up the database:
   ```bash
   cd electricity_predictor
   python manage.py migrate
   ```
   

7. Run the server:
   ```bash
   python manage.py runserver
   ```

8. Access the application at `http://127.0.0.1:8000/`.

## Usage

### Predict Energy Consumption
- Visit the homepage and input your data to get a prediction.

### Visualizations
- Navigate to the **Visualizations** page to explore interactive charts:
  - Actual vs Predicted
  - Feature Importance
  - Consumption by Season, Time of Day, Temperature, and Household Size

### API Endpoints
- **Predict**: `POST /api/predict/`
- **Sample Data**: `GET /api/sample-data/`
- **Feature Importance**: `GET /api/feature-importance/`

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

