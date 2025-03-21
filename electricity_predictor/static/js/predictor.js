// static/js/predictor.js
document.addEventListener("DOMContentLoaded", function () {
  // Initialize form with default values
  document.getElementById("temperature").value = 20;
  document.getElementById("humidity").value = 65;
  document.getElementById("season").value = "Summer";
  document.getElementById("time_of_day").value = "Evening";
  document.getElementById("num_people").value = 3;
  document.getElementById("num_children").value = 1;
  document.getElementById("appliance_count").value = 10;
  document.getElementById("weekend").value = "false";
  document.getElementById("hour").value = 18;
  document.getElementById("day_of_week").value = 2;
  document.getElementById("month").value = 7;

  // Set up form submission
  const form = document.getElementById("prediction-form");
  form.addEventListener("submit", function (event) {
    event.preventDefault();
    submitPredictionForm();
  });
});

async function submitPredictionForm() {
  const form = document.getElementById("prediction-form");
  const resultDiv = document.getElementById("prediction-result");

  // Show loading state
  resultDiv.innerHTML = '<p class="text-center">Processing prediction...</p>';

  // Prepare form data
  const formData = {
    temperature: parseFloat(document.getElementById("temperature").value),
    humidity: parseFloat(document.getElementById("humidity").value),
    season: document.getElementById("season").value,
    time_of_day: document.getElementById("time_of_day").value,
    num_people: parseInt(document.getElementById("num_people").value),
    num_children: parseInt(document.getElementById("num_children").value),
    appliance_count: parseInt(document.getElementById("appliance_count").value),
    weekend: document.getElementById("weekend").value === "true",
    hour: parseInt(document.getElementById("hour").value),
    day_of_week: parseInt(document.getElementById("day_of_week").value),
    month: parseInt(document.getElementById("month").value),
  };

  try {
    // Send form data to API
    const response = await fetch("/api/predict/", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": getCookie("csrftoken"),
      },
      body: JSON.stringify(formData),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    // Parse the response
    const data = await response.json();

    // Determine consumption category
    let consumptionCategory = "moderate";
    if (data.predicted_kwh > 3.5) {
      consumptionCategory = "high";
    } else if (data.predicted_kwh < 1.5) {
      consumptionCategory = "low";
    }

    // Display the prediction result
    resultDiv.innerHTML = `
            <div class="result-box predicted-${consumptionCategory}">
                <h4>Predicted Consumption</h4>
                <div class="result-value">${data.predicted_kwh} kWh</div>
                <p>Based on your household's characteristics</p>
            </div>
            <div class="factors-list">
                <h5>Key Factors:</h5>
                <ul>
                    <li>Season (${data.season})</li>
                    <li>Time of Day (${data.time_of_day})</li>
                    <li>Temperature (${data.temperature}°C)</li>
                    <li>Household Size (${data.num_people} people)</li>
                </ul>
            </div>
        `;

    // Update the chart
    updatePredictionChart(data.predicted_kwh);
  } catch (error) {
    console.error("Error:", error);
    resultDiv.innerHTML = `
            <div class="alert alert-danger">
                Error making prediction. Please try again.
            </div>
        `;
  }
}

function updatePredictionChart(predictedValue) {
  const ctx = document.getElementById("prediction-chart");

  // Destroy any existing chart
  if (window.predictionChart) {
    window.predictionChart.destroy();
  }

  // Create reference data for context
  const referenceData = {
    low: 1.2,
    average: 2.5,
    high: 4.0,
  };

  // Create the chart
  window.predictionChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Low Usage", "Average Usage", "High Usage", "Your Prediction"],
      datasets: [
        {
          label: "Electricity Consumption (kWh)",
          data: [
            referenceData.low,
            referenceData.average,
            referenceData.high,
            predictedValue,
          ],
          backgroundColor: [
            "rgba(75, 192, 192, 0.5)",
            "rgba(54, 162, 235, 0.5)",
            "rgba(255, 99, 132, 0.5)",
            "rgba(255, 206, 86, 0.7)",
          ],
          borderColor: [
            "rgba(75, 192, 192, 1)",
            "rgba(54, 162, 235, 1)",
            "rgba(255, 99, 132, 1)",
            "rgba(255, 206, 86, 1)",
          ],
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Energy Consumption (kWh)",
          },
        },
      },
    },
  });
}

// Helper function to get CSRF token for Django
function getCookie(name) {
  let cookieValue = null;
  if (document.cookie && document.cookie !== "") {
    const cookies = document.cookie.split(";");
    for (let i = 0; i < cookies.length; i++) {
      const cookie = cookies[i].trim();
      if (cookie.substring(0, name.length + 1) === name + "=") {
        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
        break;
      }
    }
  }
  return cookieValue;
}
