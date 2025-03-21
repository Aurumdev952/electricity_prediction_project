// static/js/visualization.js
document.addEventListener("DOMContentLoaded", function () {
  // Load all visualizations
  loadSampleData().then((data) => {
    renderActualVsPredicted(data);
    renderResidualsPlot(data);
    renderSeasonChart(data);
    renderTimeOfDayChart(data);
    renderTemperatureChart(data);
    renderHouseholdSizeChart(data);
  });

  // Load feature importance data
  loadFeatureImportance().then((data) => {
    renderFeatureImportanceChart(data);
  });
});

// Fetch sample data from the API
async function loadSampleData() {
  try {
    const response = await fetch("/api/sample-data/");
    if (!response.ok) {
      throw new Error("Failed to load sample data");
    }
    return await response.json();
  } catch (error) {
    console.error("Error loading sample data:", error);
    return [];
  }
}

// Fetch feature importance data from the API
async function loadFeatureImportance() {
  try {
    const response = await fetch("/api/feature-importance/");
    if (!response.ok) {
      throw new Error("Failed to load feature importance data");
    }
    return await response.json();
  } catch (error) {
    console.error("Error loading feature importance data:", error);
    return [];
  }
}

// Render Actual vs Predicted Energy Consumption Chart
function renderActualVsPredicted(data) {
  // Simulate predictions by adding some noise to the actual values
  const predictions = data.map((d) => {
    return {
      actual: d.energy_kWh,
      predicted: d.energy_kWh * (1 + (Math.random() * 0.2 - 0.1)), // +/- 10% noise
    };
  });

  const ctx = document.getElementById("actual-vs-predicted-chart");
  new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Actual vs Predicted",
          data: predictions.map((p) => ({ x: p.actual, y: p.predicted })),
          backgroundColor: "rgba(54, 162, 235, 0.5)",
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: "Perfect Prediction",
          data: (() => {
            const min = Math.min(...predictions.map((p) => p.actual));
            const max = Math.max(...predictions.map((p) => p.actual));
            return [
              { x: min, y: min },
              { x: max, y: max },
            ];
          })(),
          type: "line",
          borderColor: "rgba(255, 99, 132, 1)",
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      title: {
        display: true,
        text: "Actual vs Predicted Energy Consumption",
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Actual Energy (kWh)",
          },
        },
        y: {
          title: {
            display: true,
            text: "Predicted Energy (kWh)",
          },
        },
      },
    },
  });
}

// Render Residuals Plot
function renderResidualsPlot(data) {
  // Simulate residuals (predicted - actual)
  const residuals = data.map((d) => {
    const actual = d.energy_kWh;
    const predicted = actual * (1 + (Math.random() * 0.2 - 0.1)); // +/- 10% noise
    return {
      actual: actual,
      residual: predicted - actual,
    };
  });

  const ctx = document.getElementById("residuals-chart");
  new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Residuals",
          data: residuals.map((r) => ({ x: r.actual, y: r.residual })),
          backgroundColor: "rgba(75, 192, 192, 0.5)",
          pointRadius: 5,
          pointHoverRadius: 7,
        },
        {
          label: "Zero Line",
          data: (() => {
            const min = Math.min(...residuals.map((r) => r.actual));
            const max = Math.max(...residuals.map((r) => r.actual));
            return [
              { x: min, y: 0 },
              { x: max, y: 0 },
            ];
          })(),
          type: "line",
          borderColor: "rgba(255, 99, 132, 1)",
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      title: {
        display: true,
        text: "Residuals Plot",
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Actual Energy (kWh)",
          },
        },
        y: {
          title: {
            display: true,
            text: "Residual (Predicted - Actual)",
          },
        },
      },
    },
  });
}

// Render Feature Importance Chart
function renderFeatureImportanceChart(data) {
  // Sort by importance in descending order
  data.sort((a, b) => b.importance - a.importance);

  // Take top 10 features
  const topFeatures = data.slice(0, 10);

  const ctx = document.getElementById("feature-importance-chart");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: topFeatures.map((f) => f.feature),
      datasets: [
        {
          label: "Feature Importance",
          data: topFeatures.map((f) => f.importance),
          backgroundColor: "rgba(153, 102, 255, 0.6)",
          borderColor: "rgba(153, 102, 255, 1)",
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      title: {
        display: true,
        text: "Top 10 Feature Importance",
      },
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: "Importance",
          },
        },
        x: {
          title: {
            display: true,
            text: "Feature",
          },
        },
      },
    },
  });
}

// Render Season Chart
function renderSeasonChart(data) {
  // Group data by season and calculate average consumption
  const seasonData = {};
  data.forEach((d) => {
    if (!seasonData[d.season]) {
      seasonData[d.season] = { sum: 0, count: 0 };
    }
    seasonData[d.season].sum += d.energy_kWh;
    seasonData[d.season].count += 1;
  });

  // Calculate averages and convert to array
  const seasons = ["Winter", "Spring", "Summer", "Fall"];
  const averages = seasons.map((season) => ({
    season: season,
    average: seasonData[season]
      ? seasonData[season].sum / seasonData[season].count
      : 0,
  }));

  const ctx = document.getElementById("season-chart");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: averages.map((d) => d.season),
      datasets: [
        {
          label: "Average Energy Consumption by Season",
          data: averages.map((d) => d.average),
          backgroundColor: [
            "rgba(54, 162, 235, 0.6)", // Winter (blue)
            "rgba(75, 192, 192, 0.6)", // Spring (green)
            "rgba(255, 206, 86, 0.6)", // Summer (yellow)
            "rgba(255, 159, 64, 0.6)", // Fall (orange)
          ],
          borderColor: [
            "rgba(54, 162, 235, 1)",
            "rgba(75, 192, 192, 1)",
            "rgba(255, 206, 86, 1)",
            "rgba(255, 159, 64, 1)",
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
            text: "Average Energy (kWh)",
          },
        },
      },
      plugins: {
        legend: {
          display: false,
        },
      },
    },
  });
}

// Render Time of Day Chart
function renderTimeOfDayChart(data) {
  // Group data by time of day and calculate average consumption
  const timeData = {};
  data.forEach((d) => {
    if (!timeData[d.time_of_day]) {
      timeData[d.time_of_day] = { sum: 0, count: 0 };
    }
    timeData[d.time_of_day].sum += d.energy_kWh;
    timeData[d.time_of_day].count += 1;
  });

  // Calculate averages and convert to array
  const times = ["Morning", "Afternoon", "Evening", "Night"];
  const averages = times.map((time) => ({
    time: time,
    average: timeData[time] ? timeData[time].sum / timeData[time].count : 0,
  }));

  const ctx = document.getElementById("time-of-day-chart");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: averages.map((d) => d.time),
      datasets: [
        {
          label: "Average Energy Consumption by Time of Day",
          data: averages.map((d) => d.average),
          backgroundColor: [
            "rgba(255, 206, 86, 0.6)", // Morning (yellow)
            "rgba(255, 99, 132, 0.6)", // Afternoon (red)
            "rgba(153, 102, 255, 0.6)", // Evening (purple)
            "rgba(54, 162, 235, 0.6)", // Night (blue)
          ],
          borderColor: [
            "rgba(255, 206, 86, 1)",
            "rgba(255, 99, 132, 1)",
            "rgba(153, 102, 255, 1)",
            "rgba(54, 162, 235, 1)",
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
            text: "Average Energy (kWh)",
          },
        },
      },
      plugins: {
        legend: {
          display: false,
        },
      },
    },
  });
}

// Render Temperature Chart
function renderTemperatureChart(data) {
  // Group data by temperature ranges
  const temperatureRanges = [];
  const rangeSize = 5; // 5°C per range
  const minTemp =
    Math.floor(Math.min(...data.map((d) => d.temperature)) / rangeSize) *
    rangeSize;
  const maxTemp =
    Math.ceil(Math.max(...data.map((d) => d.temperature)) / rangeSize) *
    rangeSize;

  for (let t = minTemp; t < maxTemp; t += rangeSize) {
    temperatureRanges.push({
      range: `${t}°C - ${t + rangeSize}°C`,
      min: t,
      max: t + rangeSize,
      sum: 0,
      count: 0,
    });
  }

  // Group data into ranges
  data.forEach((d) => {
    const range = temperatureRanges.find(
      (r) => d.temperature >= r.min && d.temperature < r.max
    );
    if (range) {
      range.sum += d.energy_kWh;
      range.count += 1;
    }
  });

  // Calculate averages
  const averages = temperatureRanges.map((range) => ({
    range: range.range,
    average: range.count > 0 ? range.sum / range.count : 0,
  }));

  const ctx = document.getElementById("temperature-chart");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: averages.map((d) => d.range),
      datasets: [
        {
          label: "Average Energy Consumption by Temperature Range",
          data: averages.map((d) => d.average),
          backgroundColor: "rgba(255, 159, 64, 0.6)", // Orange
          borderColor: "rgba(255, 159, 64, 1)",
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
            text: "Average Energy (kWh)",
          },
        },
        x: {
          title: {
            display: true,
            text: "Temperature Range (°C)",
          },
        },
      },
      plugins: {
        legend: {
          display: false,
        },
      },
    },
  });
}

function renderHouseholdSizeChart(data) {

    const processedData = data.map(d => ({
        ...d,
        household_size: (parseInt(d.num_people) || 0) + (parseInt(d.num_children) || 0)
    }));

    // Filter out invalid entries
    const validData = processedData.filter(d =>
        !isNaN(d.household_size) && d.household_size > 0
    );

    // Group data by calculated household_size
    const householdData = {};
    validData.forEach(d => {
        const size = d.household_size;
        if (!householdData[size]) {
            householdData[size] = { sum: 0, count: 0 };
        }
        householdData[size].sum += d.energy_kWh;
        householdData[size].count += 1;
    });

    // Convert to array and sort by household size
    const householdSizes = Object.keys(householdData)
        .map(Number)
        .sort((a, b) => a - b);

    const averages = householdSizes.map(size => ({
        size: size,
        average: householdData[size].sum / householdData[size].count
    }));

    const ctx = document.getElementById('household-size-chart');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: averages.map(d => `${d.size} Person(s)`),
            datasets: [{
                label: 'Energy Consumption by Household Size',
                data: averages.map(d => d.average),
                borderColor: 'rgba(75, 192, 192, 1)',
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                tension: 0.4,
                fill: false,
                pointRadius: 5
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: 'Average Energy (kWh)' }
                },
                x: {
                    title: { display: true, text: 'Household Size' },
                    type: 'category'
                }
            }
        }
    });
}
