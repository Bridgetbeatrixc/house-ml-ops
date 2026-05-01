const form = document.querySelector("#predictionForm");
const resetButton = document.querySelector("#resetButton");
const submitButton = document.querySelector("#submitButton");
const predictionValue = document.querySelector("#predictionValue");
const predictionMeta = document.querySelector("#predictionMeta");
const modelStatus = document.querySelector("#modelStatus");
const modelVersion = document.querySelector("#modelVersion");

const numberFields = new Set([
  "OverallQual",
  "GrLivArea",
  "GarageCars",
  "GarageArea",
  "TotalBsmtSF",
  "FullBath",
  "YearBuilt",
  "YearRemodAdd",
  "LotArea",
]);

function buildPayload() {
  const data = new FormData(form);
  const payload = {};

  for (const [key, value] of data.entries()) {
    if (value === "") {
      payload[key] = null;
    } else if (numberFields.has(key)) {
      payload[key] = Number(value);
    } else {
      payload[key] = value;
    }
  }

  return payload;
}

function formatCurrency(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const health = await response.json();
    modelStatus.textContent = health.model_loaded ? "Model ready" : "Model missing";
    modelStatus.classList.toggle("ok", health.model_loaded);
    modelStatus.classList.toggle("error", !health.model_loaded);
  } catch (error) {
    modelStatus.textContent = "API offline";
    modelStatus.classList.add("error");
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "Predicting...";
  predictionMeta.textContent = "Sending request to FastAPI.";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });

    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Prediction failed");
    }

    predictionValue.textContent = formatCurrency(result.prediction);
    modelVersion.textContent = result.model_version;
    predictionMeta.textContent = "Prediction completed successfully.";
  } catch (error) {
    predictionMeta.textContent = error.message;
    predictionValue.textContent = "$0";
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict Price";
  }
});

resetButton.addEventListener("click", () => {
  form.reset();
  predictionValue.textContent = "$0";
  predictionMeta.textContent = "Submit the form to get a prediction.";
});

checkHealth();
