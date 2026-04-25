let latestStatus = null;

const statusDot = document.querySelector(".status-dot");
const statusText = document.getElementById("status-text");
const debugLog = document.getElementById("debug-log");
const auditWarnings = document.getElementById("audit-warnings");
const recommendationSummary = document.getElementById("recommendation-summary");
const recommendationMode = document.getElementById("recommendation-mode");
const recommendationResultTitle = document.getElementById("recommendation-result-title");

const formatJSON = (obj) => JSON.stringify(obj, null, 2);

async function fetchStatus() {
  try {
    const response = await fetch("/api/status");
    if (!response.ok) {
      throw new Error("Backend Offline");
    }
    const data = await response.json();
    latestStatus = data;
    updateStatusUI(data);
  } catch (error) {
    statusDot.className = "status-dot offline";
    statusText.textContent = "System Offline";
    console.error(error);
  }
}

function updateStatusUI(data) {
  const isReady = Boolean(data.soil_audit?.runnable);
  statusDot.className = `status-dot ${isReady ? "online" : "offline"}`;
  statusText.textContent = isReady ? "System Ready" : "System Issues Detected";
  debugLog.textContent = formatJSON(data);

  auditWarnings.innerHTML = "";
  const warnings = [
    ...(data.soil_audit?.warnings || []),
    ...(data.recommendation_audit?.warnings || []),
  ];
  warnings.forEach((warning) => {
    const p = document.createElement("p");
    p.textContent = `• ${warning}`;
    p.style.color = "#e63946";
    p.style.fontSize = "0.85rem";
    p.style.margin = "0.2rem 0";
    auditWarnings.appendChild(p);
  });

  const recommendationAudit = data.recommendation_audit || {};
  if (recommendationAudit.active_mode === "crop_recommendation") {
    recommendationSummary.textContent =
      `Input soil and climate parameters to receive crop recommendations across ${recommendationAudit.crop_type_count || "multiple"} crop types.`;
    recommendationMode.textContent =
      `Active model: ${recommendationAudit.best_crop_model || "Crop recommendation model"}`;
    recommendationResultTitle.textContent = "Recommended Crop";
  } else {
    recommendationSummary.textContent =
      "Input soil and climate parameters to inspect the legacy fallback model while the crop recommendation model is still pending integration.";
    recommendationMode.textContent = "Active model: Legacy Soil ANN fallback";
    recommendationResultTitle.textContent = "Predicted Soil Type";
  }
}

function initTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  const panels = document.querySelectorAll(".tab-panel");

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const target = tab.dataset.tab;
      tabs.forEach((item) => item.classList.remove("active"));
      panels.forEach((panel) => panel.classList.remove("active"));
      tab.classList.add("active");
      document.getElementById(target).classList.add("active");
    });
  });
}

async function handleTabularSubmit(event) {
  event.preventDefault();
  const resultCard = document.getElementById("ann-result-card");
  const display = document.getElementById("ann-display");
  const emptyState = resultCard.querySelector(".result-empty");

  const payload = {
    soil_type: document.getElementById("soil_type").value,
    irrigation_available: document.getElementById("irrigation_available").value,
    farm_size_acres: document.getElementById("farm_size_acres").value,
    soil_ph: document.getElementById("soil_ph").value,
    soil_nitrogen: document.getElementById("soil_nitrogen").value,
    soil_organic_matter: document.getElementById("soil_organic_matter").value,
    temperature: document.getElementById("temperature").value,
    rainfall: document.getElementById("rainfall").value,
    humidity: document.getElementById("humidity").value,
  };

  emptyState.classList.add("hidden");
  display.classList.remove("hidden");
  document.getElementById("ann-soil-type").textContent = "Analyzing...";
  document.getElementById("ann-details").textContent = "";
  document.getElementById("ann-confidence-bar").style.width = "0%";

  try {
    const response = await fetch("/api/soil/predict-tabular", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.error);
    }
    renderANNResult(result);
  } catch (err) {
    document.getElementById("ann-soil-type").textContent = "Error";
    document.getElementById("ann-details").textContent = err.message;
  }
}

function renderANNResult(result) {
  const label = result.label.replace(/_/g, " ");
  const probability = result.top_probability * 100;
  document.getElementById("ann-soil-type").textContent = label;
  document.getElementById("ann-confidence-bar").style.width = `${probability}%`;

  const details = document.getElementById("ann-details");
  const note = result.warning ? `<strong>Note:</strong> ${result.warning}<br><br>` : "";
  const title = result.model_name || "Probabilities";
  details.innerHTML =
    `${note}<strong>${result.mode === "crop_recommendation" ? "Confidence Rankings" : title}:</strong><br>` +
    result.ranked_predictions
      .map((item) => `${item.label.replace(/_/g, " ")}: ${Math.round(item.probability * 100)}%`)
      .join("<br>");
}

function initImageUpload() {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("soil-file");
  const preview = document.getElementById("soil-preview");
  const previewContainer = document.getElementById("image-preview-container");

  dropZone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      previewContainer.classList.remove("hidden");
      dropZone.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  });
}

async function handleImageSubmit(event) {
  event.preventDefault();
  const fileInput = document.getElementById("soil-file");
  const display = document.getElementById("cnn-display");
  const emptyState = document.querySelector("#cnn-result-card .result-empty");
  const file = fileInput.files[0];
  if (!file) {
    return;
  }

  emptyState.classList.add("hidden");
  display.classList.remove("hidden");
  document.getElementById("cnn-soil-type").textContent = "Scanning...";
  document.getElementById("cnn-ranked-list").textContent = "";
  document.getElementById("cnn-confidence-bar").style.width = "0%";

  const reader = new FileReader();
  reader.readAsDataURL(file);
  reader.onload = async () => {
    const base64 = reader.result.split(",")[1];
    try {
      const response = await fetch("/api/soil/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: base64 }),
      });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.error);
      }
      renderCNNResult(result);
    } catch (err) {
      document.getElementById("cnn-soil-type").textContent = "Error";
      document.getElementById("cnn-ranked-list").textContent = err.message;
    }
  };
}

function renderCNNResult(result) {
  const label = result.label.replace(/_/g, " ");
  const probability = result.top_probability * 100;
  document.getElementById("cnn-soil-type").textContent = label;
  document.getElementById("cnn-confidence-bar").style.width = `${probability}%`;

  const list = document.getElementById("cnn-ranked-list");
  list.innerHTML =
    "<strong>Confidence Rankings:</strong><br>" +
    result.ranked_predictions
      .map((item) => `• ${item.label.replace(/_/g, " ")}: ${Math.round(item.probability * 100)}%`)
      .join("<br>");
}

function initModals() {
  const toggle = document.getElementById("debug-toggle");
  const modal = document.getElementById("debug-modal");
  const close = modal.querySelector(".close-btn");

  toggle.addEventListener("click", () => modal.classList.remove("hidden"));
  close.addEventListener("click", () => modal.classList.add("hidden"));
  window.addEventListener("click", (event) => {
    if (event.target === modal) {
      modal.classList.add("hidden");
    }
  });
}

function init() {
  initTabs();
  initImageUpload();
  initModals();
  fetchStatus();
  document.getElementById("tabular-form").addEventListener("submit", handleTabularSubmit);
  document.getElementById("soil-form").addEventListener("submit", handleImageSubmit);
}

document.addEventListener("DOMContentLoaded", init);
