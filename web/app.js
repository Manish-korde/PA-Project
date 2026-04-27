let latestStatus = null;

const statusDot = document.querySelector(".status-dot");
const statusText = document.getElementById("status-text");
const debugLog = document.getElementById("debug-log");
const auditWarnings = document.getElementById("audit-warnings");
const recommendationSummary = document.getElementById("recommendation-summary");
const recommendationMode = document.getElementById("recommendation-mode");

const formatJSON = (obj) => JSON.stringify(obj, null, 2);

let soilAnalysisLoaded = false;

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
  } else {
    recommendationSummary.textContent =
      "Input soil and climate parameters to inspect the legacy fallback model while the crop recommendation model is still pending integration.";
    recommendationMode.textContent = "Active model: Legacy Soil ANN fallback";
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

      if (target === "analysis" && !soilAnalysisLoaded) {
        loadSoilAnalysis();
      }
    });
  });
}

async function loadSoilAnalysis() {
  const status = document.getElementById("analysis-status");
  const grid = document.getElementById("analysis-plot-grid");
  if (!status || !grid) return;

  status.textContent = "Loading graphs…";
  grid.innerHTML = "";

  try {
    const response = await fetch("/api/soil/analysis/list");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to load plot list");

    const plots = data.plots || [];
    if (!plots.length) {
      status.textContent = "No plots available.";
      soilAnalysisLoaded = true;
      return;
    }

    status.textContent = `Loaded ${plots.length} graphs.`;
    plots.forEach((plot) => {
      const wrapper = document.createElement("div");
      wrapper.className = "analysis-plot";

      const title = document.createElement("h4");
      title.textContent = plot.title || plot.name;

      const img = document.createElement("img");
      img.alt = plot.title || plot.name;
      img.loading = "lazy";
      img.src = `/api/soil/analysis/plot?name=${encodeURIComponent(plot.name)}`;

      wrapper.appendChild(title);
      wrapper.appendChild(img);
      grid.appendChild(wrapper);
    });

    soilAnalysisLoaded = true;
  } catch (err) {
    status.textContent = `Error loading graphs: ${err.message || err}`;
  }
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
    soil_phosphorus: document.getElementById("soil_phosphorus").value,
    soil_potassium: document.getElementById("soil_potassium").value,
    soil_organic_matter: document.getElementById("soil_organic_matter").value,
    temperature: document.getElementById("temperature").value,
    rainfall: document.getElementById("rainfall").value,
    humidity: document.getElementById("humidity").value,
  };

  emptyState.classList.add("hidden");
  display.classList.remove("hidden");
    document.getElementById("ann-crop-type").textContent = "Analyzing...";
    document.getElementById("ann-soil-type").textContent = "Analyzing...";
    document.getElementById("ann-details").classList.add("hidden");
    document.getElementById("action-plan-section").style.display = "none";
    document.getElementById("chat-section").style.display = "none";
    document.getElementById("chat-history").innerHTML = "";
    resultCard.querySelector(".confidence-section").classList.remove("hidden"); // Reset if needed

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
  if (result.mode === "hybrid_recommendation") {
    document.getElementById("ann-crop-type").textContent = result.crop_label.replace(/_/g, " ");
    // Hide confidence for tabular as requested
    document.querySelector("#ann-result-card .confidence-section").classList.add("hidden");
    document.querySelector("#ann-result-card .confidence-section").classList.add("hidden");
    
    // Show Action Plan & Chat
    if (result.action_plan) {
      document.getElementById("action-plan-section").style.display = "block";
      document.getElementById("chat-section").style.display = "block";
      
      let cleanPlan = result.action_plan.trim();
      if (cleanPlan.startsWith("```")) {
        cleanPlan = cleanPlan.replace(/^```(markdown|md)?\s*/i, "").replace(/\s*```$/i, "");
      }
      document.getElementById("action-plan-content").innerHTML = marked.parse(cleanPlan);
      
      // Initialize Chat Form listener if not already initialized
      const chatForm = document.getElementById("chat-form");
      chatForm.onsubmit = async (e) => {
        e.preventDefault();
        const input = document.getElementById("chat-input");
        const msg = input.value.trim();
        if(!msg) return;
        
        const history = document.getElementById("chat-history");
        history.innerHTML += `<div style="margin-bottom: 8px;"><strong>You:</strong> ${msg}</div>`;
        input.value = "";
        
        const typingId = "typing-" + Date.now();
        history.innerHTML += `<div id="${typingId}" style="margin-bottom: 8px; color: gray;"><em>Agent is typing...</em></div>`;
        
        try {
          const res = await fetch("/api/agent/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({message: msg})
          });
          const data = await res.json();
          document.getElementById(typingId).remove();
          if(res.ok) {
            let cleanReply = data.reply.trim();
            if (cleanReply.startsWith("```")) {
              cleanReply = cleanReply.replace(/^```(markdown|md)?\s*/i, "").replace(/\s*```$/i, "");
            }
            history.innerHTML += `<div style="margin-bottom: 8px; color: var(--primary);"><strong>Agent:</strong> ${marked.parse(cleanReply)}</div>`;
          } else {
            history.innerHTML += `<div style="margin-bottom: 8px; color: red;"><strong>Error:</strong> ${data.error}</div>`;
          }
        } catch(err) {
          document.getElementById(typingId).remove();
          history.innerHTML += `<div style="margin-bottom: 8px; color: red;"><strong>Error:</strong> Network issue</div>`;
        }
      };
    }
  } else {
    const label = result.label.replace(/_/g, " ");
    const probability = Math.round(result.top_probability * 100);
    document.getElementById("ann-soil-type").textContent = label;
    document.getElementById("ann-confidence-bar").style.width = `${probability}%`;
    document.getElementById("ann-confidence-percent").textContent = `${probability}%`;
  }

  const details = document.getElementById("ann-details");
  const note = result.warning ? `<div class="warning-note"><strong>Note:</strong> ${result.warning}</div>` : "";
  const title = result.mode === "crop_recommendation" ? "Confidence Rankings" : (result.model_name || "Probabilities");
  
  details.innerHTML = note + 
    `<div class="rankings-title">${title}</div>` +
    `<div class="rankings-grid">` +
    result.ranked_predictions
      .map((item) => `
        <div class="ranking-item">
          <span class="ranking-label">${item.label.replace(/_/g, " ")}</span>
          <span class="ranking-value">${Math.round(item.probability * 100)}%</span>
        </div>
      `)
      .join("") + 
    `</div>`;
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
  document.getElementById("cnn-ranked-list").classList.add("hidden");
  document.getElementById("cnn-confidence-bar").style.width = "0%";
  document.getElementById("cnn-confidence-percent").textContent = "0%";

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
  const probability = Math.round(result.top_probability * 100);
  
  document.getElementById("cnn-soil-type").textContent = label;
  document.getElementById("cnn-confidence-bar").style.width = `${probability}%`;
  document.getElementById("cnn-confidence-percent").textContent = `${probability}%`;

  // Smart Lock: Update and disable the manual soil type selector
  const manualSelector = document.getElementById("soil_type");
  if (manualSelector) {
    for (let i = 0; i < manualSelector.options.length; i++) {
      if (manualSelector.options[i].text.toLowerCase().includes(label.toLowerCase())) {
        manualSelector.selectedIndex = i;
        break;
      }
    }
    manualSelector.disabled = true;
    
    const parent = manualSelector.closest(".form-group");
    if (parent && !document.getElementById("lock-hint")) {
      const hint = document.createElement("span");
      hint.id = "lock-hint";
      hint.style.fontSize = "0.7rem";
      hint.style.color = "var(--primary)";
      hint.style.marginTop = "4px";
      hint.style.display = "block";
      hint.innerText = "✓ Locked to scanned result";
      parent.appendChild(hint);
    }
  }

  const list = document.getElementById("cnn-ranked-list");
  list.innerHTML =
    `<div class="rankings-title">Confidence Rankings</div>` +
    `<div class="rankings-grid">` +
    result.ranked_predictions
      .map((item) => `
        <div class="ranking-item">
          <span class="ranking-label">${item.label.replace(/_/g, " ")}</span>
          <span class="ranking-value">${Math.round(item.probability * 100)}%</span>
        </div>
      `)
      .join("") +
    `</div>`;
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

  // Toggle buttons
  document.getElementById("ann-toggle-details").addEventListener("click", () => {
    document.getElementById("ann-details").classList.toggle("hidden");
  });
  document.getElementById("cnn-toggle-details").addEventListener("click", () => {
    document.getElementById("cnn-ranked-list").classList.toggle("hidden");
  });

  document.getElementById("tabular-form").addEventListener("submit", handleTabularSubmit);
  document.getElementById("soil-form").addEventListener("submit", handleImageSubmit);
}

document.addEventListener("DOMContentLoaded", init);
