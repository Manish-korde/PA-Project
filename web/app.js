let latestStatus = null;

const statusDot = document.querySelector(".status-dot");
const statusText = document.getElementById("status-text");
const debugLog = document.getElementById("debug-log");
const auditWarnings = document.getElementById("audit-warnings");
const recommendationSummary = document.getElementById("recommendation-summary");
const recommendationMode = document.getElementById("recommendation-mode");

const formatJSON = (obj) => JSON.stringify(obj, null, 2);

let soilAnalysisLoaded = false;
let imageAnalysisLoaded = false;

const CROP_ICONS = {
  "rice": "fa-solid fa-wheat-awn",
  "maize": "fa-solid fa-seedling",
  "chickpea": "fa-solid fa-bowl-food",
  "kidneybeans": "fa-solid fa-seedling",
  "pigeonpeas": "fa-solid fa-seedling",
  "mothbeans": "fa-solid fa-seedling",
  "mungbean": "fa-solid fa-seedling",
  "blackgram": "fa-solid fa-seedling",
  "lentil": "fa-solid fa-seedling",
  "pomegranate": "fa-solid fa-apple-whole",
  "banana": "fa-solid fa-leaf",
  "mango": "fa-solid fa-apple-whole",
  "grapes": "fa-solid fa-grapes",
  "watermelon": "fa-solid fa-apple-whole",
  "muskmelon": "fa-solid fa-apple-whole",
  "apple": "fa-solid fa-apple-whole",
  "orange": "fa-solid fa-apple-whole",
  "papaya": "fa-solid fa-leaf",
  "coconut": "fa-solid fa-leaf",
  "cotton": "fa-solid fa-clover",
  "jute": "fa-solid fa-leaf",
  "coffee": "fa-solid fa-mug-hot"
};

const SOIL_ICONS = {
  "alluvial": "fa-solid fa-water",
  "black": "fa-solid fa-mountain",
  "clay": "fa-solid fa-faucet-drip",
  "red": "fa-solid fa-fire",
  "laterite": "fa-solid fa-brick",
  "marshy": "fa-solid fa-water",
  "sandy": "fa-solid fa-umbrella-beach"
};

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
      if (target === "model-insights" && !imageAnalysisLoaded) {
        loadImageAnalysis();
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
    const crop = result.crop_label.toLowerCase().replace(/_/g, "").replace(/\s/g, "");
    const soil = result.soil_label ? result.soil_label.toLowerCase().replace(/_/g, "").replace(/\s/g, "") : "";
    
    // Update Crop Label and Icon
    document.getElementById("ann-crop-type").textContent = result.crop_label.replace(/_/g, " ");
    const cropIconWrapper = document.getElementById("crop-icon-wrapper");
    if (cropIconWrapper && CROP_ICONS[crop]) {
      cropIconWrapper.innerHTML = `<i class="${CROP_ICONS[crop]}"></i>`;
    }

    // Update the result card
    if (result.soil_label) {
      const label = result.soil_label.replace(/_/g, " ");
      const soilDisplay = document.getElementById("ann-soil-type");
      if (soilDisplay) {
        soilDisplay.textContent = label;
      }
      
      // Update Soil Icon if possible
      const soilContainer = soilDisplay ? soilDisplay.closest(".main-prediction") : null;
      const soilIconEl = soilContainer ? soilContainer.querySelector(".prediction-icon") : null;
      if (soilIconEl && SOIL_ICONS[soil]) {
        soilIconEl.innerHTML = `<i class="${SOIL_ICONS[soil]}"></i>`;
      }
      
      // Smart Lock: Update and disable the manual soil type selector
      const manualSelector = document.getElementById("soil_type");
      if (manualSelector) {
        // Try to find matching option
        for (let i = 0; i < manualSelector.options.length; i++) {
          if (manualSelector.options[i].text.toLowerCase().includes(label.toLowerCase())) {
            manualSelector.selectedIndex = i;
            break;
          }
        }
        manualSelector.disabled = true;
        
        // Add a small hint that it's locked
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
    }
    // Hide confidence for tabular as requested
    const confSection = document.querySelector("#ann-result-card .confidence-section");
    if (confSection) confSection.classList.add("hidden");
    
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
  const fileInput = document.getElementById("soil-image");
  const preview = document.getElementById("image-preview");
  const previewContainer = document.getElementById("image-preview-container");

  // Click to upload
  dropZone.addEventListener("click", () => fileInput.click());

  // Drag and drop support
  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelect();
    }
  });

  fileInput.addEventListener("change", handleFileSelect);

  function handleFileSelect() {
    const file = fileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      previewContainer.classList.remove("hidden");
      dropZone.classList.add("hidden");
    };
    reader.readAsDataURL(file);
  }

  // Remove image functionality
  document.getElementById("remove-image")?.addEventListener("click", () => {
    fileInput.value = "";
    preview.src = "";
    previewContainer.classList.add("hidden");
    dropZone.classList.remove("hidden");
  });
}

async function handleImageSubmit(event) {
  event.preventDefault();
  const fileInput = document.getElementById("soil-image");
  const resultCard = document.getElementById("cnn-result-card");
  const file = fileInput.files[0];
  if (!file) {
    alert("Please select or drag an image first.");
    return;
  }

  resultCard.classList.remove("hidden");
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
    let found = false;
    const baseName = label.toLowerCase().replace(" soil", "");
    for (let i = 0; i < manualSelector.options.length; i++) {
      if (manualSelector.options[i].text.toLowerCase().includes(baseName)) {
        manualSelector.selectedIndex = i;
        found = true;
        break;
      }
    }
    
    if (!found) {
      const newOption = document.createElement("option");
      newOption.value = result.label;
      newOption.text = label;
      manualSelector.appendChild(newOption);
      manualSelector.selectedIndex = manualSelector.options.length - 1;
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

async function loadImageAnalysis() {
  const status = document.getElementById("image-analysis-status");
  const grid = document.getElementById("image-plot-grid");
  if (!status || !grid) return;

  status.textContent = "Loading image model insights...";
  grid.innerHTML = "";

  try {
    const response = await fetch("/api/image-metrics/list");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to load image insights");

    const plots = data.plots || [];
    if (!plots.length) {
      status.textContent = "No image model insights available.";
      imageAnalysisLoaded = true;
      return;
    }

    for (const plot of plots) {
      const card = document.createElement("div");
      card.className = "plot-card";
      card.innerHTML = `
        <h3>${plot.title}</h3>
        <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">${plot.description}</p>
        <div class="plot-container">
          <img src="/api/image-metrics/plot?name=${plot.name}" alt="${plot.title}">
        </div>
      `;
      grid.appendChild(card);
    }

    status.textContent = "Analysis based on source dataset and latest CNN training metrics.";
    imageAnalysisLoaded = true;
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    console.error(err);
  }
}

document.addEventListener("DOMContentLoaded", init);
