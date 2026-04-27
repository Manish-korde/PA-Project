# SoilIntel: Hybrid Agricultural Analysis Platform
## Comprehensive Project Documentation & Presentation Guide

This document serves as the "Master Blueprint" for the SoilIntel project. It is designed to equip any reader—whether a developer, presenter, or stakeholder—with a deep understanding of the architecture, machine learning models, Agentic AI, and system workflows necessary to present the project or answer technical questions.

---

## 1. Executive Summary

**SoilIntel** is an advanced, AI-driven agricultural platform that helps farmers and agronomists make data-driven decisions. Instead of relying on a single technology, the platform uses a **hybrid approach**:
1. **Computer Vision (CNN)** to identify soil types from uploaded images.
2. **Machine Learning (Random Forest / ANN)** to recommend the best crops based on soil and climate parameters.
3. **Agentic AI (ReAct Pattern via Llama 3.3)** to act as a virtual agronomist, analyzing the ML results, fetching real-time weather data, and generating a highly personalized, actionable farming plan.

---

## 2. Technology Stack

* **Frontend:** Vanilla HTML, CSS (Glassmorphism design, CSS Variables), and JavaScript (ES6+). Uses `marked.js` for rendering AI markdown responses. No heavy frameworks (React/Vue) to keep the client lightweight.
* **Backend:** Native Python `http.server.ThreadingHTTPServer`. A custom-built, multi-threaded REST API without heavy frameworks like Flask or Django, ensuring low overhead.
* **Machine Learning:** PyTorch (Neural Networks/CNNs), Scikit-Learn (Random Forest, Label Encoders, Scalers), Joblib.
* **Generative AI:** Groq API (OpenAI-compatible REST endpoints) utilizing the `llama-3.3-70b-versatile` model.
* **External APIs:** Open-Meteo (for real-time geolocation and weather forecasting).

---

## 3. Core Features & System Workflow

The platform is divided into three main user workflows (Tabs):

### A. Visual Identification (Image to Soil)
* **How it works:** The user uploads a photo of their soil. The image is converted to Base64 and sent to the backend `/api/soil/predict`.
* **The Brains:** A custom Convolutional Neural Network (CNN) processes the image and returns the identified soil type (e.g., "Alluvial soils") along with a confidence percentage.
* **Smart UI Interaction:** Upon a successful scan, the frontend triggers a "Smart Lock." It automatically switches to the Parameter-Based Analysis tab, selects the identified soil type in the dropdown, and locks it so the user doesn't have to enter it manually.

### B. Parameter-Based Analysis (Tabular Data to Crop)
* **How it works:** The user inputs precise environmental data: Nitrogen (N), Phosphorus (P), Potassium (K), pH, Farm Size, Organic Matter, Temperature, Rainfall, and Humidity.
* **The Brains:** The data is sent to `/api/soil/predict-tabular`. It is scaled using a saved standard scaler and fed into a **Random Forest Classifier**.
* **The Output:** The model predicts the most suitable crop (e.g., "Coffee") out of 22 possible classes.

### C. The Agronomist AI (Data to Action)
* **How it works:** Once the ML model predicts the crop, the backend automatically hands the parameters and the ML prediction over to the `FarmAgent`.
* **The Brains:** The Agent utilizes the **ReAct (Reasoning and Acting)** framework. It doesn't just guess; it uses "Tools".
* **The Output:** The Agent outputs a structured Farm Action Plan (Suitability Reasoning, Risks, Actionable Steps). The user can then use the built-in chat interface to ask follow-up questions (e.g., "What specific fertilizer brand should I use?").

---

## 4. Deep Dive: Machine Learning Models

All production models are stored in `Models/final models/`.

### Crop Recommendation (Random Forest - Active)
* **Algorithm:** Random Forest Classifier (`crop_rf_model.pkl`).
* **Why RF over ANN?** Tabular datasets with non-linear environmental relationships (like NPK ratios vs. specific crop thresholds) often perform better and are less prone to overfitting with ensemble tree methods than with standard Artificial Neural Networks.
* **Inputs:** N, P, K, Temperature, Humidity, pH, Rainfall.

### Soil Classification (CNN - Active)
* **Algorithm:** PyTorch Convolutional Neural Network (`soil_cnn_updated_model.pth`).
* **Purpose:** Image-based classification.

### Legacy/Fallback Models
* The repository also contains ANN models (`crop_ann_model.pth`, `soil_ann_model_7class.pth`). The backend architecture is designed to fall back to these if the primary models fail to load, ensuring system resilience.

---

## 5. Deep Dive: Agentic AI & ReAct Pattern

The true innovation of this project is the `FarmAgent` (`agent.py`). It does not use standard conversational AI; it uses an **Agentic framework**.

### What is ReAct?
ReAct stands for **Reasoning and Acting**. Instead of just answering a prompt, the LLM is given a system prompt explaining that it has access to specific Python functions (Tools). 
1. **Thought:** The AI thinks: "The user needs a farm plan for Coffee. I should check the weather for their location to advise on irrigation."
2. **Action:** The AI outputs a specific command string: `ACTION: get_weather({"location": "..."})`
3. **Observation:** Our Python backend intercepts this, runs the `get_weather` function using the Open-Meteo API, and injects the result back into the prompt as an observation.
4. **Final Answer:** Once the AI has gathered all necessary data, it generates the final Markdown report.

### The Tools Available to the Agent:
1. `run_crop_model()`: Allows the agent to independently verify crop suitability.
2. `get_weather(location)`: Fetches 7-day max/min temps and precipitation from Open-Meteo.
3. `get_fertilizer_price(name)`: Looks up local market prices for urea, DAP, etc., to provide cost-aware recommendations.

---

## 6. Backend Routing

The `app.py` script extends `BaseHTTPRequestHandler` to create a lightweight REST API.
* `GET /`: Serves the `index.html` frontend.
* `GET /api/status`: Returns system health, loaded models, and AI provider status.
* `POST /api/soil/predict`: Handles Base64 image uploads -> CNN model.
* `POST /api/soil/predict-tabular`: Handles JSON parameters -> RF model -> Agentic AI Plan.
* `POST /api/agent/chat`: Handles follow-up conversations with the context-aware Agent.

---

## 7. Setup & Execution

1. **Environment Variables:** Ensure `.env` exists in the root directory and contains `GROQ_API_KEY=gsk_...`
2. **Dependencies:** `pip install torch torchvision scikit-learn pandas joblib python-dotenv`
3. **Run the Server:** `python app.py`
4. **Access the UI:** Open a browser and navigate to `http://127.0.0.1:8000/`

---

## 8. Presentation Q&A Preparation

If you are presenting this project, be prepared to answer these common technical questions:

**Q: Why didn't you use a framework like Flask or FastAPI for the backend?**
*A: We wanted to demonstrate a deep understanding of Python's underlying networking and HTTP protocols. By using `http.server.ThreadingHTTPServer`, we kept the application incredibly lightweight, reduced external dependencies, and implemented a custom multi-threaded router that perfectly fits our specific needs.*

**Q: Why use Groq / Llama-3 instead of ChatGPT or Gemini?**
*A: Through rigorous testing, we found that Agentic workflows (ReAct loops) require multiple rapid API calls to execute tools (like checking the weather). Groq's LPU architecture provides near-instantaneous inference speeds, meaning the Agent can think, use a tool, and respond in a fraction of the time it takes traditional cloud providers. We originally used Gemini, but hit strict rate limits (429 errors), making Groq the superior architectural choice.*

**Q: How does the "Smart Lock" feature work?**
*A: It's an event-driven UI interaction. When the `/api/soil/predict` endpoint successfully returns a classification from the CNN, the frontend JavaScript parses the result, maps it to the DOM options in the Parameter tab, programmatically selects it, and sets the element to `disabled=true`. This bridges the Computer Vision capability with the Tabular Machine Learning capability seamlessly.*

**Q: What happens if the API key fails or the user goes offline?**
*A: The system is designed with grace-degradation. If `GROQ_API_KEY` is missing or the network drops, `agent.py` detects that the LLM is unavailable and instantly switches to `_mock_action_plan()`. The user still gets their core ML predictions and a statically generated farm plan, preventing a hard crash.*

**Q: How does the AI know current weather if LLMs are frozen in time?**
*A: The LLM itself doesn't know. We gave it a `get_weather` tool. The LLM simply asks our Python backend to run the tool, our backend pings the Open-Meteo REST API, and feeds the live JSON data back into the LLM's context window. This is the definition of an "Agentic" AI vs a standard Chatbot.*
