# Assignment: Applying Agentic AI to Agricultural Datasets

## 1. Project Overview
The goal of this assignment is to demonstrate the application of an **Agentic AI Model** to a specific dataset. For this project, we utilize the **Crop Recommendation Dataset**, which contains environmental parameters such as Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall to recommend the most suitable crop for a given piece of land.

## 2. Dataset Description
- **Source:** Crop Recommendation Dataset (CSV format).
- **Features:**
  - `N`: Ratio of Nitrogen content in soil.
  - `P`: Ratio of Phosphorus content in soil.
  - `K`: Ratio of Potassium content in soil.
  - `temperature`: Temperature in degrees Celsius.
  - `humidity`: Relative humidity in %.
  - `ph`: pH value of the soil.
  - `rainfall`: Rainfall in mm.
- **Target:** `label` (The specific crop recommended, e.g., rice, maize, chickpea, etc.).

## 3. Agentic AI Model Selection
We have implemented a **Farm Planning Agent** based on the **ReAct (Reasoning and Acting)** framework.

### Why "Agentic"?
Unlike a standard classifier that just outputs a label, an agent can:
1.  **Reason**: Analyze the ML model's prediction alongside environmental constraints.
2.  **Act**: Use external tools (e.g., fetching real-time weather data or fertilizer prices).
3.  **Iterate**: Engage in a dialogue with the user to refine the agricultural plan.

### Technical Stack
- **Base LLM:** Llama-3-70B (via Groq API).
- **Architecture:** ReAct loop (Thought -> Action -> Observation -> Response).
- **Tools Integrated:**
  - `get_weather`: Fetches real-time forecast for local alignment.
  - `get_fertilizer_price`: Retrieves current market rates for soil supplements.
  - `run_crop_model`: Calls local Random Forest/ANN models for primary prediction.

## 4. Implementation Logic
The agent follows a multi-step reasoning process:
1.  **Input Consumption:** Receives soil and climate parameters from the dataset.
2.  **Inference:** Invokes the local ML model (Trained on the dataset) to get a base recommendation.
3.  **Tool Engagement:** If the user asks for specific advice (e.g., "When should I plant?"), the agent calls `get_weather` to see if current rainfall matches the crop's needs.
4.  **Synthesis:** Combines ML prediction + Tool data + Knowledge Base to generate a Markdown-formatted **Action Plan**.

## 5. Sample Application
### Scenario: Rice Recommendation
**Input:** `N=90, P=42, K=43, ph=6.5, Rainfall=202mm`
**ML Model Prediction:** `Rice (98% Confidence)`

**Agent Reasoning (Internal Thought):**
> "The ML model suggests Rice. Given the high rainfall (202mm) and pH of 6.5, this is highly suitable. I should check the upcoming weather to advise on irrigation and lookup Urea prices for N-tuning."

**Final Agent Output:**
### Agronomist Action Plan
**Assessment:**
The environmental conditions (High Rainfall & Neutral pH) are ideal for **Rice** cultivation.

**Recommendations:**
1.  **Nitrogen Management:** Since N is at 90, supplement with **Urea** (current price ~₹7/kg) to maintain peak growth.
2.  **Irrigation:** The 10-day forecast shows continued rain; ensure proper drainage to prevent waterlogging.
3.  **Soil Health:** Maintain pH at 6.5 using organic compost to keep soil porous.

## 6. Conclusion
By applying an Agentic AI layer over the traditional ML classification dataset, we transform static predictions into **dynamic, actionable intelligence**. This bridge between "data output" and "real-world action" is the core value proposition of Agentic systems in precision agriculture.
