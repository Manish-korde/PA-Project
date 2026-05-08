# SoilIntel: Hybrid Agricultural Analysis Platform
## Comprehensive Project Documentation & Presentation Guide

This document serves as the **Master Blueprint** for the SoilIntel project. It is designed to equip any reader—whether a developer, presenter, or stakeholder—with a deep understanding of the architecture, machine learning models, Agentic AI, and system workflows.

---

## 1. Executive Summary

**SoilIntel** is an advanced, AI-driven agricultural platform that helps farmers and agronomists make data-driven decisions. The platform utilizes a **Hybrid Multi-Model Architecture**:
1. **Computer Vision (CNN):** Identifies soil types from smartphone photos with micro-texture enhancement.
2. **Machine Learning (Random Forest / ANN):** Recommends optimal crops based on NPK, pH, and climate parameters.
3. **Agentic AI (ReAct Pattern):** Acts as a virtual agronomist, using tools to fetch real-time weather and analyze data.
4. **Local Predictive Engine:** Provides offline-capable sequence forecasting (LSTM) and NLP-driven local analysis.

---

## 2. Technology Stack

* **Frontend:** Vanilla HTML5, CSS3 (Glassmorphism, CSS Variables), and JavaScript (ES6+). Uses `marked.js` for AI rendering and `FontAwesome` for iconography.
* **Backend:** Native Python `ThreadingHTTPServer`. A custom-built REST API without heavy frameworks, demonstrating core networking proficiency.
* **Computer Vision:** PyTorch CNN for high-accuracy soil classification.
* **Machine Learning:** Scikit-Learn (Random Forest) and PyTorch (ANN) for crop recommendation.
* **Sequence Modeling:** PyTorch LSTM for time-series nutrient forecasting.
* **Generative AI:** Groq API (`llama-3.3-70b-versatile`) for high-speed agent reasoning.
* **Natural Language Processing:** NLTK for local intent detection and lemmatization.

---

## 3. Advanced Features & Innovation

### A. Micro-Texture Enhancement Trace
Since field photos are often low-quality, we implemented a custom enhancement pipeline. It uses noise reduction and sharpness synthesis to highlight soil cracks and mineral hues, allowing human experts to verify AI decisions.

### B. Smart Lock Automation
When a soil image is scanned via the CNN, the system triggers a "Smart Lock." It automatically pre-fills the identified soil type in the tabular analysis tab and locks the input, creating a seamless bridge between Computer Vision and Tabular ML.

### C. ReAct Agentic Workflow
The `FarmAgent` uses the **Reasoning and Acting (ReAct)** framework. It can decide to use external tools (Weather API, Fertilizer Pricing) or internal ML models before providing a final recommendation.

### D. Hybrid Local/Cloud LLM
The platform features a dual-layer AI approach:
- **Cloud (Groq):** For complex reasoning and multi-step tool use.
- **Local (SoilIntel Engine):** For instant, offline-capable analysis of soil trends using LSTM and NLP intent mapping.

### E. Interactive Data Visualization
The "Soil Data" and "AI Insights" tabs provide high-resolution plots of training performance, clustering (K-Means), and dataset distribution. All charts feature an **Enlarge on Click** zoom feature for detailed inspection.

---

## 4. Machine Learning & Deep Learning Detail

### Soil Identification (CNN)
- **Architecture:** Sequential Conv2d layers with ReLU and MaxPool, followed by a Dropout-stabilized Linear classifier.
- **Training:** Uses **In-Memory Data Augmentation** (Rotation, Jitter, Noise) to expand a small dataset into 8,000+ virtual samples without disk overhead.

### Crop Recommendation (Random Forest & ANN)
- **Random Forest:** Preferred for tabular data due to its ensemble nature and resistance to overfitting.
- **ANN:** Provides a deep-learning alternative for complex non-linear relationships.
- **Preprocessing:** Standard Scaler and Label Encoding are used to normalize environmental ranges (0-2000mm rainfall, etc.).

### Nutrient Forecasting (LSTM)
- **Concept:** Uses Unit-5 sequence modeling to predict future Nitrogen levels based on a sliding window of historical data.

---

## 5. Syllabus Alignment (Quick Reference)

| Unit | Concept | Project Implementation |
| :--- | :--- | :--- |
| **Unit 1** | Naive Bayes / Classification | `crop_recommendation_model_comparison.ipynb` (Baseline models). |
| **Unit 2** | SVM / Random Forest | Random Forest is the primary Crop Advisor model. |
| **Unit 3** | K-Means Clustering | "Soil Data" tab features K-Means plots for soil grouping. |
| **Unit 4** | NLP / Text Processing | `local_llm.py` uses Lemmatization and Intent Parsing. |
| **Unit 5** | LSTM / RNN | `Lstm.ipynb` and `local_llm.py` for time-series prediction. |
| **Unit 6** | CNN / GAN / Augmentation | Soil CNN and UI-based Data Augmentation Trace. |

---

## 6. Setup & Execution

1. **Environment:** Create `.env` with `GROQ_API_KEY=gsk_...`
2. **Dependencies:** `pip install torch torchvision scikit-learn pandas joblib python-dotenv nltk`
3. **Run:** `python app.py`
4. **Access:** `http://127.0.0.1:8000/`
