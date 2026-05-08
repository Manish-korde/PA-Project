# SoilIntel: Syllabus Mapping & Implementation Guide
## Purpose-Driven Academic Concepts

This document maps the project's technical features to your college syllabus, explaining the **purpose** and **evidence** (plots/code) for each concept.

---

### 📘 Unit 1: Introduction to AI & Classification
*   **Purpose:** To categorize soil and climate data into distinct, actionable labels (Crops and Soil Types).
*   **Implementation:** 
    *   **Baseline Identification:** We analyzed the dataset to identify target classes using the **Crop Type Distribution** plot. This helped us understand the class balance (22 distinct crops) before training.
    *   **Feature Engineering:** Input features like Nitrogen (N), Phosphorus (P), and Potassium (K) were mapped to these classes.
*   **Key Evidence:** "Distributions" plot showing the frequency of crops in our source dataset.

### 📘 Unit 2: Supervised Learning (SVM, RF, ANN)
*   **Purpose:** To build a predictive model that accurately recommends the best crop based on historical "labeled" success data.
*   **Implementation:**
    *   **Model Comparison:** We didn't just pick one model; we used a **Classifier Accuracy Comparison** plot to test Logistic Regression, Decision Trees, and Random Forest.
    *   **Primary Model:** **Random Forest** was selected as the primary advisor due to its high accuracy (99%+) and robustness against overfitting.
    *   **Deep Learning (ANN/DNN):** We compared a standard Artificial Neural Network (ANN) with a Deep Neural Network (DNN) using **Training/Validation Curves** to ensure the model converges correctly without losing generalization.
*   **Key Evidence:** "Classifier Accuracy" bar chart and "ANN/DNN Test Accuracy" comparison plots.

### 📘 Unit 3: Unsupervised Learning (Clustering)
*   **Purpose:** To discover hidden relationships in soil data without using labels, identifying which regional climates naturally group together.
*   **Implementation:**
    *   **K-Means Clustering:** Grouped our soil dataset into 3 primary clusters based on environmental variables.
    *   **Optimization:** Used the **Elbow Method (WCSS)** to mathematically determine the most efficient number of clusters.
*   **Key Evidence:** "KMeans Scatter (Temperature vs Rainfall)" showing clear regional groupings, and the "KMeans Elbow" plot.



---

## 🚜 System Workflow: User Experience

### Step 1: Visual Scan (Tab 1)
1.  **Farmer Interaction:** The farmer drags a photo of their soil into the "Visual Scan" zone.
2.  **Processing:** The system pings the **CNN Model** (`soil_cnn_updated_model.pth`).
3.  **Visual Support:** Simultaneously, a **Micro-Texture Enhancement** pipeline runs. It sharpens cracks and highlights mineral hues to help a human agronomist verify the AI's scan.
4.  **The Result:** The soil type is identified (e.g., "Black Soil"), and the **Smart Lock** feature instantly sends this data to the next tab.

### Step 2: Crop Advisor (Tab 2)
1.  **Automation:** All inputs (Soil Type, NPK, pH) are **automatically pre-filled** based on the scientific baselines for the soil identified in Tab 1.
2.  **Analyze Button:** When the farmer clicks **"Analyze Parameters"**:
    *   The **Random Forest Model** predicts the best crop (e.g., "Cotton").
    *   The **FarmAgent (Agentic AI)** triggers a reasoning loop (ReAct) to fetch real-time weather and create a multi-step "Action Plan."
    *   The **Chatbot** opens, allowing the farmer to ask follow-up questions about fertilizers or irrigation.

---

## 📊 Data Insights & Augmentation Trace

### Expanding the Data (Unit 6 Concepts)
We started with a source dataset of **832 images**, which was insufficient for a deep CNN. To solve this:
*   **In-Memory Augmentation:** We implemented a stratified augmentation pipeline that virtually expanded the training set to **8,000 samples** (roughly 1,140 images per class).
*   **Log Verification:** This process is recorded in the `Models/final models/soil_cnn_updated_model_metrics.json` log file, which explicitly tracks the `train_source_images: 832` vs `train: 8000` counts.
*   **Augmentation Impact Graph:** The "AI Insights" tab features a bar chart showing the "Original vs. Augmented" distribution, proving how we balanced the dataset classes.

### Performance Verification
The "AI Insights" tab displays the **CNN Training Metrics** (Validation Accuracy over Epochs) from the log file, proving the reliability of the visual scan feature.
