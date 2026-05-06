import os
from agent import FarmAgent

def main():
    # 1. Initialize the Agent
    # Note: Ensure GROQ_API_KEY is set in your .env file for full agentic capabilities.
    agent = FarmAgent()
    
    # 2. Sample data from the dataset (Rice example)
    # N=90, P=42, K=43, temperature=20.88, humidity=82.00, ph=6.50, rainfall=202.94
    features = {
        "soil_nitrogen": 90,
        "soil_phosphorus": 42,
        "soil_potassium": 43,
        "temperature": 20.88,
        "humidity": 82.00,
        "soil_ph": 6.50,
        "rainfall": 202.94
    }
    
    # 3. Simulate getting results from ML models
    crop_result = {
        "label": "rice",
        "top_probability": 0.98
    }
    soil_result = "Alluvial" # Example soil type
    
    print("--- Applying Agentic AI Model ---")
    print(f"Input Features: {features}")
    print(f"Model Prediction: {crop_result['label']} ({crop_result['top_probability']*100:.1f}%)")
    
    # 4. Generate Action Plan using the Agent
    print("\nGenerating Action Plan...")
    action_plan = agent.generate_action_plan(features, crop_result, soil_result)
    print("\n--- Agent Action Plan ---")
    print(action_plan)
    
    # 5. Interactive Chat with the Agent (Example)
    print("\n--- Interactive Chat Example ---")
    question = "What fertilizer should I use for rice in these conditions?"
    print(f"Farmer: {question}")
    reply = agent.chat(question)
    print(f"Agent: {reply}")

if __name__ == "__main__":
    main()
