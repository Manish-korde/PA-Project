import pandas as pd
df = pd.read_csv('Dataset/solit_dataset/Soil-Climate-data.csv')
df_clean = df[df['Compatible'] == 1]
features = ["Soil_pH", "Soil_Nitrogen", "Soil_Organic_Matter", "Temperature", "Rainfall", "Humidity"]
stats = df_clean.groupby('Crop_Type')[features].mean()
print(stats)
