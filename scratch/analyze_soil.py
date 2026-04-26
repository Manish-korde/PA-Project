import pandas as pd
df = pd.read_csv('Dataset/solit_dataset/Soil-Climate-data.csv')
df_clean = df[df['Compatible'] == 1]
print(df_clean.groupby(['Soil_Type', 'Crop_Type']).size())
