import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

df = pd.read_csv('Dataset/solit_dataset/Soil-Climate-data.csv')
df_clean = df[df['Compatible'] == 1]

X = df_clean[["Soil_pH", "Soil_Nitrogen", "Soil_Organic_Matter", "Temperature", "Rainfall", "Humidity"]]
y = df_clean["Crop_Type"]

# Encode
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 1. SPLIT FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

# 2. SMOTE ONLY TRAINING
print(f"Before SMOTE: {len(X_train)} samples")
sm = SMOTE(random_state=42, k_neighbors=1) # k=1 because some classes are tiny
X_res, y_res = sm.fit_resample(X_train, y_train)
print(f"After SMOTE: {len(X_res)} samples")

# 3. Train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_res, y_res)
preds = rf.predict(X_test)
print(f"SMOTE-Sampled Accuracy (Honest): {accuracy_score(y_test, preds):.4f}")
