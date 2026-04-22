# create_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

os.makedirs('models', exist_ok=True)

# ============================================
# CREATE SKIN MODEL (5 features)
# Features in app.py: redness, oiliness, wrinkles, dark_spots, texture
# ============================================
print("[INFO] Creating skin analysis model...")

skin_data = {
    'redness': [0.8, 0.2, 0.3, 0.4, 0.1, 0.9, 0.2, 0.3, 0.7, 0.8, 0.2, 0.3, 0.9, 0.1, 0.4],
    'oiliness': [0.9, 0.3, 0.2, 0.8, 0.2, 0.1, 0.8, 0.3, 0.2, 0.9, 0.1, 0.2, 0.3, 0.7, 0.8],
    'wrinkles': [0.1, 0.2, 0.9, 0.2, 0.3, 0.1, 0.2, 0.8, 0.2, 0.1, 0.9, 0.7, 0.2, 0.3, 0.1],
    'dark_spots': [0.2, 0.1, 0.3, 0.9, 0.2, 0.1, 0.2, 0.3, 0.8, 0.2, 0.1, 0.2, 0.9, 0.7, 0.1],
    'texture': [0.7, 0.8, 0.2, 0.3, 0.9, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.9, 0.1, 0.2, 0.3]
}

skin_labels = [
    'Acne', 'Dry Skin', 'Wrinkles', 'Oily Skin', 'Normal', 
    'Redness', 'Acne', 'Wrinkles', 'Dark Spots', 'Acne', 
    'Dry Skin', 'Wrinkles', 'Dark Spots', 'Oily Skin', 'Acne'
]

skin_df = pd.DataFrame(skin_data)
skin_le = LabelEncoder()
y_skin = skin_le.fit_transform(skin_labels)

skin_model = RandomForestClassifier(n_estimators=50, random_state=42)
skin_model.fit(skin_df.values, y_skin)

joblib.dump(skin_model, 'models/skin_model.pkl')
joblib.dump(skin_le, 'models/skin_label_encoder.pkl')

print("[OK] Skin model created and saved!")

# ============================================
# CREATE HAIR MODEL (4 features)
# Features in app.py: hair_density, oiliness, dandruff, redness
# ============================================
print("[INFO] Creating hair analysis model...")

hair_data = {
    'hair_density': [0.1, 0.9, 0.8, 0.2, 0.7, 0.2, 0.9, 0.8, 0.2, 0.3],
    'oiliness': [0.2, 0.1, 0.8, 0.9, 0.3, 0.2, 0.1, 0.9, 0.8, 0.2],
    'dandruff': [0.1, 0.2, 0.1, 0.8, 0.9, 0.1, 0.2, 0.8, 0.9, 0.1],
    'redness': [0.8, 0.1, 0.2, 0.3, 0.8, 0.9, 0.2, 0.1, 0.8, 0.2]
}

hair_labels = [
    'Thinning Hair', 'Normal', 'Oily Scalp', 'Oily Scalp', 'Dandruff',
    'Thinning Hair', 'Normal', 'Oily Scalp', 'Dandruff', 'Thinning Hair'
]

hair_df = pd.DataFrame(hair_data)
hair_le = LabelEncoder()
y_hair = hair_le.fit_transform(hair_labels)

hair_model = RandomForestClassifier(n_estimators=50, random_state=42)
hair_model.fit(hair_df.values, y_hair)

joblib.dump(hair_model, 'models/hair_model.pkl')
joblib.dump(hair_le, 'models/hair_label_encoder.pkl')

print("[OK] Hair model created and saved!")