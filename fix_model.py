import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

print("🔄 Loading data...")
df = pd.read_csv('Crop_recommendation.csv')

# Feature columns (all 10)
feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 
                'NPK_ratio', 'temp_humidity_index', 'soil_fertility_score']

X = df[feature_cols]
y = df['label']

print(f"📊 Using {len(feature_cols)} features: {feature_cols}")
print(f"📈 Dataset shape: {X.shape}")
print(f"🌱 Crop types: {len(y.unique())}")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Create and fit scalers for 10 features
print("📏 Creating scalers...")
ms = MinMaxScaler()
sc = StandardScaler()

X_scaled = ms.fit_transform(X)
X_final = sc.fit_transform(X_scaled)

print("🤖 Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_final, y_encoded)

print("💾 Saving model and scalers...")
# Save with compatible protocol
pickle.dump(model, open('model.pkl', 'wb'), protocol=4)
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'), protocol=4)
pickle.dump(sc, open('standscaler.pkl', 'wb'), protocol=4)

print("✅ Model and scalers saved successfully with 10 features!")
print("🚀 Now run: python app.py")
