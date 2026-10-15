import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load your model and data
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Load original dataset
df = pd.read_csv('Crop_recommendation.csv')

# Use all 10 features directly from dataset (already calculated)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'NPK_ratio', 'temp_humidity_index', 'soil_fertility_score']].copy()

# Use LabelEncoder to match how model was trained
le = LabelEncoder()
y = le.fit_transform(df['label'])

# Split into train and test (80/20 split) - REAL ACCURACY TEST
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Reorder columns to match training data order
X_test = X_test[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'NPK_ratio', 'temp_humidity_index', 'soil_fertility_score']]

# Make predictions on TEST data only (unseen data)
X_scaled = ms.transform(X_test)
X_final = sc.transform(X_scaled)
predictions = model.predict(X_final)

# Calculate accuracy on TEST data (unseen data)
accuracy = accuracy_score(y_test, predictions)
print(f"REAL Model Accuracy: {accuracy * 100:.2f}%")

# Debug: Check test results
print(f"Test samples: {len(y_test)}")
print(f"Correct predictions: {sum(y_test == predictions)}")
print(f"Wrong predictions: {sum(y_test != predictions)}")

# Show some sample predictions vs actual from TEST data
print("\nSample predictions vs actual (TEST DATA):")
for i in range(5):
    actual = le.inverse_transform([y_test[i]])[0]
    predicted = le.inverse_transform([predictions[i]])[0]
    print(f"Actual: {actual:10} | Predicted: {predicted:10} | Match: {y_test[i] == predictions[i]}")