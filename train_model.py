import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

fashion_df = pd.read_csv('fashion_mnist.csv')

X = fashion_df.drop('label', axis=1)
y = fashion_df['label']

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Standardize the features
X_scaled = scaler.transform(X)

# Load the model
with open('fashion_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make predictions
y_pred = model.predict(X_scaled)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

# Print accuracy
print(f"Accuracy: {accuracy}")

# Print classification report
print(f"Classification Report:\n{report}")

# Print predicted latest trends for example
latest_trends = model.predict(X[:10])  # Example: Predicting trends for first 10 samples
print("Predicted Latest Trends:")
for i, trend in enumerate(latest_trends):
    print(f"Sample {i+1}: Predicted Class {trend} - {fashion_df['label'].unique()[trend]}")

