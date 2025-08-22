import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("car.csv")

# Features & target
X = data.drop(columns=["Car_Name", "Selling_Price"])
y = data["Selling_Price"]

# One-hot encode categorical columns
X = pd.get_dummies(X, drop_first=True)

# Save feature names
pickle.dump(X.columns, open("features.pkl", "wb"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("✅ Model and features saved successfully!")