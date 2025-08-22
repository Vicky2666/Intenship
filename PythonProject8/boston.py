import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load Ames Housing dataset
housing = fetch_openml(name="house_prices", as_frame=True)

df = housing.frame

# Drop rows with missing values for simplicity
df = df.select_dtypes(include=[np.number]).dropna()

# Features (selecting 13 numeric features similar to Boston Housing)
feature_names = [
    "LotArea", "OverallQual", "OverallCond", "YearBuilt",
    "1stFlrSF", "2ndFlrSF", "GrLivArea", "FullBath",
    "BedroomAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageCars", "GarageArea"
]

X = df[feature_names]
y = df["SalePrice"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, "boston_model.pkl")

print("Model trained and saved as boston_model.pkl")