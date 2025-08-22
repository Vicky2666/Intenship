from flask import Flask, render_template, request
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64
import numpy as np
import pandas as pd

app = Flask(__name__)  # ✅ FIXED here

# Load Boston Housing dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)
X, y = boston.data, boston.target
feature_names = X.columns.tolist()

# Ensure all columns are numeric
for col in X.columns:
    if X[col].dtype.name == "category":
        X[col] = X[col].astype(float)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)


@app.route("/", methods=["GET", "POST"])
def index():
    coeff_plot = pred_plot = None
    predicted_price = None

    if request.method == "POST":
        try:
            # Collect feature inputs from form
            input_data = []
            for feat in feature_names:
                val = float(request.form.get(feat, 0))
                input_data.append(val)

            input_array = np.array(input_data).reshape(1, -1)

            # Predict price
            predicted_price = model.predict(input_array)[0]
            predicted_price = round(float(predicted_price), 2)

            # --- Plot 1: Feature Coefficients ---
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.barh(
                feature_names,
                model.coef_,
                color=["orange" if c > 0 else "blue" for c in model.coef_]
            )
            ax.set_title("Feature Coefficients (Linear Regression)")
            ax.set_xlabel("Coefficient Value")
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            coeff_plot = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

            # --- Plot 2: Actual vs Predicted Prices ---
            y_pred_test = model.predict(X_test)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.scatter(y_test, y_pred_test, alpha=0.6, color="teal", label="Test Data")
            ax.scatter([predicted_price], [predicted_price],
                       color="red", s=100, label="Your Prediction")
            ax.plot([y_test.min(), y_test.max()],
                    [y_test.min(), y_test.max()],
                    "r--", lw=2, label="Perfect Fit")

            ax.set_xlabel("Actual Price (in Lakhs)")
            ax.set_ylabel("Predicted Price (in Lakhs)")
            ax.set_title("Actual vs Predicted Prices")
            ax.legend()
            buf = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buf, format="png")
            buf.seek(0)
            pred_plot = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

        except Exception as e:
            print("Error:", e)

    return render_template(
        "boston.html",
        coeff_plot=coeff_plot,
        pred_plot=pred_plot,
        predicted_price=predicted_price,
        features=feature_names
    )


if __name__ == "__main__":  # ✅ FIXED here
    app.run(debug=True)
