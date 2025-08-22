from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)  # Corrected here

# Load model and features
model = pickle.load(open("model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# Keep history
history = []

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_price = None
    if request.method == "POST":
        try:
            car_name = request.form.get("Car_Name")

            input_data = {
                "Year": float(request.form.get("Year")),
                "Present_Price": float(request.form.get("Present_Price")),
                "Kms_Driven": float(request.form.get("Kms_Driven")),
                "Owner": float(request.form.get("Owner"))
            }

            categorical_fields = {
                "Fuel_Type": request.form.get("Fuel_Type"),
                "Seller_Type": request.form.get("Seller_Type"),
                "Transmission": request.form.get("Transmission")
            }

            final_input = pd.DataFrame(np.zeros((1, len(features))), columns=features)

            for col in input_data:
                if col in final_input.columns:
                    final_input[col] = input_data[col]

            for field, value in categorical_fields.items():
                col_name = f"{field}_{value}"
                if col_name in final_input.columns:
                    final_input[col_name] = 1

            predicted_price = round(model.predict(final_input)[0])

            history.append({
                "Car_Name": car_name,
                "Year": input_data["Year"],
                "Present_Price": input_data["Present_Price"],
                "Kms_Driven": input_data["Kms_Driven"],
                "Owner": input_data["Owner"],
                "Fuel_Type": categorical_fields["Fuel_Type"],
                "Seller_Type": categorical_fields["Seller_Type"],
                "Transmission": categorical_fields["Transmission"],
                "Predicted_Price": predicted_price
            })

        except Exception as e:
            print("Error:", e)

    return render_template("car.html", predicted_price=predicted_price)


@app.route("/history")
def view_history():
    return render_template("history.html", history=history)


# ✅ Corrected this line too
if __name__ == "__main__":
    app.run(debug=True)
