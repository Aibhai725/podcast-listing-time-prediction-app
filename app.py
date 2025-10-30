from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# === Load Model and Encoders ===
MODEL_FILES = ["podcast.pkl", "potcast.pkl", "model.pkl"]
model = None
for fn in MODEL_FILES:
    if os.path.exists(fn):
        with open(fn, "rb") as f:
            model = pickle.load(f)
        break
if model is None:
    raise FileNotFoundError("Model file not found. Please keep your model as 'podcast.pkl' or 'potcast.pkl'.")

# Load LabelEncoder(s)
with open("label_encoder.pkl", "rb") as f:
    encoders = pickle.load(f)

# Detect whether it's a single encoder or multiple
if hasattr(encoders, "transform"):
    # Single encoder → make it multi-column compatible
    encoders = {"Episode_Sentiment": encoders}

# Load OneHotEncoder
with open("onehot_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

GENRE_COLUMNS = [
    "Genre_Business", "Genre_Comedy", "Genre_Education", "Genre_Health",
    "Genre_Lifestyle", "Genre_Music", "Genre_News", "Genre_Sports",
    "Genre_Technology", "Genre_True Crime"
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # === Collect Inputs ===
        Episode_Length_minutes = float(request.form.get("Episode_Length_minutes", 0))
        Host_Popularity_percentage = float(request.form.get("Host_Popularity_percentage", 0))
        Guest_Popularity_percentage = float(request.form.get("Guest_Popularity_percentage", 0))
        Number_of_Ads = int(request.form.get("Number_of_Ads", 0))
        Publication_Day = request.form.get("Publication_Day")
        Publication_Time = request.form.get("Publication_Time")
        Episode_Sentiment = request.form.get("Episode_Sentiment")
        Genre = request.form.get("Genre")

        # === Create DataFrame ===
        input_data = pd.DataFrame({
            "Episode_Length_minutes": [Episode_Length_minutes],
            "Host_Popularity_percentage": [Host_Popularity_percentage],
            "Publication_Day": [Publication_Day],
            "Publication_Time": [Publication_Time],
            "Guest_Popularity_percentage": [Guest_Popularity_percentage],
            "Number_of_Ads": [Number_of_Ads],
            "Episode_Sentiment": [Episode_Sentiment],
            "Genre": [Genre]
        })

        # === Label Encode categorical features using loaded encoders ONLY ===
        # Columns we want encoded with label encoders:
        cols_to_encode = ["Episode_Sentiment", "Publication_Time", "Publication_Day"]

        for col in cols_to_encode:
            if col not in input_data.columns:
                continue

            val = input_data.at[0, col]

            # 1) If there's a direct encoder saved for this column, use it
            if col in encoders:
                encoder = encoders[col]
                try:
                    if val in getattr(encoder, "classes_", []):
                        input_data[col] = encoder.transform([val])
                    else:
                        input_data[col] = -1
                except Exception:
                    input_data[col] = -1
                continue

            # 2) Otherwise, try to find any encoder in encoders whose classes_ contains this value
            found = False
            for enc_name, encoder in encoders.items():
                try:
                    if val in getattr(encoder, "classes_", []):
                        input_data[col] = encoder.transform([val])
                        found = True
                        break
                except Exception:
                    continue

            # 3) If not found, fall back to -1 (unseen)
            if not found:
                input_data[col] = -1

        # === OneHot Encode Genre ===
        try:
            genre_encoded = ohe.transform(input_data[["Genre"]]).toarray()
            genre_df = pd.DataFrame(
                genre_encoded, columns=ohe.get_feature_names_out(["Genre"]), dtype=int
            )
        except Exception:
            zero_row = {c: 0 for c in GENRE_COLUMNS}
            genre_col = "Genre_" + Genre.strip().title()
            if genre_col in GENRE_COLUMNS:
                zero_row[genre_col] = 1
            genre_df = pd.DataFrame([zero_row])

        # === Combine Encoded Columns ===
        X = pd.concat([input_data.drop(columns=["Genre"]), genre_df], axis=1)
        for col in GENRE_COLUMNS:
            if col not in X.columns:
                X[col] = 0

        X = X.reindex(columns=[
            "Episode_Length_minutes", "Host_Popularity_percentage",
            "Publication_Day", "Publication_Time", "Guest_Popularity_percentage",
            "Number_of_Ads", "Episode_Sentiment"
        ] + GENRE_COLUMNS, fill_value=0)

        # === Predict ===
        prediction = model.predict(X)[0]

        return render_template("index.html",
                            prediction_text=f"🎧 Predicted Listening Time: {round(float(prediction), 2)} minutes")

    except Exception as e:
        return render_template("index.html", prediction_text=f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
