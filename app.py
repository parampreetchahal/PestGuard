from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import config

from concurrent.futures import ThreadPoolExecutor
import json
import config
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable
import requests
import time
from supabase import create_client
import config

supabase = create_client(
    config.SUPABASE_URL,
    config.SUPABASE_KEY
)
# Deep learning model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model once
model = load_model("best_model2.h5")

# Load class indices from training
with open("class_labels.json", "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> class name
pest_classes = {v: k for k, v in class_indices.items()}


# -------------------------------
# -------------------------------

#DATABASE CONNECTIVE
def get_pest_info(pest_name):
    """
    Fetch pest info from Supabase (PostgreSQL)
    Table: pestguard
    Columns: pest_name, effects, natural_solution, chemical_solution
    """
    default = {
        "effects": "Unknown",
        "natural_solution": "Unknown",
        "pesticide_solution": "Unknown"
    }

    if not pest_name:
        return default

    try:
        response = (
            supabase
            .table("pestguard")
            .select("pest_name,effects,natural_solution,chemical_solution")
            .ilike("pest_name", pest_name)
            .limit(1)
            .execute()
        )

        if response.data:
            row = response.data[0]
            return {
                "effects": row.get("effects", "Unknown"),
                "natural_solution": row.get("natural_solution", "Unknown"),
                # 🔁 map chemical_solution → pesticide_solution (your app expects this key)
                "pesticide_solution": row.get("chemical_solution", "Unknown")
            }

        return default

    except Exception as e:
        print("❌ Supabase error in get_pest_info:", e)
        return {
            "effects": "Error",
            "natural_solution": "N/A",
            "pesticide_solution": "N/A"
        }


# Cache coordinates per city (reduces repeated requests)

_COORDINATE_CACHE = {}

def get_coordinates(city_name, retries=3, timeout=10):
    """Return (lat, lon) for a city using Nominatim with Photon fallback."""
    if not city_name:
        return None, None

    key = city_name.strip().lower()
    if key in _COORDINATE_CACHE:
        return _COORDINATE_CACHE[key]

    # --- Try Nominatim ---
    geolocator = Nominatim(user_agent="pestguard_app (contact: your_email@example.com)", timeout=timeout)
    for attempt in range(1, retries + 1):
        try:
            location = geolocator.geocode(city_name, exactly_one=True)
            if location:
                lat, lon = location.latitude, location.longitude
                _COORDINATE_CACHE[key] = (lat, lon)
                return lat, lon
        except (GeocoderTimedOut, GeocoderServiceError, GeocoderUnavailable) as e:
            print(f"Nominatim error (attempt {attempt}): {e}")
            time.sleep(1.5 * attempt)
        except Exception as e:
            print(f"Unexpected geocode error: {e}")
            break

    # --- Fallback to Photon API ---
    try:
        url = f"https://photon.komoot.io/api/?q={city_name}&limit=1"
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            if data.get("features"):
                coords = data["features"][0]["geometry"]["coordinates"]
                lon, lat = coords[0], coords[1]
                _COORDINATE_CACHE[key] = (lat, lon)
                return lat, lon
    except Exception as e:
        print("Photon fallback error:", e)

    return None, None

def get_nearby_stores(city_name, radius=15000, max_results=5):
    from config import GOOGLE_MAPS_API_KEY

    lat, lon = get_coordinates(city_name)
    if not lat or not lon:
        print(f"⚠️ Could not fetch coordinates for {city_name}")
        return []

    # ✅ Google Places API (Correct Format)
    if GOOGLE_MAPS_API_KEY and GOOGLE_MAPS_API_KEY != "DUMMY_KEY":
        google_url = (
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            f"?location={lat},{lon}"
            f"&radius={radius}"
            f"&keyword=pesticide%20fertilizer%20agro%20farm%20consultant"
            f"&key={GOOGLE_MAPS_API_KEY}"
        )

        try:
            resp = requests.get(google_url, timeout=10)
            data = resp.json()

            if data.get("status") == "OK":
                stores = []
                for place in data["results"][:max_results]:
                    name = place.get("name", "Store")
                    address = place.get("vicinity", "Address not available")

                    plat = place["geometry"]["location"]["lat"]
                    plon = place["geometry"]["location"]["lng"]

                    maps_url = f"https://www.google.com/maps/dir/?api=1&destination={plat},{plon}"

                    stores.append({
                        "name": name,
                        "address": address,
                        "maps_url": maps_url,
                    })

                return stores
            else:
                print("⚠️ Google Error:", data.get("status"), data.get("error_message"))
        except Exception as e:
            print("❌ Google API error:", e)

    # ✅ Fallback to Photon
    try:
        url = f"https://photon.komoot.io/api/?q={city_name}%20fertilizer%20shop&limit={max_results}"
        resp = requests.get(url, timeout=10)
        data = resp.json()

        stores = []
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            name = props.get("name") or "Local Agro Store"
            city = props.get("city") or city_name
            street = props.get("street", "")
            address = f"{street}, {city}".strip(", ")

            coords = feature["geometry"]["coordinates"]
            maps_url = f"https://www.google.com/maps/dir/?api=1&destination={coords[1]},{coords[0]}"

            stores.append({
                "name": name,
                "address": address,
                "maps_url": maps_url
            })

        return stores

    except Exception as e:
        print("❌ Photon error:", e)
        return []

# -------------------------------
# Preprocess image
# -------------------------------
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((244, 244))  # trained size
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -------------------------------
# Thread pool for async predictions
# -------------------------------
executor = ThreadPoolExecutor(max_workers=2)


def predict_image(image_path, threshold=50):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    pest_index = np.argmax(prediction, axis=1)[0]
    confidence = round(np.max(prediction) * 100, 2)

    if confidence < threshold:  # Low confidence → No pest
        return {
            "filename": os.path.basename(image_path),
            "pest": "No pest found",
            "confidence": confidence,
            "effects": "N/A",
            "natural_solution": "N/A",
            "pesticide_solution": "N/A",
            "stores": []
        }

    detected_pest = pest_classes[pest_index]
    pest_info = get_pest_info(detected_pest)

    return {
        "filename": os.path.basename(image_path),
        "pest": detected_pest,
        "confidence": confidence,
        "effects": pest_info.get("effects", "Unknown"),
        "natural_solution": pest_info.get("natural_solution", "Unknown"),
        "pesticide_solution": pest_info.get("pesticide_solution", "Unknown"),
        "stores": []  # Stores will be added in /detect
    }


# -------------------------------
# AI Info Fetch Function
# -------------------------------
def getinfo(api_key, detected_pest):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled in fetching practical pest details for farmers."},
        {"role": "user", "content": f"Give details about this pest: {detected_pest}. Include life cycle, effects, and farmer solutions. Provide only useful info without AI-style filler text."}
    ]
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.7
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        print("❌ API Error:", response.status_code, response.text)
        response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


# -------------------------------
# Flask Routes
# -------------------------------
@app.route('/')
def home():
    return render_template("home.html")


@app.route('/detect', methods=["GET", "POST"])
def detect():
    if request.method == "POST":
        city_name = request.form.get("shoplocation")
        if not city_name:
            city_name = "Delhi"  # fallback

        files = request.files.getlist("images")
        results = []

        futures = []
        for file in files:
            if file.filename == "":
                continue
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            futures.append(executor.submit(predict_image, filepath))

        for future in futures:
            res = future.result()
            if res["pest"] != "No pest found":
                res["stores"] = get_nearby_stores(city_name)
            results.append(res)

        return render_template("result.html", results=results)

    return render_template("detect.html")


@app.route('/result')
def result():
    return redirect(url_for("detect"))


@app.route('/more_info', methods=['POST'])
def more_info():
    api_key = config.GROQ_API_KEY
    data = request.get_json()
    detected_pest = data.get("pest")

    if not detected_pest or not api_key:
        return jsonify({"error": "Missing pest name or API key"}), 400

    try:
        info = getinfo(api_key, detected_pest)
        return jsonify({"info": info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# Run the Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
