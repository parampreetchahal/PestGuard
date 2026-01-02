import os
from flask import Flask, render_template, request, redirect, url_for, jsonify

from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from concurrent.futures import ThreadPoolExecutor
import json

from supabase import create_client
import config

import time
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from keras.models import load_model
# Deep learning model
#from keras.applications.efficientnet_v2 import preprocess_input
import tensorflow as tf
from keras.applications.efficientnet import preprocess_input

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

supabase = create_client(
    config.SUPABASE_URL,
    config.SUPABASE_KEY
)

app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model once
model = load_model("best_pestnet.keras")

# Load class indices from training
with open("class_names.json", "r") as f:
    pest_classes = json.load(f)



# -------------------------------
# -------------------------------

# DATABASE CONNECTIVE

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

# Simple in-memory cache to avoid repeated API calls
_COORDINATE_CACHE = {}
_STORE_CACHE = {}
_FAILED_CITY_CACHE = set()


def get_coordinates(city_name):
    """
    Robust coordinate lookup:
    1) Photon API (FAST, FREE, STABLE)
    2) Nominatim fallback (RATE LIMITED)
    """

    if not city_name:
        return None, None

    key = city_name.strip().lower()
    if key in _COORDINATE_CACHE:
        return _COORDINATE_CACHE[key]

    # -------------------------------------------------
    # 1️⃣ PHOTON FIRST (Highly recommended)
    # -------------------------------------------------
    try:
        photon_url = f"https://photon.komoot.io/api/?q={city_name}&limit=1"
        r = requests.get(photon_url, timeout=10)

        if r.status_code == 200:
            data = r.json()
            if data.get("features"):
                lon, lat = data["features"][0]["geometry"]["coordinates"]
                _COORDINATE_CACHE[key] = (lat, lon)
                return lat, lon
    except Exception as e:
        print("⚠️ Photon error:", e)

    # Small delay before fallback (important)
    time.sleep(1.2)

    # -------------------------------------------------
    # NOMINATIM FALLBACK (STRICT RATE LIMIT)
    # -------------------------------------------------
    try:
        geolocator = Nominatim(
            user_agent="pestguard-app/1.0 (student project, India)",
            timeout=10
        )

        location = geolocator.geocode(
            city_name,
            exactly_one=True
        )

        if location:
            lat, lon = location.latitude, location.longitude
            _COORDINATE_CACHE[key] = (lat, lon)
            return lat, lon

    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print("⚠️ Nominatim error:", e)
    except Exception as e:
        print("⚠️ Unexpected geocoding error:", e)

    # -------------------------------------------------
    # FAILED
    # -------------------------------------------------
    print(f"❌ Could not determine coordinates for: {city_name}")
    return None, None


def get_nearby_stores(city_name, radius=15000, max_results=5):
    if not city_name:
        return []

    key = city_name.strip().lower()

    # ===============================
    # 1️⃣ SUCCESS CACHE (NON-EMPTY ONLY)
    # ===============================
    if key in _STORE_CACHE and _STORE_CACHE[key]:
        print("✅ Using cached store results")
        return _STORE_CACHE[key]

    # ===============================
    # 2️⃣ FAILURE / CIRCUIT BREAKER
    # ===============================
    if key in _FAILED_CITY_CACHE:
        print("⛔ Skipping APIs (previous failure)")
        return [{
            "name": "Search nearby pesticide shops",
            "address": city_name,
            "maps_url": f"https://www.google.com/maps/search/pesticide+shop+near+{city_name}"
        }]

    # ===============================
    # 3️⃣ GET COORDINATES
    # ===============================
    lat, lon = get_coordinates(city_name)
    if not lat or not lon:
        _FAILED_CITY_CACHE.add(key)
        return [{
            "name": "Search nearby pesticide shops",
            "address": city_name,
            "maps_url": f"https://www.google.com/maps/search/pesticide+shop+near+{city_name}"
        }]

    stores = []

    # ===============================
    # 4️⃣ OVERPASS API
    # ===============================
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["shop"~"agrovet|farm|garden|agricultural"](around:{radius},{lat},{lon});
      node["amenity"="pharmacy"](around:{radius},{lat},{lon});
    );
    out tags center;
    """

    try:
        time.sleep(1.5)  # rate safety
        resp = requests.post(overpass_url, data=query, timeout=20)

        if resp.status_code != 200 or not resp.text.strip():
            raise Exception("Overpass blocked or empty")

        data = resp.json()

        for el in data.get("elements", [])[:max_results]:
            tags = el.get("tags", {})
            stores.append({
                "name": tags.get("name", "Local Agro Store"),
                "address": tags.get("addr:city", city_name),
                "maps_url": f"https://www.google.com/maps/dir/?api=1&destination={el['lat']},{el['lon']}"
            })

        if stores:
            _STORE_CACHE[key] = stores
            print("✅ Stores cached from Overpass")
            return stores

    except Exception as e:
        print("❌ Overpass error:", e)

    # ===============================
    # 5️⃣ PHOTON FALLBACK
    # ===============================
    try:
        url = f"https://photon.komoot.io/api/?q=pesticide%20shop%20{city_name}&limit={max_results}"
        resp = requests.get(url, timeout=10)

        if resp.status_code != 200 or not resp.text.strip():
            raise Exception("Photon empty")

        data = resp.json()

        for feature in data.get("features", []):
            props = feature.get("properties", {})
            coords = feature["geometry"]["coordinates"]
            stores.append({
                "name": props.get("name", "Local Agro Store"),
                "address": props.get("city", city_name),
                "maps_url": f"https://www.google.com/maps/dir/?api=1&destination={coords[1]},{coords[0]}"
            })

        if stores:
            _STORE_CACHE[key] = stores
            print("✅ Stores cached from Photon")
            return stores

    except Exception as e:
        print("❌ Photon fallback failed:", e)

    # ===============================
    # 6️⃣ MARK CITY FAILED (STOP RETRIES)
    # ===============================
    _FAILED_CITY_CACHE.add(key)
    print("⛔ Marking city as failed, using hard fallback")

    return [{
        "name": "Search nearby pesticide shops",
        "address": city_name,
        "maps_url": f"https://www.google.com/maps/search/pesticide+shop+near+{city_name}"
    }]

# -------------------------------
# Preprocess image
# -------------------------------


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((300, 300))  # trained size
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# -------------------------------
# Thread pool for async predictions
# -------------------------------
executor = ThreadPoolExecutor(max_workers=2)


def predict_image(image_path, threshold=75):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)
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
        {"role": "system", "content": "You are a helpful assistant skilled in "
                                      "fetching practical pest details for farmers."},
        {"role": "user", "content": f"Give details about this pest: {detected_pest}. Include life cycle, effects, and farmer solutions."
                                    "Provide only useful info without AI-style filler text."}
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
        city_name = request.form.get("shoplocation", "").strip()

        if not city_name:
            city_name = None

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

            if res["pest"] != "No pest found" and city_name:
                res["stores"] = get_nearby_stores(city_name)
            else:
                res["stores"] = []

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
