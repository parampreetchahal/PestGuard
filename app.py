from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_mail import Mail, Message

from PIL import Image
import numpy as np

import json

from supabase import create_client
import config

import time
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from keras.models import load_model

from uuid import uuid4
from datetime import datetime
from scipy.stats import entropy
import re

from keras.applications.efficientnet import preprocess_input
from threading import Lock
from upstash_redis import Redis

import io
import base64

geo_lock=Lock()

LOW_CONF_BUCKET = "low_confidence_images"
LOW_CONF_TABLE = "low_confidence_detections"

supabase = create_client(
    config.SUPABASE_URL,
    config.SUPABASE_KEY
)

redis = Redis(
    url=config.UPSTASH_REDIS_REST_URL,
    token=config.UPSTASH_REDIS_REST_TOKEN
)



app = Flask(__name__)

app.secret_key = 'pestguard_super_secret_key_123'

app.config['MAIL_SERVER'] = config.MAIL_SERVER
app.config['MAIL_PORT'] = config.MAIL_PORT
app.config['MAIL_USE_TLS'] = config.MAIL_USE_TLS
app.config['MAIL_USERNAME'] = config.MAIL_USERNAME
app.config['MAIL_PASSWORD'] = config.MAIL_PASSWORD

mail = Mail(app)

app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20MB limit



# Load trained model once
model = load_model("./Models/best_pestnet.keras")
def warmup_model():
    dummy_input = np.zeros((1,300,300,3))
    model.predict(dummy_input,verbose=1)

warmup_model()

# Load class indices from training
with open("class_names.json", "r") as f:
    pest_classes = json.load(f)


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

    cache_key = f"pest:{pest_name.lower()}"

    cached = redis.get(cache_key)
    if cached:
        print("⚡⚡ Redis Hit")

        if isinstance(cached, bytes):
            cached = cached.decode("utf-8")

        return json.loads(cached)

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
            pest_result = {
                "effects": row.get("effects", "Unknown"),
                "natural_solution": row.get("natural_solution", "Unknown"),
                # 🔁 map chemical_solution → pesticide_solution (your app expects this key)
                "pesticide_solution": row.get("chemical_solution", "Unknown")
            }
            #store in redis
            redis.set(cache_key,json.dumps(pest_result),ex=86400)
            print("redis set")
            return pest_result


        return default

    except Exception as e:
        print("❌ Supabase error in get_pest_info:", e)
        return {
            "effects": "Error",
            "natural_solution": "N/A",
            "pesticide_solution": "N/A"
        }



def is_valid_city(city_name):
    if not city_name:
        return False

    city_name = city_name.strip()

    # Must contain letters
    if not re.search(r"[a-zA-Z]", city_name):
        return False

    # Reject symbols-only / garbage
    if re.fullmatch(r"[^a-zA-Z\s\-\.]+", city_name):
        return False

    # Minimum length
    if len(city_name) < 3:
        return False

    return True


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
    if not is_valid_city(city_name):
        print(f"❌ Invalid city input: {city_name}")
        return None, None

    key = city_name.strip().lower()
    if key in _COORDINATE_CACHE:
        return _COORDINATE_CACHE[key]

    # -------------------------------------------------
    # 1️⃣ PHOTON FIRST (Highly recommended)
    # -------------------------------------------------
    with geo_lock:
        try:
            #query=f"{city_name}, India"
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
          nw["shop"~"agrovet|farm|garden|agricultural"](around:{radius},{lat},{lon});
          nw["amenity"="pharmacy"](around:{radius},{lat},{lon});
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
def preprocess_image_from_pil(pil_image):
    img = pil_image.convert("RGB").resize((300, 300))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def store_low_confidence_image(image_bytes, confidence):
    """
    Stores raw image + metadata for low-confidence predictions
    (No local disk usage)
    """
    try:
        storage_path = f"{datetime.now().strftime('%Y/%m')}/{uuid4()}.jpg"

        # Upload directly from bytes
        supabase.storage.from_(LOW_CONF_BUCKET).upload(
            storage_path,
            image_bytes,
            {"content-type": "image/jpeg"}
        )

        # Insert metadata
        supabase.table(LOW_CONF_TABLE).insert({
            "image_path": storage_path,
            "confidence": confidence,
            "predicted_pest": None
        }).execute()

        print("📥 Stored low-confidence image:", storage_path)

    except Exception as e:
        print("❌ Failed to store low-confidence image:", e)

def predict_image_from_pil(pil_image, image_bytes, threshold=75):

    img_array = preprocess_image_from_pil(pil_image)
    prediction = model.predict(img_array, verbose=1)

    probs = prediction[0]

    pest_index = int(np.argmax(probs))
    detected_pest = pest_classes[pest_index]

    confidence = round(float(np.max(probs)) * 100, 2)
    uncertainty = entropy(probs)

    top2 = np.sort(probs)[-2:]
    top2_gap = top2[1] - top2[0]

    MIN_CONFIDENCE = 60
    MAX_ENTROPY = 1.5
    MIN_TOP2_GAP = 0.15

    # 🟡 LOW CONFIDENCE → STORE IMAGE
    if (
        confidence < max(threshold, MIN_CONFIDENCE)
        or uncertainty > MAX_ENTROPY
        or top2_gap < MIN_TOP2_GAP
    ):
        store_low_confidence_image(image_bytes, confidence)

        return {
            "pest": "No pest found",
            "confidence": confidence,
            "effects": "N/A",
            "natural_solution": "N/A",
            "pesticide_solution": "N/A",
            "stores": []
        }

    # 🟢 CONFIDENT PREDICTION
    pest_info = get_pest_info(detected_pest)

    return {
        "pest": detected_pest,
        "confidence": confidence,
        "effects": pest_info.get("effects", "Unknown"),
        "natural_solution": pest_info.get("natural_solution", "Unknown"),
        "pesticide_solution": pest_info.get("pesticide_solution", "Unknown"),
        "stores": []
    }
# -------------------------------
# AI Info Fetch Function
def getinfo(api_key, detected_pest):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert agricultural entomologist. Provide technical data "
                "structured with clear Markdown headers (###), bold keys, and bullet points. "
                "Do not include introductory or concluding conversational filler."
            )
        },
        {
            "role": "user",
            "content": (
                f"Create a technical dossier for the pest: {detected_pest}.\n\n"
                "Use this exact structure:\n"
                "### 🧬 Scientific Classification\n"
                "**Scientific Name**: [Name]\n"
                "**Host Crops**: [List]\n\n"
                "### 🔄 Life Cycle\n"
                "* **Egg Stage**: [Details]\n"
                "* **Larval Stage**: [Details]\n"
                "* **Pupal/Adult Stages**: [Details]\n\n"
                "### ⚠️ Agricultural Impact\n"
                "* **Yield Loss**: [Details]\n"
                "* **Physical Damage**: [Details]\n\n"
                "### 🛡️ Management Strategies\n"
                "* **Cultural Control**: [Actionable steps]\n"
                "* **Biological Control**: [Predators/solutions]"
            )
        }
    ]
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.2 # Lower temperature for more consistent formatting
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    return data["choices"][0]["message"]["content"]

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

        for file in files:
            if file.filename == "":
                continue

            image_bytes = file.read()
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except Exception:
                continue  # skip invalid image

            res = predict_image_from_pil(pil_image, image_bytes)

            encoded_img = base64.b64encode(image_bytes).decode("utf-8")
            res["image_data"] = encoded_img

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


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # ADD THIS LINE TO DEBUG:
        print(f"DEBUG: Form received! Name: {request.form.get('name')}")

        user_name = request.form.get('name')
        user_email = request.form.get('email')
        user_subject = request.form.get('subject')
        user_msg = request.form.get('message')

        try:
            # EMAIL 1: Notification to YOU
            msg_to_self = Message(
                subject=f"New PestGuard Inquiry: {user_subject}",
                sender=app.config['MAIL_USERNAME'],
                recipients=['parampreets537@gmail.com'],
                body=f"Hello Parampreet,\n\nYou have a new message from {user_name} ({user_email}):\n\n{user_msg}"
            )
            mail.send(msg_to_self)

            # EMAIL 2: Automatic Reply to the USER
            msg_to_user = Message(
                subject="We've received your message! - PestGuard AI",
                sender=app.config['MAIL_USERNAME'],
                recipients=[user_email],
                body=f"Hi {user_name},\n\nThank you for reaching out to PestGuard AI. "
                     f"We have received your inquiry regarding '{user_subject}' and our team "
                     f"will get back to you within 24-48 hours.\n\n"
                     f"In the meantime, feel free to explore our latest pest detection features.\n\n"
                     f"Best regards,\nThe PestGuard Team"
            )
            mail.send(msg_to_user)

            flash("Message sent! Check your inbox for a confirmation. 🚀", "success")
        except Exception as e:
            print(f"Mail Error: {e}")
            flash("System busy. Please try again later.", "error")

        return redirect(url_for('contact'))

    return render_template("contact.html")

# Run the Flask App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
