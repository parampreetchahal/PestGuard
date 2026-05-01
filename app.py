from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
from flask_mail import Mail, Message
from flask_limiter import Limiter
from flask_compress import Compress
from werkzeug.exceptions import HTTPException

from uuid import uuid4
from PIL import Image
import numpy as np

import json
import logging

from supabase import create_client
import config

import time
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from keras.models import load_model

from datetime import datetime
from scipy.stats import entropy
import re

from keras.applications.efficientnet import preprocess_input
from threading import Lock
from upstash_redis import Redis

import io
import base64

geo_lock = Lock()
model_lock = Lock()

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
limiter = Limiter(app)
Compress(app)

logging.basicConfig(level=logging.INFO)

app.config['SECRET_KEY'] = config.SECRET_KEY

app.config['MAIL_SERVER'] = config.MAIL_SERVER
app.config['MAIL_PORT'] = config.MAIL_PORT
app.config['MAIL_USE_TLS'] = config.MAIL_USE_TLS
app.config['MAIL_USERNAME'] = config.MAIL_USERNAME
app.config['MAIL_PASSWORD'] = config.MAIL_PASSWORD

mail = Mail(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB limit


# Load trained model once
model = load_model("./best_pestnet.keras")


def warmup_model():
    dummy_input = np.zeros((1,300,300,3))
    model.predict(dummy_input,verbose=1)


warmup_model()

RESULT_CACHE = {}

# Load class indices from training
with open("class_names.json", "r") as f:
    pest_classes = json.load(f)


def safe_request(method, url, **kwargs):
    if "timeout" in kwargs:
        del kwargs["timeout"]

    timeout = 10

    for _ in range(3):
        try:
            return requests.request(method, url, timeout=timeout, **kwargs)
        except requests.exceptions.RequestException as e:
            logging.info(f"Retrying request...", {e})
            time.sleep(0.5)

    return None


# DATABASE CONNECTIVE
def get_pest_info(pest_name, lang="en"):

    default = {
        "effects": "Unknown",
        "natural_solution": "Unknown",
        "pesticide_solution": "Unknown"
    }

    if not pest_name:
        return default

    pest_name = pest_name.strip().lower()

    # ✅ Redis key WITH language
    cache_key = f"pest:{pest_name}:{lang}"

    # 🔥 REDIS READ
    cached = redis.get(cache_key)
    if cached:
        logging.info(f"⚡ Redis HIT: {cache_key}")

        if isinstance(cached, bytes):
            cached = cached.decode("utf-8")

        return json.loads(cached)

    try:
        # ✅ SAFE FETCH (no ilike crash)
        pest_res = supabase.table("pestguard").select("id,pest_name").execute()

        pest = next(
            (p for p in pest_res.data if pest_name in p["pest_name"].lower()),
            None
        )

        if not pest:
            return default

        pest_id = pest["id"]

        # ✅ GET TRANSLATION
        trans_res = (
            supabase
            .table("pestguard_translations")
            .select("effects,natural_solution,chemical_solution")
            .eq("pest_id", pest_id)
            .eq("lang", lang)
            .limit(1)
            .execute()
        )

        # 🔁 FALLBACK → EN
        if not trans_res.data:
            trans_res = (
                supabase
                .table("pestguard_translations")
                .select("effects,natural_solution,chemical_solution")
                .eq("pest_id", pest_id)
                .eq("lang", "en")
                .limit(1)
                .execute()
            )

        if trans_res.data:
            row = trans_res.data[0]

            result = {
                "effects": row.get("effects", "Unknown"),
                "natural_solution": row.get("natural_solution", "Unknown"),
                "pesticide_solution": row.get("chemical_solution", "Unknown")
            }

            # 🔥 REDIS WRITE
            redis.set(cache_key, json.dumps(result), ex=86400)

            logging.info(f"💾 Redis SET: {cache_key}")

            return result

        return default

    except Exception as e:
        logging.info(f"DB error: {e}")
        return default

def is_valid_city(city_name):
    if not city_name:
        return False

    city_name = city_name.strip()

    # Must contain letters
    if not re.search(r"[a-zA-Z]", city_name):
        return False

    # Reject symbols-only / garbage
    if re.fullmatch(r"[^a-zA-Z\s\-.]+", city_name):
        return False

    # Minimum length
    if len(city_name) < 3:
        return False

    return True


# Cache coordinates per city (reduces repeated requests)
# Simple in-memory cache to avoid repeated API calls
_COORDINATE_CACHE = {}
_STORE_CACHE = {}
_FAILED_CITY_CACHE = {}  # city -> timestamp
FAIL_TIMEOUT = 300  # 5 minutes retry window


def get_coordinates(city_name):
    """
    Robust coordinate lookup:
    1) Photon API (FAST, FREE, STABLE)
    2) Nominatim fallback (RATE LIMITED)
    """
    if not is_valid_city(city_name):
        logging.info(f"Invalid city input: {city_name}")
        return None, None

    key = city_name.strip().lower()
    if key in _COORDINATE_CACHE:
        return _COORDINATE_CACHE[key]

    # -------------------------------------------------
    # 1️⃣ PHOTON FIRST (Highly recommended)
    # -------------------------------------------------
    with geo_lock:
        try:
            # query=f"{city_name}, India"
            photon_url = f"https://photon.komoot.io/api/?q={city_name}&limit=1"
            r = safe_request("GET",photon_url)

            if r.status_code == 200:
                data = r.json()
                if data.get("features"):
                    lon, lat = data["features"][0]["geometry"]["coordinates"]
                    _COORDINATE_CACHE[key] = (lat, lon)
                    return lat, lon
        except Exception as e:
            logging.info(f"Photon error:", {e})

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
            logging.info(f"Nominatim error:", {e})
        except Exception as e:
            logging.info(f"Unexpected geocoding error:", {e})

        # -------------------------------------------------
        # FAILED
        # -------------------------------------------------
        logging.info(f"Could not determine coordinates for: {city_name}")
        return None, None


def get_nearby_stores(city_name, radius=15000, max_results=5):
    if not city_name:
        return []

    key = city_name.strip().lower()

    # ===============================
    # 1️⃣ SUCCESS CACHE (NON-EMPTY ONLY)
    # ===============================
    if key in _STORE_CACHE and _STORE_CACHE[key]:
        logging.info("Using cached store results")
        return _STORE_CACHE[key]

    # ===============================
    # 2️⃣ FAILURE / CIRCUIT BREAKER
    # ===============================
    now = time.time()

    if key in _FAILED_CITY_CACHE:
        if now - _FAILED_CITY_CACHE[key] < FAIL_TIMEOUT:
            logging.info("Skipping APIs (temporary failure cache)")
            return [{
                "name": "Search nearby pesticide shops",
                "address": city_name,
                "maps_url": f"https://www.google.com/maps/search/pesticide+shop+near+{city_name}"
            }]
        else:
            del _FAILED_CITY_CACHE[key]  # retry after timeout

    # ===============================
    # GET COORDINATES
    # ===============================
    lat, lon = get_coordinates(city_name)
    logging.info(f"📍 City: {city_name}, Lat: {lat}, Lon: {lon}")
    if not lat or not lon:
        _FAILED_CITY_CACHE[key] = time.time()
        return [{
            "name": "Search nearby pesticide shops",
            "address": city_name,
            "maps_url": f"https://www.google.com/maps/search/pesticide+shop+near+{city_name}"
        }]

    stores = []

    # ===============================
    # OVERPASS API
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
        time.sleep(0.2)  # rate safety
        headers = {
            "User-Agent": "PestGuard/1.0 (student project)"
        }
        resp = safe_request("POST", overpass_url, data=query, headers=headers)

        if not resp or resp.status_code != 200 or not resp.text.strip():
            raise Exception("API failed")

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
            logging.info("Stores cached from Overpass")
            return stores

    except Exception as e:
        logging.info(f"Overpass error: %s", {e})

    # ===============================
    # PHOTON FALLBACK
    # ===============================
    try:
        url = f"https://photon.komoot.io/api/?q=agriculture%20shop%20{city_name}%20India&limit={max_results}"
        resp = safe_request("GET",url)

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
            logging.info("Stores cached from Photon")
            return stores

    except Exception as e:
        logging.info(f"Photon fallback failed: %s", {e})

    # ===============================
    # MARK CITY FAILED (STOP RETRIES)
    # ===============================
    _FAILED_CITY_CACHE[key] = time.time()
    logging.info("Marking city as failed, using hard fallback")

    return [{
        "name": "Search nearby pesticide shops",
        "address": city_name,
        "maps_url": f"https://www.google.com/maps/search/pesticide+shop+near+{city_name}"
    }]


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

        for attempt in range(3):
            try:
                supabase.storage.from_(LOW_CONF_BUCKET).upload(
                    storage_path,
                    image_bytes,
                    {"content-type": "image/jpeg"}
                )
                logging.info(f"Upload success: {storage_path}")
                break
            except Exception as e:
                logging.info(f"Upload completely failed:", {storage_path})
                return

        # INSERT INTO DB ONLY AFTER SUCCESS
        db_res = supabase.table(LOW_CONF_TABLE).insert({
            "image_path": storage_path,
            "confidence": confidence,
            "predicted_pest": None
        }).execute()

        # CHECK DB RESPONSE
        if hasattr(db_res, "error") and db_res.error:
            logging.info(f"DB insert failed:", {db_res.error})
        else:
            logging.info(f"📥 Stored in DB + Storage: {storage_path}")

    except Exception as e:
        logging.info(f"Failed to store low-confidence image:", {e})


def predict_image_from_pil(pil_image, image_bytes, lang="en", threshold=78):

    img_array = preprocess_image_from_pil(pil_image)
    with model_lock:
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

    #  LOW CONFIDENCE → STORE IMAGE
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

    # CONFIDENT PREDICTION
    pest_info = get_pest_info(detected_pest, lang)

    return {
        "pest": detected_pest,
        "confidence": confidence,
        "effects": pest_info.get("effects", "Unknown"),
        "natural_solution": pest_info.get("natural_solution", "Unknown"),
        "pesticide_solution": pest_info.get("pesticide_solution", "Unknown"),
        "stores": []
    }

# AI Info Fetch Function


def getinfo(api_key, detected_pest, lang="en"):

    language_map = {
        "en": "English",
        "hi": "Hindi"
    }

    selected_lang = language_map.get(lang, "English")

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": (
                f"You are an expert agricultural entomologist.\n"
                f"Respond STRICTLY in {selected_lang}.\n"
                "Do NOT mix languages.\n"
                "Keep headings and content in the same language.\n"
                "Use Markdown with headers (###), bold keys, bullet points."
            )
        },
        {
            "role": "user",
            "content": (
                f"Create a technical dossier for the pest: {detected_pest}.\n\n"

                f"IMPORTANT:\n"
                f"- Output MUST be completely in {selected_lang}\n"
                f"- Do NOT use English if language is Hindi\n\n"

                "Structure:\n"
                "###  Scientific Classification\n"
                "**Scientific Name**: [Name]\n"
                "**Host Crops**: [List]\n\n"
                "###  Life Cycle\n"
                "* **Egg Stage**: [Details]\n"
                "* **Larval Stage**: [Details]\n"
                "* **Pupal/Adult Stages**: [Details]\n\n"
                "###  Agricultural Impact\n"
                "* **Yield Loss**: [Details]\n"
                "* **Physical Damage**: [Details]\n\n"
                "###  Management Strategies\n"
                "* **Cultural Control**: [Steps]\n"
                "* **Biological Control**: [Solutions]"
            )
        }
    ]

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
        "temperature": 0.2
    }

    response = safe_request("POST", url, headers=headers, json=payload)
    data = response.json()

    return data["choices"][0]["message"]["content"]

# Flask Routes


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/detect', methods=["GET", "POST"])
@limiter.limit("5 per minute")
def detect():
    lang = request.form.get("lang", "en")
    if request.method == "POST":
        city_name = request.form.get("shoplocation", "").strip() or None
        files = request.files.getlist("images")

        results = []

        for file in files:
            if file.filename == "":
                continue

            image_bytes = file.read()
            try:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            except:
                continue

            res = predict_image_from_pil(pil_image, image_bytes, lang=lang)

            encoded_img = base64.b64encode(image_bytes).decode("utf-8")
            res["image_data"] = encoded_img

            if res["pest"] != "No pest found" and city_name:
                res["stores"] = get_nearby_stores(city_name)
            else:
                res["stores"] = []

            results.append(res)

        # store results in memory (not session)
        result_id = str(uuid4())
        RESULT_CACHE[result_id] = results

        # redirect instead of render
        return redirect(url_for("result", rid=result_id))

    return render_template("detect.html")


@app.route('/result')
def result():
    rid = request.args.get("rid")

    if not rid or rid not in RESULT_CACHE:
        return redirect(url_for("detect"))

    results = RESULT_CACHE.get(rid)  # remove after use

    return render_template("result.html", results=results)


@app.route('/more_info', methods=['POST'])
def more_info():
    api_key = config.GROQ_API_KEY
    data = request.get_json()
    detected_pest = data.get("pest")
    lang = data.get("lang", "en")

    if not detected_pest or not api_key:
        return jsonify({"error": "Missing pest name or API key"}), 400

    try:
        info = getinfo(api_key, detected_pest, lang)
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
        logging.info(f"DEBUG: Form received! Name: {request.form.get('name')}")

        user_name = request.form.get('name')
        user_email = request.form.get('email')
        user_subject = request.form.get('subject')
        user_msg = request.form.get('message')

        try:
            # EMAIL 1: Notification to YOU
            msg_to_self = Message(
                subject=f"New PestGuard Inquiry: {user_subject}",
                sender=app.config['MAIL_USERNAME'],
                recipients=['pestguardteam@gmail.com'],
                body=f"Hello Team PestGurad,\n\nYou have a new message from {user_name} ({user_email}):\n\n{user_msg}"
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

            flash("Message sent! Check your inbox for a confirmation. ", "success")
        except Exception as e:
            logging.info(f"Mail Error: {e}")
            flash("System busy. Please try again later.", "error")

        return redirect(url_for('contact'))

    return render_template("contact.html")


@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors (like 404s) so Flask handles them normally
    if isinstance(e, HTTPException):
        return e

    # Log the actual traceback for real server errors (500)
    logging.exception("UNHANDLED EXCEPTION")
    return jsonify({
        "error": "Something went wrong. Please try again later."
    }), 500


@app.after_request
def add_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route('/favicon.ico')
def favicon():
    return '', 204

# Run the Flask App


if __name__ == "__main__":
    app.run(use_reloader=False)
