import os
from tkinter import Tk, filedialog, simpledialog, messagebox
from supabase import create_client
from uuid import uuid4
from datetime import datetime

from config import SUPABASE_URL, SUPABASE_KEY

# -----------------------
# Supabase init
# -----------------------
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "low_confidence_images"
TABLE_NAME = "low_confidence_detections"

# -----------------------
# Upload helper
# -----------------------
def upload_image(image_path):
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    storage_path = f"{datetime.now().strftime('%Y/%m')}/{uuid4()}.jpg"

    supabase.storage.from_(BUCKET_NAME).upload(
        storage_path,
        image_bytes,
        {"content-type": "image/jpeg"}
    )

    return storage_path

# -----------------------
# MAIN GUI LOGIC
# -----------------------
def run_popup_test():
    root = Tk()
    root.withdraw()  # hide main window

    # Step 1: Select images
    image_files = filedialog.askopenfilenames(
        title="Select pest images (LOW confidence test)",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )

    if not image_files:
        messagebox.showerror("Error", "No images selected")
        return

    # Step 2: Enter threshold
    threshold = simpledialog.askfloat(
        "Threshold",
        "Enter confidence threshold (e.g., 0.8):",
        minvalue=0.0,
        maxvalue=1.0
    )

    if threshold is None:
        messagebox.showerror("Error", "Threshold not provided")
        return

    # Step 3: Process images
    success_count = 0

    for img_path in image_files:
        # 🔴 Mock confidence (replace with model output later)
        confidence = round(0.4 + (0.4 * os.urandom(1)[0] / 255), 2)
        predicted_pest = "unknown"

        if confidence >= threshold:
            continue

        try:
            stored_path = upload_image(img_path)

            supabase.table(TABLE_NAME).insert({
                "image_path": stored_path,
                "predicted_pest": predicted_pest,
                "confidence": confidence
            }).execute()

            success_count += 1

        except Exception as e:
            print("Error:", e)

    messagebox.showinfo(
        "Completed",
        f"{success_count} low-confidence images stored successfully."
    )

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    run_popup_test()
