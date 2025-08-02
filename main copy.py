import os
import shutil
import csv
import uuid
import face_recognition
import requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile
from typing import List

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
IMAGES_DIR = "images"
OUTPUT_DIR = "output"
TEMP_GUEST_DIR = "temp_guests"
MATCH_THRESHOLD = 0.6
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_GUEST_DIR, exist_ok=True)


def find_matches(guest_name: str, guest_image_path: str) -> List[str]:
    matched_files = []
    try:
        guest_image = face_recognition.load_image_file(guest_image_path)
        guest_encodings = face_recognition.face_encodings(guest_image)

        if not guest_encodings:
            return []

        guest_encoding = guest_encodings[0]

        for image_file in os.listdir(IMAGES_DIR):
            full_path = os.path.join(IMAGES_DIR, image_file)
            image = face_recognition.load_image_file(full_path)
            encodings = face_recognition.face_encodings(image)

            matches = face_recognition.compare_faces(encodings, guest_encoding, tolerance=MATCH_THRESHOLD)
            if any(matches):
                output_dir = os.path.join(OUTPUT_DIR, guest_name)
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy(full_path, output_dir)
                matched_files.append(image_file)
    except Exception as e:
        print(f"Error matching guest {guest_name}: {e}")
    return matched_files


@app.post("/upload-photo")
async def upload_photo(name: str = Form(...), file: UploadFile = File(...)):
    try:
        temp_path = os.path.join(TEMP_GUEST_DIR, f"{uuid.uuid4()}.jpg")
        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        matched = find_matches(name, temp_path)
        os.remove(temp_path)
        return {"guest": name, "matched_files": matched}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/upload-csv")
async def upload_csv(csv_file: UploadFile = File(...)):
    try:
        csv_path = os.path.join(TEMP_GUEST_DIR, "uploaded.csv")
        with open(csv_path, "wb") as f:
            content = await csv_file.read()
            f.write(content)

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name'].strip()
                url = row['image_url'].strip()
                guest_img_path = os.path.join(TEMP_GUEST_DIR, f"{uuid.uuid4()}.jpg")

                # Download image
                res = requests.get(url, timeout=10)
                if res.status_code == 200:
                    with open(guest_img_path, "wb") as img:
                        img.write(res.content)
                    find_matches(name, guest_img_path)
                    os.remove(guest_img_path)

        os.remove(csv_path)
        return {"status": "success", "message": "CSV processed"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
