import os
import uvicorn
import face_recognition
import pickle
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
from PIL import Image
from io import BytesIO
import numpy as np
from typing import Optional


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_DIR = "images"
ENCODING_FILE = "image_encodings.pkl"

# Load or generate face encodings
def load_image_encodings():
    encodings_data = []
    if os.path.exists(ENCODING_FILE):
        with open(ENCODING_FILE, "rb") as f:
            encodings_data = pickle.load(f)
        print("[INFO] Loaded encodings from file.")
    else:
        print("[INFO] Generating encodings...")
        for filename in os.listdir(IMAGE_DIR):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(IMAGE_DIR, filename)
                image = face_recognition.load_image_file(path)
                face_locations = face_recognition.face_locations(image)
                encodings = face_recognition.face_encodings(image, face_locations)
                for encoding in encodings:
                    encodings_data.append({
                        "filename": filename,
                        "encoding": encoding
                    })
        with open(ENCODING_FILE, "wb") as f:
            pickle.dump(encodings_data, f)
        print("[INFO] Encodings generated and saved.")
    return encodings_data

# Load on startup
all_encodings = load_image_encodings()

@app.post("/upload-photo")
async def upload_photo(
    name: Optional[str] = Form(None),
    image: UploadFile = File(...),
    tolerance: Optional[float] = Form(0.6)
):
    image_bytes = await image.read()
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(pil_image)

    try:
        face_locations = face_recognition.face_locations(np_image)
        uploaded_encodings = face_recognition.face_encodings(np_image, face_locations)
        if not uploaded_encodings:
            return JSONResponse(status_code=400, content={"error": "No face detected in uploaded image."})
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Face encoding failed: {e}"})

    uploaded_encoding = uploaded_encodings[0]
    matched_files = []

    for data in all_encodings:
        result = face_recognition.compare_faces([data["encoding"]], uploaded_encoding, tolerance=tolerance)
        distance = face_recognition.face_distance([data["encoding"]], uploaded_encoding)[0]
        if result[0]:
            matched_files.append({
                "filename": data["filename"],
                "distance": round(float(distance), 4),
                "url": f"/images/{data['filename']}"
            })

    return {
        "name": name,
        "matched": matched_files,
        "count": len(matched_files)
    }


# Serve static image files
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, reload=True)
