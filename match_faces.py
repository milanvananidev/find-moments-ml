import os
import shutil
import face_recognition

# ===== Configuration =====
GUESTS_DIR = "guests"
IMAGES_DIR = "images"
OUTPUT_DIR = "output"
MATCH_THRESHOLD = 0.6  # Lower = stricter (default: 0.6)

# ===== Step 1: Load guest reference faces =====
guest_encodings = {}

print("🔍 Loading guest reference images...")
for guest_file in os.listdir(GUESTS_DIR):
    guest_path = os.path.join(GUESTS_DIR, guest_file)
    guest_name = os.path.splitext(guest_file)[0]

    try:
        image = face_recognition.load_image_file(guest_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            guest_encodings[guest_name] = encodings[0]
            print(f"✅ Loaded: {guest_name}")
        else:
            print(f"⚠️ No face found in {guest_file}")
    except Exception as e:
        print(f"❌ Error loading {guest_file}: {e}")

# ===== Step 2: Match guests in event photos =====
print("\n📸 Processing event images...\n")

for image_file in os.listdir(IMAGES_DIR):
    image_path = os.path.join(IMAGES_DIR, image_file)

    try:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)

        for guest_name, guest_encoding in guest_encodings.items():
            matches = face_recognition.compare_faces(face_encodings, guest_encoding, tolerance=MATCH_THRESHOLD)

            if any(matches):
                guest_output_dir = os.path.join(OUTPUT_DIR, guest_name)
                os.makedirs(guest_output_dir, exist_ok=True)
                shutil.copy(image_path, guest_output_dir)
                print(f"✅ Matched: {guest_name} → {image_file}")
    except Exception as e:
        print(f"❌ Error processing {image_file}: {e}")

print("\n🎉 Done! Check the 'output' folder.")
