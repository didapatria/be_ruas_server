from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.python.keras.models import load_model
import os

app = Flask(__name__)
CORS(app, origins="*")  # Izinkan permintaan dari domain yang berbeda

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.h5")

if os.path.exists(model_path):
    model = load_model(model_path)  # Memuat model dari file H5
else:
    raise FileNotFoundError("Model file 'model.h5' not found.")


def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    face_coordinates = [
        {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
        for (x, y, w, h) in faces
    ]
    return face_coordinates


def classify_faces(frame):
    face_coordinates = detect_faces(frame)

    for face in face_coordinates:
        x, y, w, h = face["x"], face["y"], face["width"], face["height"]
        face_img = frame[y : y + h, x : x + w]
        face_img = cv2.resize(
            face_img, (150, 150)
        )  # Ubah ukuran gambar menjadi 150x150
        face_img = np.expand_dims(face_img, axis=0)  # Tambahkan dimensi batch
        face_img = face_img / 255.0  # Normalisasi piksel

        # Klasifikasikan gambar wajah menggunakan model
        result = model.predict(face_img)
        label = np.argmax(result)
        face["label"] = label
        face["prediction"] = result.tolist()[0]

    return face_coordinates


def get_category_name(label):
    if label == 0:
        return "menyontek-lihat-atas-bawah"
    elif label == 1:
        return "menyontek-tengok-depan-belakang"
    elif label == 2:
        return "menyontek-tengok-kiri-kanan"
    elif label == 3:
        return "normal" 
    else:
        return "-"


@app.route("/process-video", methods=["POST"])
def process_video():
    try:
        # Menerima video streaming dari permintaan
        video = request.files["video"]

        # Ubah video menjadi format yang dapat digunakan oleh OpenCV
        video_array = np.frombuffer(video.read(), dtype=np.uint8)
        frame = cv2.imdecode(video_array, cv2.IMREAD_COLOR)

        # Deteksi wajah dalam setiap frame video dan klasifikasi
        face_coordinates = classify_faces(frame)

        # Tambahkan koordinat border, persentase hasil klasifikasi, dan kategori ke setiap wajah
        for face in face_coordinates:
            face["border_coordinates"] = {
                "x1": face["x"],
                "y1": face["y"],
                "x2": face["x"] + face["width"],
                "y2": face["y"] + face["height"],
            }
            face["classification_percentage"] = round( face["prediction"][face["label"]] * 100, 2) 
            face["category"] = get_category_name(
                face["label"]
            )  # Dapatkan nama kategori berdasarkan label

        # Gabungkan hasil koordinat dan kategori wajah
        results = []
        for face in face_coordinates:
            result = {
                "border_coordinates": face["border_coordinates"],
                "classification_percentage": face["classification_percentage"],
                "category": face["category"],
            }
            results.append(result)

        # Kirim hasil deteksi wajah dan klasifikasi ke client React.js dalam format JSON
        print(results)
        return jsonify({"faces": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run()
