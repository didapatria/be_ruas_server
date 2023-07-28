import os

# TensorFlow and tf.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Helper libraries
import numpy as np

# load model
model = load_model("model/data-modeling.h5")

# Mengatur ukuran batch dan dimensi gambar
batch_size = 32
img_width, img_height = 224, 224

# Image to predict
img_to_test = [
    "./dataset-evaluate/test-0.png",
    "./dataset-evaluate/test-1.png",
    "./dataset-evaluate/test-2.jpg",
    "./dataset-evaluate/test-3.jpg",
    "./dataset-evaluate/test-4.jpg",
    "./dataset-evaluate/0_normal_laptop (3).png",
]

# predicting images
for img_test in img_to_test:
    img = image.load_img(img_test, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    predict_x = model.predict(images, batch_size)
    classes_x = np.argmax(predict_x, axis=1)

    classes_value = ""

    if classes_x == 0:
        classes_value = "menyontek-lihat-atas"
    elif classes_x == 1:
        classes_value = "menyontek-lihat-depan"
    elif classes_x == 2:
        classes_value = "menyontek-menengok"
    elif classes_x == 3:
        classes_value = "normal"

    print("Prediksi untuk gambar ini '" + img_test + "' : " + classes_value)
