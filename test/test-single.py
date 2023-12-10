import os

# TensorFlow and tf.keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Helper libraries
import numpy as np

# load model
current_dir = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(current_dir, "../model/model.h5"))

# Mengatur ukuran batch dan dimensi gambar
batch_size = 32
img_width, img_height = 150, 150

# Image to predict
img_test = os.path.join(current_dir, "../data/test/test-0.png")

# predicting image
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
    classes_value = "menyontek-tengok-kiri-kanan"
elif classes_x == 3:
    classes_value = "normal"


print("Prediksi untuk gambar ini '" + img_test + "' : " + classes_value)
