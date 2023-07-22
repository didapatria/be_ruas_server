import os

# TensorFlow and tf.keras
import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Helper libraries
import numpy as np

# load model
model = tf.keras.models.load_model("model150.h5")

# Mengatur ukuran batch dan dimensi gambar
batch_size = 32
img_width, img_height = 150, 150

# Image to predict
img_test = "./dataset-evaluate/test-4.jpg"

# predicting images
img = image.load_img(img_test, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
predict_x = model.predict(images, batch_size)
classes_x = np.argmax(predict_x, axis=1)

classes_value = ""

if classes_x == 0:
    classes_value = "normal"
elif classes_x == 1:
    classes_value = "tengok-kiri-kanan"
elif classes_x == 2:
    classes_value = "tengok-depan-belakang"
elif classes_x == 3:
    classes_value = "lirik-kiri-kanan"
elif classes_x == 4:
    classes_value = "lihat-atas"


print("Prediksi untuk gambar ini " + img_test + " : " + classes_value)
