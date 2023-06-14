from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os

# Path ke direktori data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

# Praproses dan Augmentasi Data
datagen = ImageDataGenerator(
    rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
)

# Mengatur ukuran batch dan dimensi gambar
batch_size = 32
img_width, img_height = 150, 150

# Menggunakan generator untuk memuat data gambar dari direktori berdasarkan label folder
generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
)

# Membangun model CNN
model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(generator.num_classes, activation="softmax"))

# Mengompilasi dan melatih model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(generator, steps_per_epoch=generator.samples // batch_size, epochs=100)

# Menyimpan model ke file H5
model.save("be_ruas_server/model.h5")
