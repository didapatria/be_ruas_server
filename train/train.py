from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os

# Path ke direktori data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_train = os.path.join(current_dir, "../data/train")
data_valid = os.path.join(current_dir, "../data/valid")

# Praproses, Augmentasi Data, Splitting Train-Test Data
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
)

# Mengatur ukuran batch dan dimensi gambar
batch_size = 32
img_width, img_height = 150, 150

# Membuat training & validation generator
train_generator = datagen.flow_from_directory(
    data_train,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
)
valid_generator = datagen.flow_from_directory(
    data_valid,
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
model.add(Dropout(0.5))
model.add(Dense(train_generator.num_classes, activation="softmax"))

# Mengompilasi dan melatih model
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    epochs=2,
)

# Check Loss & Accuracy
test_loss, test_acc = model.evaluate(valid_generator, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# Menyimpan model ke file H5
model.save(os.path.join(current_dir, "../model/model.h5"))

# Show result on graph
plt.subplot(1, 2, 1)  # row 1, col 2 index 1
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)  # row 1, col 2 index 2
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")

plt.show()
