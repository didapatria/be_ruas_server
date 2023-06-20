from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import os

# Path ke direktori data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "data")

# Praproses, Augmentasi Data, Splitting Train-Test Data
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.1,
)

# Mengatur ukuran batch dan dimensi gambar
batch_size = 32
img_width, img_height = 150, 150

# Membuat training & validation generator
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",  # set as training data
)
valid_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",  # set as validation data
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
model.add(Dense(train_generator.num_classes, activation="softmax"))

# Mengompilasi dan melatih model
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# model.fit(generator, steps_per_epoch=generator.samples // batch_size, epochs=100)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=valid_generator,
    # validation_steps=valid_generator.samples//batch_size,
    # validation_freq=1,
    epochs=100,
)

# score=model.evaluate(valid_generator, verbose=0)

# print("Test loss: ", score[0])
# print("Test accuracy: ", score[1])

plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim([0.5, 1])
plt.legend(loc="lower right")
plt.show()

test_loss, test_acc = model.evaluate(valid_generator, verbose=2)
# print(test_acc)
print("Test loss: ", test_loss)
print("Test accuracy: ", test_acc)

# Menyimpan model ke file H5
model.save("be_ruas_server/model.h5")
