from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    AveragePooling2D,
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Path ke direktori data
# current_dir = os.path.dirname(os.path.abspath(__file__))
# data_dir = os.path.join(current_dir, "dataset")

imagePaths = list(paths.list_images("dataset"))
data = []
labels = []
# loop over the image paths
for imagePath in imagePaths:
    # extract the class label from the filename
    label = imagePath.split(os.path.sep)[-2]

    # load the input image (224x224) and preprocess it
    image = load_img(imagePath, target_size=(224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    # update the data and labels lists, respectively
    data.append(image)
    labels.append(label)
# convert the data and labels to NumPy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

baseModel = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(4, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

lb = MultiLabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
print("data: ", data)
print("labels: ", labels)
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.20, stratify=labels, random_state=42
)
# construct the training image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Praproses, Augmentasi Data, Splitting Train-Test Data
# datagen = ImageDataGenerator(
#     rescale=1.0 / 255,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     validation_split=0.1,
# )

INIT_LR = 1e-4
EPOCHS = 20
BS = 32
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR, weight_decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
print("len testX:", len(testX))

H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS,
)

# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     validation_data=valid_generator,
#     # validation_steps=valid_generator.samples // batch_size,
#     # validation_freq=1,
#     epochs=100,
# )


# Mengatur ukuran batch dan dimensi gambar
# batch_size = 32
# img_width, img_height = 224, 224

# Membuat training & validation generator
# train_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="training",  # set as training data
# )
# valid_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size,
#     class_mode="categorical",
#     subset="validation",  # set as validation data
# )

# Membangun model CNN
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(64, (3, 3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(128, (3, 3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())
# # model.add(Dropout(0.5))
# model.add(Dense(128, activation="relu"))
# model.add(Dense(train_generator.num_classes, activation="softmax"))

# # Mengompilasi dan melatih model
# model.summary()
# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# # model.fit(generator, steps_per_epoch=generator.samples // batch_size, epochs=100)

# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     validation_data=valid_generator,
#     # validation_steps=valid_generator.samples // batch_size,
#     # validation_freq=1,
#     epochs=100,
# )

# Check Loss & Accuracy
# test_loss, test_acc = model.evaluate(valid_generator, verbose=2)
# print(test_acc)
# print("Test loss:", test_loss)
# print("Test accuracy:", test_acc)

# Menyimpan model ke file H5
# model.save("be_ruas_server/model4.h5")
# To save the trained model
model.save("mask_recog_ver2.h5")

# Show result on graph
# plt.subplot(1, 2, 1)  # row 1, col 2 index 1
# plt.plot(history.history["accuracy"], label="accuracy")
# plt.plot(history.history["val_accuracy"], label="val_accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.ylim([0.5, 1])
# plt.legend(loc="lower right")

# plt.subplot(1, 2, 2)  # row 1, col 2 index 2
# plt.plot(history.history["loss"], label="loss")
# plt.plot(history.history["val_loss"], label="val_loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.ylim([0.5, 1])
# plt.legend(loc="lower right")

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")

plt.show()
