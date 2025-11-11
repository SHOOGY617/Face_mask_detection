# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import math
import os

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\Users\Agraw\Downloads\Face-Mask-Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

# grab the list of images in our dataset directory, then initialize
# the list of data (i.e., images) and class images
# Instead of loading the entire dataset into memory (which can cause
# large allocations), use ImageDataGenerator.flow_from_directory to stream
# images from disk in batches. This avoids creating a huge NumPy array.
# We use a 20% validation split handled by the generator.
trainAug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest",
	preprocessing_function=preprocess_input,
	validation_split=0.20)

valAug = ImageDataGenerator(
	preprocessing_function=preprocess_input,
	validation_split=0.20)

# Create generators that read images from the directory on-the-fly
train_generator = trainAug.flow_from_directory(
	DIRECTORY,
	classes=CATEGORIES,
	target_size=(224, 224),
	batch_size=BS,
	class_mode="categorical",
	subset="training",
	shuffle=True)

val_generator = valAug.flow_from_directory(
	DIRECTORY,
	classes=CATEGORIES,
	target_size=(224, 224),
	batch_size=BS,
	class_mode="categorical",
	subset="validation",
	shuffle=False)

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)  # use learning_rate instead of lr
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
H = model.fit(
	train_generator,
	steps_per_epoch=train_generator.samples // BS,
	validation_data=val_generator,
	validation_steps=val_generator.samples // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
# predict on the validation generator (make sure to cover all samples)
preds = model.predict(val_generator, steps=math.ceil(val_generator.samples / BS))

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(preds, axis=1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
# show a nicely formatted classification report. `val_generator.classes`
# holds the true class indices for the validation subset (in the same
# order as the generator yields them when shuffle=False).
true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())
print(classification_report(true_classes, predIdxs,
	target_names=class_labels))

# serialize the model to disk
print("[INFO] saving mask detector model...")
# Keras 3 deprecates the `save_format` argument. Provide a filename
# with a supported extension instead (either .h5 or .keras).
model.save("mask_detector.model.h5")

# plot the training loss and accuracy
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
plt.savefig("plot.png")