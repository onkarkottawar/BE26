# =========================
# 1. IMPORT LIBRARIES
# =========================

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image

import matplotlib.pyplot as plt
import numpy as np


# =========================
# 2. LOAD DATASET
# =========================

(train_images, train_labels), (test_images, test_labels) = \
    tf.keras.datasets.fashion_mnist.load_data()

print("Dataset Loaded Successfully")


# =========================
# 3. NORMALIZE DATA
# =========================

train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape for CNN
train_images = train_images.reshape((-1, 28, 28, 1))
test_images = test_images.reshape((-1, 28, 28, 1))


# =========================
# 4. BUILD CNN MODEL
# =========================

model = models.Sequential([

    # First Convolution Layer
    layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(28, 28, 1)
    ),

    layers.MaxPooling2D((2, 2)),

    # Second Convolution Layer
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    layers.MaxPooling2D((2, 2)),

    # Third Convolution Layer
    layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    # Flatten Layer
    layers.Flatten(),

    # Dense Layer
    layers.Dense(64, activation='relu'),

    # Output Layer
    layers.Dense(10, activation='softmax')
])


# =========================
# 5. COMPILE MODEL
# =========================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# =========================
# 6. TRAIN MODEL
# =========================

history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels)
)


# =========================
# 7. EVALUATE MODEL
# =========================

test_loss, test_acc = model.evaluate(
    test_images,
    test_labels
)

print("\nTest Accuracy:", test_acc)


# =========================
# 8. MAKE PREDICTIONS
# =========================

predictions = model.predict(test_images)

print("\nPredicted Label:",
      np.argmax(predictions[0]))

print("Actual Label:",
      test_labels[0])


# =========================
# 9. TEST CUSTOM IMAGE
# =========================

img = image.load_img(
    "shoe.jpg",
    target_size=(28, 28),
    color_mode='grayscale'
)

img = image.img_to_array(img)

img = img / 255.0

# Invert image colors
img = 1 - img

img = img.reshape(1, 28, 28, 1)

pred = model.predict(img)

print("\nCustom Image Prediction:",
      np.argmax(pred))


# =========================
# 10. PLOT ACCURACY GRAPH
# =========================

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")

plt.title("Training vs Validation Accuracy")

plt.legend()

plt.show()