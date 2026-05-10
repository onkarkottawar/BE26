# Step 1: Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Step 2: Load Dataset

df = pd.read_csv("1_boston_housing.csv")

# Remove quotes from column names
df.columns = df.columns.str.replace('"', '')

print(df.head())


# Step 3: Split Features and Target

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)


# Step 4: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Step 5: Feature Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Step 6: Build Neural Network Model

model = Sequential()

# Input + Output Layer
model.add(Dense(1, input_shape=(X_train.shape[1],), activation='linear'))

# Compile Model
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# Display Model Summary
model.summary()


# Step 7: Train the Model

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    validation_split=0.05,
    verbose=1
)


# Step 8: Evaluate Model

loss, mae = model.evaluate(X_test, y_test)

print("\nTest Loss:", loss)
print("Test MAE:", mae)


# Step 9: Predict Values

predictions = model.predict(X_test)

print("\nFirst 5 Predictions:")
print(predictions[:5])


# Step 10: Plot Loss Graph

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()

plt.show()