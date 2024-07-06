import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.model_selection import train_test_split
import warnings

# Digit classifier MNIST
X = np.load("X.npy")
y = np.load("y.npy")

# 70-15-15 split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# cool plot
m, n = X.shape
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
plt.tight_layout(pad=0.1)
warnings.simplefilter(action='ignore', category=FutureWarning)

for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(m)
    randomX = X[random_index].reshape(20, 20).T
    ax.imshow(randomX, cmap="grey")
    ax.set_title(y[random_index, 0])
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)

plt.show()

model = Sequential(
    [
        Input(shape=(X.shape[1],)),
        Dense(25, activation="relu"),
        Dense(20, activation="relu"),
        Dense(15, activation="relu"),
        Dense(10, activation="linear"),
    ]
)
print(model.summary())

model.compile(
    loss=SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.01),
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=80, validation_data=(X_cv, y_cv))
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

logits = model.predict(X_test)
predictions = np.argmax(logits, axis=1)
fig, axes = plt.subplots(8, 8, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(len(X_test))
    random_val_x = X_test[random_index].reshape(20, 20)
    ax.imshow(random_val_x, cmap="grey")
    ax.set_title(f"{y_test[random_index,0]}, {predictions[random_index]}")
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=10)
plt.tight_layout()
plt.show()
