import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import warnings

# Digit classifier MNIST
X = np.load("X.npy")
y = np.load("y.npy")

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
    optimizer=tf.keras.optimizers.Adam(0.01)
)
model.fit(X, y, epochs=40)
logits = model.predict(X)
f_x = np.zeros(len(logits))

for i in range(len(logits)):
    f_x[i] = np.argmax(logits[i])

fig, axes = plt.subplots(8, 8, figsize=(8, 8))

for i, ax in enumerate(axes.flat):
    random_index = np.random.randint(m)
    random_val_x = X[random_index].reshape(20, 20)
    ax.imshow(random_val_x, cmap="grey")
    ax.set_title(f"{y[random_index,0]}, {f_x[random_index]}")
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize=14)
plt.show()

arr = np.mean(f_x == y.squeeze())
print(f"Accuracy of the model, {arr}")
