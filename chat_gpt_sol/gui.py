import tkinter as tk
import numpy as np
from tensorflow import keras
from PIL import Image, ImageDraw
import io

# Load the MNIST dataset
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the input data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0
y_train_one_hot = keras.utils.to_categorical(y_train, num_classes=10)
y_test_one_hot = keras.utils.to_categorical(y_test, num_classes=10)

# Define the neural network architecture
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train_one_hot, batch_size=128, epochs=1, validation_data=(X_test, y_test_one_hot))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f"Test loss: {loss}")
print(f"Test accuracy: {accuracy}")

# Create a canvas to draw on
canvas_width = 280
canvas_height = 280
canvas_color = "white"
brush_size = 15

def paint(event):
    x1, y1 = event.x, event.y
    x2, y2 = event.x + brush_size, event.y + brush_size
    canvas.create_oval(x1, y1, x2, y2, fill="black")

def clear_canvas():
    canvas.delete("all")
    
def recognize_digit():
    digit_data = canvas.postscript(colormode="gray")
    img = Image.open(io.BytesIO(digit_data.encode("utf-8")))
    img = img.resize((28, 28))
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype("float32") / 255.0
    pred = model.predict(img)
    digit_label.config(text=f"Predicted digit: {np.argmax(pred)}")

root = tk.Tk()
root.title("Handwritten Digit Recognizer")

canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg=canvas_color)
canvas.pack(side="top", fill="both", expand=True)
canvas.bind("<B1-Motion>", paint)

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack(side="left")

recognize_button = tk.Button(root, text="Recognize", command=recognize_digit)
recognize_button.pack(side="left")

digit_label = tk.Label(root, text="")
digit_label.pack(side="left")

root.mainloop()
