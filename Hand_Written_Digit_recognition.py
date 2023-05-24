import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained model
model = load_model(r'C:\Users\Dev Patel\Documents\Collage_files\Data Science subjects\DL\project.h5')

# Set up Streamlit layout
st.title("Handwritten Digit Recognition")
st.write("Draw a digit on the canvas and let the model predict it.")

# Create a canvas to draw the digit
canvas = st_canvas(
    fill_color="rgb(0, 0, 0)",
    stroke_width=20,
    stroke_color="rgb(255, 255, 255)",
    background_color="rgb(0, 0, 0)",
    width=150,
    height=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Preprocess the image
def preprocess_image(image):
    img = Image.fromarray(image.astype('uint8')).convert('L')
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape((1, 28, 28))
    return img

# Predict the digit
if st.button("Predict"):
    # Preprocess the canvas image
    img = preprocess_image(canvas.image_data.astype('float32'))

    # Make the prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    # Display the prediction
    st.subheader("Prediction")
    st.write(f"The model predicts the digit as: {predicted_class}")

    # Display the canvas image
    st.subheader("Digit Image")
    plt.imshow(canvas.image_data, cmap='gray')
    plt.axis('off')
    st.pyplot()

# Reset the canvas
if st.button("Clear"):
    canvas.image_data = np.zeros((150, 150, 3), dtype=np.uint8)

# Explanation and information
st.write("This app uses a pre-trained machine learning model to recognize handwritten digits. It is trained on the MNIST dataset, which consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The model takes an image of size 28x28 pixels and predicts the corresponding digit. You can draw a digit on the canvas, and the model will predict the digit based on your drawing.")
st.set_option('deprecation.showPyplotGlobalUse', False)
