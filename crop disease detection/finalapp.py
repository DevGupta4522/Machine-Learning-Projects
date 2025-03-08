import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from tensorflow.keras.preprocessing import image 

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/best_vgg16_model.keras")  

model = load_model()

with open('model/class_indices.json') as f:
    class_indices = json.load(f)
    classes=class_indices
    

# Disease Solutions Dictionary
with open('model/solutions_indices.json') as f:
    solutions_indices = json.load(f)
    solutions = solutions_indices

# Define function for making predictions
def predict(img):
    img = img.resize((224, 224))  # Resize image
    img_array = np.array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions
    img_array = img_array / 255.0  # Normalize

    predictions = model.predict(img_array)
    print(predictions)
    predicted_class = classes[str(np.argmax(predictions[0]))]
    predicted_solutions= solutions[predicted_class]
    confidence = round(100 * np.max(predictions[0]), 2)
    
    return predicted_class, confidence, predicted_solutions
    

st.title("AI Crop Disease Detector")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg","JPG"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        result = predict(image)
        st.write(f"Prediction: {result}")