import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.applications import ResNet50 #type:ignore
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions #type:ignore

# Title
st.title('Medical Image Diagnosis Tool')

# Upload image
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display the uploaded image
    st.image(image, caption='Uploaded Chest X-ray', use_column_width=True)
    
    # Preprocess the image
    image = cv2.resize(image, (224, 224))
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    # Load pre-trained ResNet50 model
    model = ResNet50(weights='imagenet')
    
    # Make prediction
    prediction = model.predict(image)
    decoded_predictions = decode_predictions(prediction, top=1)[0]
    label = decoded_predictions[0][1]
    confidence = decoded_predictions[0][2]
    
    # Display prediction
    st.write(f"Predicted Label: {label}")
    st.write(f"Confidence: {confidence}")
