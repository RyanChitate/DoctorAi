import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the pre-trained ResNet50 model
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Add a new classification layer for your medical image classification task
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)  # Assuming binary classification

# Create the new model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define class labels
class_labels = ['Normal', 'Abnormal']

# Function to preprocess the image
def preprocess_image(image):
    img = image.convert('RGB')  # Convert image to RGB format
    img = img.resize((224, 224))  # Resize image to the input size required by ResNet50
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# Function to make predictions
def predict(image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)
    return predictions

# Define the Streamlit app
def main():
    st.title('Chest X-ray Classifier')

    st.write('Upload a chest X-ray image for classification.')

    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make predictions using the model
        predictions = predict(image)

        # Display the prediction result
        predicted_class_index = np.argmax(predictions)
        predicted_class_label = class_labels[predicted_class_index]
        predicted_probability = predictions[0][predicted_class_index]

        st.write(f'Predicted Class: {predicted_class_label}')
        st.write(f'Probability: {predicted_probability:.4f}')

if __name__ == '__main__':
    main()
