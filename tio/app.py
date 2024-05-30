import os
import json
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
from torchvision.datasets import ImageFolder

# Load the trained model
model_path = '/Users/tio/Documents/GitHub/DoctorAi/model/aimodel.pth'
checkpoint = torch.load(model_path)
num_classes = checkpoint['num_classes']

loaded_model = models.resnet18(pretrained=False)
loaded_model.fc = nn.Linear(loaded_model.fc.in_features, num_classes)
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the feedback data for mapping
feedback_path = '/Users/tio/Documents/GitHub/DoctorAi/model/feedback/feedback.json'
feedback_data = []

if os.path.exists(feedback_path):
    try:
        with open(feedback_path, 'r') as f:
            feedback_data = json.load(f)
        print("Feedback JSON is valid.")
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in feedback file: {e}")

# Extract unique classes from feedback data
class_to_attributes = {}
for item in feedback_data:
    label = item['correct_label']
    body_part = item['correct_body_part']
    body_disease = item['body_disease']
    class_to_attributes[label] = {'body_part': body_part, 'body_disease': body_disease}

# Get class labels from the dataset
dataset_path = '/Users/tio/Documents/GitHub/DoctorAi/model/dataset'
train_dataset = ImageFolder(root=dataset_path, transform=transform)
class_labels = train_dataset.classes

# Function to make predictions
def predict(image):
    img = Image.open(image).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = loaded_model(img)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Function to display predictions with styling
def display_prediction(uploaded_file, idx):
    predicted_class_idx = predict(uploaded_file)
    predicted_class_label = class_labels[predicted_class_idx]
    
    # Get attributes for the predicted class
    attributes = class_to_attributes.get(predicted_class_label, {'body_part': 'Unknown', 'body_disease': 'Unknown'})
    health_status = 'Healthy' if predicted_class_label == 'Healthy' else 'Unhealthy'
    body_part = attributes['body_part']
    body_disease = attributes['body_disease']
    
    # Display uploaded image
    st.write(f'---')
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image(uploaded_file, caption=f'Uploaded Image {idx}', use_column_width=True)
    with col2:
        # Display prediction
        st.write(f'**Predicted class for Image {idx}:** {"🌱 Healthy" if health_status == "Healthy" else "🩺 Unhealthy"}')
        if health_status == 'Healthy':
            st.write('Not severe. 🩹 No need for follow-up.')
        else:
            st.write('Address urgently! 🚨 Urgent care needed.')
        st.write(f'**Body Part:** {body_part}')
        st.write(f'**Body Disease:** {body_disease}')

# Streamlit app
st.title('Disease Diagnosis Tool')

uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Display predictions for each uploaded image
    for idx, uploaded_file in enumerate(uploaded_files):
        display_prediction(uploaded_file, idx)
