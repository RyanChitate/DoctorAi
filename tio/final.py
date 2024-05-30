import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image

# Load the trained model
model_path = '/Users/tio/Documents/GitHub/DoctorAi/model/trained_model.pth'
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

# Function to make predictions
def predict(images):
    predictions = []
    for img in images:
        img = Image.open(img).convert('RGB')
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = loaded_model(img)
            _, predicted = torch.max(output, 1)
        
        class_labels = ['Healthy', 'Unhealthy']  # Define human-readable class labels
        
        # Map the predicted class to 'Healthy' or 'Unhealthy'
        predicted_idx = predicted.item()
        if predicted_idx < len(class_labels):
            predicted_class = class_labels[predicted_idx]
        else:
            predicted_class = 'Unknown'
        
        predictions.append(predicted_class)
    
    return predictions

# Streamlit app
st.title('Disease Diagnosis Tool 👨‍⚕️')  # Add doctor emoji to the title

uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # Display the uploaded images and their predictions
    for uploaded_file in uploaded_files:
        st.write(f'---')
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(uploaded_file, caption=f'📄 {uploaded_file.name}', use_column_width=True)
        with col2:
            predictions = predict([uploaded_file])
            for prediction in predictions:
                if prediction == 'Healthy':
                    st.write(f'**Prediction:** 🌱 Healthy')
                    st.write('Not severe. 🩹 No need for follow-up.')
                else:
                    st.write(f'**Prediction:** 🩺 Unhealthy')
                    st.write('Address urgently! 🚨  Urgent care needed.')

