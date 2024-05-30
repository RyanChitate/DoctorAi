# 2APP.py
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
def predict(image):
    img = Image.open(image).convert('RGB')
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = loaded_model(img)
        predicted_class = torch.argmax(output).item()
        if predicted_class == 0:
            prediction_text = 'Healthy'
        else:
            prediction_text = 'Unhealthy'
    return prediction_text

def predict_body_part(image):
    # Your code to predict the body part from the image
    # This function will return the predicted body part
    pass

# Streamlit app
st.title('Disease Diagnosis Tool')

uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

for idx, uploaded_file in enumerate(uploaded_files):
    st.image(uploaded_file, caption=f'Uploaded Image {idx}', use_column_width=True)
    prediction = predict(uploaded_file)
    st.write(f'Predicted condition for Image {idx}: {prediction}')
    
    body_part_prediction = predict_body_part(uploaded_file)  # Call function to predict body part
    st.write(f'Predicted body part for Image {idx}: {body_part_prediction}')
    
    feedback = st.radio(f'Is the prediction correct for Image {idx}?', ['Yes', 'No'])
    if feedback == 'No':
        correct_prediction = st.radio(f'What is the correct condition for Image {idx}?', ['Healthy', 'Unhealthy'], key=f'correct_{idx}')
        correct_body_part = st.text_input(f'What is the correct body part for Image {idx}?')
        body_disease = st.selectbox(f'Select the body disease for Image {idx}', ['Spinal', 'Liver', 'Brain', 'Bone'])
        if st.button(f'Submit Feedback {idx}'):
            with open('/Users/tio/Documents/GitHub/DoctorAi/model/feedback/feedback.json', 'a') as f:
                f.write(f'{{"image_path": "{uploaded_file.name}", "correct_label": "{correct_prediction}", "correct_body_part": "{correct_body_part}", "body_disease": "{body_disease}"}}\n')
            st.success('Feedback submitted successfully!')
