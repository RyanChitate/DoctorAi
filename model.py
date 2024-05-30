import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image

def run():
    try:
        # Load the trained model
        model_path = '/Users/tio/Documents/GitHub/DoctorAi/tio/trained_model.pth'
        checkpoint = torch.load(model_path)

        # Check if 'num_classes' key exists in the checkpoint dictionary
        num_classes = checkpoint.get('num_classes', 2)  # Default to 2 if 'num_classes' is not found

        # Load the model architecture
        loaded_model = models.resnet18(pretrained=False)

        # Use the appropriate attribute for the final fully connected layer
        if hasattr(loaded_model, 'fc'):
            in_features = loaded_model.fc.in_features
            loaded_model.fc = nn.Linear(in_features, num_classes)
        elif hasattr(loaded_model, 'classifier'):
            in_features = loaded_model.classifier.in_features
            loaded_model.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Unable to determine the final fully connected layer")

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

                class_labels = ['Healthy', 'Unhealthy', 'Brain Cancer', 'Liver Disease', 'Broken Bone']  # Define human-readable class labels

                # Ensure predicted item is within the range of available labels
                predicted_item = min(predicted.item(), len(class_labels) - 1)

                predicted_class = class_labels[predicted_item]  # Get the predicted class label

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
                            st.write(f'**Predicted class:** {"🌱 **Healthy**"}')
                            st.write('Not severe. 🩹 No need for follow-up.')
                        else:
                            st.write(f'**Predicted class:** {"🩺 **Unhealthy**"}')
                            st.write('Address urgently! 🚨 Urgent care needed.')

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.stop()

run()
