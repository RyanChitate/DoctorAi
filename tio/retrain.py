import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from PIL import Image

# Define the class labels
class_labels = ['healthy', 'unhealthy']  # Adjust these labels according to your dataset

# Define the transforms for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load original training dataset
train_dataset = ImageFolder(root='/Users/tio/Documents/GitHub/DoctorAi/model/dataset', transform=transform)

# Load feedback data
feedback_file = '/Users/tio/Documents/GitHub/DoctorAi/model/feedback/feedback.json'
if os.path.exists(feedback_file):
    with open(feedback_file, 'r') as f:
        feedback_data = json.load(f)
    
    feedback_images = []
    feedback_labels = []
    
    for item in feedback_data:
        img = Image.open(item['image_path']).convert('RGB')
        img = transform(img)
        label = class_labels.index(item['correct_label'])
        
        feedback_images.append(img)
        feedback_labels.append(label)
    
    feedback_dataset = TensorDataset(torch.stack(feedback_images), torch.tensor(feedback_labels))
    combined_dataset = ConcatDataset([train_dataset, feedback_dataset])
else:
    combined_dataset = train_dataset

# Create data loader
train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

# Load a pre-trained ResNet model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer for our multi-class classification task
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_labels))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Save the trained model
model_path = '/Users/tio/Documents/GitHub/DoctorAi/model/trained_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': len(class_labels)
}, model_path)

# Verify if the loaded model is valid
loaded_model = models.resnet18(weights=None)
loaded_model.fc = nn.Linear(loaded_model.fc.in_features, len(class_labels))
loaded_model.load_state_dict(torch.load(model_path)['model_state_dict'])
assert isinstance(loaded_model, nn.Module), "Loaded model is not an instance of a PyTorch model"
print("Model retrained and saved successfully.")
