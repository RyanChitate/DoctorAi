import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from PIL import Image

# Define the transforms for preprocessing the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to fit the input size of the pre-trained model
    transforms.ToTensor(),           # Convert images to tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
])

# Load the dataset using ImageFolder
train_dataset = ImageFolder(root='/Users/tio/Documents/GitHub/DoctorAi/model/dataset', transform=transform)
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer for our multi-class classification task
num_ftrs = model.fc.in_features
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(num_ftrs, num_classes)

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
    
    epoch_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Save the trained model
model_path = '/Users/tio/Documents/GitHub/DoctorAi/model/aimodel.pth'
torch.save(model.state_dict(), model_path)

# Load the saved model
loaded_model = models.resnet18(pretrained=False)  # Create a new instance of the model
loaded_model.fc = nn.Linear(loaded_model.fc.in_features, num_classes)  # Adjust the final layer
loaded_model.load_state_dict(torch.load(model_path))  # Load the saved state dictionary

# Verify if the loaded model is valid
assert isinstance(loaded_model, nn.Module), "Loaded model is not an instance of a PyTorch model"
print("Model loaded successfully.")

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on the training set: {100 * accuracy:.2f}%')
