import os
import json
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

# Load the existing dataset
dataset_path = '/Users/tio/Documents/GitHub/DoctorAi/model/dataset'
train_dataset = ImageFolder(root=dataset_path, transform=transform)

# Load the feedback data
feedback_path = '/Users/tio/Documents/GitHub/DoctorAi/model/feedback'
feedback_file = os.path.join(feedback_path, 'feedback.json')

feedback_data = []

if os.path.exists(feedback_file):
    with open(feedback_file, 'r') as f:
        for line in f:  # Read each line in the file separately
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Skip empty lines
                try:
                    feedback = json.loads(line)  # Parse JSON from each line
                    image_path = os.path.join(dataset_path, feedback['image_path'])
                    label = feedback['correct_label']
                    if os.path.exists(image_path):  # Check if the image file exists
                        feedback_data.append((image_path, label))
                    else:
                        print(f"Image file not found: {image_path}")
                except json.JSONDecodeError:
                    print(f"Ignoring invalid JSON: {line}")

# Custom dataset to include feedback data
class CustomDataset(data.Dataset):
    def __init__(self, original_dataset, feedback_data, transform):
        self.original_dataset = original_dataset
        self.feedback_data = feedback_data
        self.transform = transform
        self.classes = original_dataset.classes
    
    def __len__(self):
        return len(self.original_dataset) + len(self.feedback_data)
    
    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            img_path, label = self.feedback_data[idx - len(self.original_dataset)]
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img)
            # Convert label to index
            label_idx = self.classes.index(label) if label in self.classes else -1
            return img, label_idx

combined_dataset = CustomDataset(train_dataset, feedback_data, transform)

# Exclude samples with label index -1 (not found in classes)
filtered_dataset = [data for data in combined_dataset if data[1] != -1]

train_loader = data.DataLoader(filtered_dataset, batch_size=32, shuffle=True)

# Load the pre-trained ResNet model
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
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

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
    
    epoch_loss = running_loss / len(filtered_dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

# Save the trained model
model_path = '/Users/tio/Documents/GitHub/DoctorAi/model/trained_model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes
}, model_path)

print("Model retrained and saved successfully.")
