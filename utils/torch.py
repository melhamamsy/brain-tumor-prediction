import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm.auto import tqdm
from utils.image import CLASS_2_ID_DICT, ID_2_CLASS_DICT


class SimpleCNN(nn.Module):
    """
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 4)  # Assuming 4 classes: glioma, meningioma, notumor, pituitary

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def prepare_datasets(data_dir="data", batch_size=32, train_perc=0.8):
    """
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Training"), transform=transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, "Testing"), transform=transform)

    # Apply class to ID mapping
    dataset.classes = list(CLASS_2_ID_DICT.keys())
    dataset.class_to_idx = CLASS_2_ID_DICT
    test_dataset.classes = list(CLASS_2_ID_DICT.keys())
    test_dataset.class_to_idx = CLASS_2_ID_DICT
    
    # Split dataset into training and validation
    train_size = int(train_perc * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.001):
    """
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, end="\n\n")

    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training for {epochs} epochs...", end="\n")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        
        # Evaluate on validation set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f"Validation Accuracy: {100 * correct / total}%")

    
    return model


def save_model(model, model_path):
    """
    """
    torch.save(model.state_dict(), model_path)
    print(f"Model saved in '{model_path}'")


def load_model(model_path):
    """
    """
    return torch.load(model_path, weights_only=True)


def get_misclassified_images(model, data_loader, n_error=None, device="cpu", return_tensors=False):
    """
    Get misclassified images from the data_loader.

    Args:
    - model: The trained model.
    - data_loader: DataLoader containing the data.
    - n_error (int, optional): Number of misclassified images to return. If None, return all misclassified images.
    - device (str): Device to run the model on ("cpu" or "cuda").
    - return_tensors (bool): whether or not to return tensors

    Returns:
    - List of tuples: (image_tensor, correct_label, predicted_label, image_name)
    """
    model.to(device)
    model.eval()
    misclassified_images = []

    original_dataset = data_loader.dataset.dataset  # Access the original dataset
    indices = data_loader.dataset.indices  # Access the indices used in the Subset

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Collect misclassified images
            for i in range(len(labels)):
                if n_error is not None and len(misclassified_images) >= n_error:
                    break
                if predicted[i] != labels[i]:
                    image_index = indices[i]  # Get the original index
                    image_name = original_dataset.samples[image_index][0]  # Get the image name using the original index
                    misclassified_images.append(
                        (
                            ID_2_CLASS_DICT.get(labels[i].cpu().item()),
                            image_name.split('/')[-1],
                            ID_2_CLASS_DICT.get(predicted[i].cpu().item()),
                        )
                    )

                    if return_tensors:
                        misclassified_images[-1] = (images[i].cpu(),) +\
                            misclassified_images[-1]

            if n_error is not None and len(misclassified_images) >= n_error:
                break

    return misclassified_images
