import os
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm.auto import tqdm
from utils.image import (CLASS_2_ID_DICT, ID_2_CLASS_DICT,
                         PREFIX_2_CLASS_DICT)


class CustomDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, root_dir, is_vit=True, **kwargs):
        self.dataset = datasets.ImageFolder(
            root=root_dir,
            transform=kwargs.get('transform')
        )
        self.dataset.classes = list(CLASS_2_ID_DICT.keys())
        self.dataset.class_to_idx = CLASS_2_ID_DICT
        
        for i in range(len(self.dataset.samples)):
            path = self.dataset.samples[i][0]
            prefix = path.split('/')[-1][3:5]
            label = PREFIX_2_CLASS_DICT[prefix]

            self.dataset.samples[i] = (path, CLASS_2_ID_DICT[label])
        
        self.is_vit = is_vit
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.is_vit:
            image, label = self.dataset[idx]
            image = image.convert("RGB")
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs = {key: val.squeeze() for key, val in inputs.items()}  # Remove batch dimension
            inputs['labels'] = torch.tensor(label)
            return inputs
        else:
            return self.dataset[idx]
    
    def train_val_split(self, train_perc=0.8, seed=42):
        train_size = int(train_perc * len(self))
        val_size = len(self) - train_size
        return random_split(
            self, 
            [train_size, val_size], 
            generator=torch.Generator().manual_seed(seed)
        )
    

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
    

def get_dataset_counts(dataset):
    data_counts_dict = {}

    for idx in dataset.indices:
        label = dataset.dataset.dataset[idx][1]
        
        label_class = ID_2_CLASS_DICT[label]
        
        data_counts_dict[label_class] =\
            data_counts_dict.get(label_class, 0) + 1

    return data_counts_dict


def prepare_datasets(data_dir="data", is_vit=False):
    """
    """
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = CustomDataset(
        root_dir=os.path.join(data_dir, "Training"),
        is_vit=is_vit,
        transform=transform,
    )
    test_dataset = CustomDataset(
        root_dir=os.path.join(data_dir, "Testing"),
        is_vit=is_vit,
        transform=transform,
    )

    return dataset, test_dataset


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

    original_dataset = data_loader.dataset.dataset.dataset
    indices = data_loader.dataset.indices  # Access the indices used in the Subset
    batch_size = data_loader.batch_size

    with torch.no_grad():
        for k, loaded_data in enumerate(tqdm(data_loader)):
            if data_loader.dataset.dataset.is_vit:
                images = loaded_data['pixel_values']
                labels = loaded_data['labels'] 
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.logits, 1)
            else:
                images, labels = loaded_data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
            
            # Collect misclassified images
            for i in range(len(labels)):
                if n_error is not None and len(misclassified_images) >= n_error:
                    break
                if predicted[i] != labels[i]:
                    image_index = indices[k*batch_size + i]  # Get the original index
                    image_name = original_dataset.samples[image_index][0]
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
