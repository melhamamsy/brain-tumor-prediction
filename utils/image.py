import os
import random
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


# Define a dictionary to map folder names to labels
CLASS_2_ID_DICT = {
    "notumor": 0,
    "pituitary": 1,
    "meningioma": 2,
    "glioma": 3,
}

ID_2_CLASS_DICT = {
    0: "notumor",
    1: "pituitary",
    2: "meningioma",
    3: "glioma",
}

PREFIX_2_CLASS_DICT = {
    "no": "notumor",
    "pi": "pituitary",
    "me": "meningioma",
    "gl": "glioma",
}


def load_images(data_dir="data", type="Training", image_names=None, seed=None, **kwargs):
    """
    Load a specified number of images from each category and return them as a list of tuples
    (image_tensor, label).
    
    Args:
    - data_dir (str): Root directory containing the data.
    - type (str): Either "Training" or "Testing".
    - kwargs (dict): Number of images to load from each category 
        (glioma, meningioma, notumor, pituitary). Defaults to 1 per category.
    
    Returns:
    - List of tuples (image_tensor, label).
    """
    if seed is not None:
        random.seed(seed)

    assert type in {"Training", "Testing"}, "`type` must be in {'Training', 'Testing'}"
    
    # Initialize transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    data_path = os.path.join(data_dir, type)
    image_data = []
    
    for category, label in CLASS_2_ID_DICT.items():
        num_images = kwargs.get(category, 1)
        category_path = os.path.join(data_path, category)
        
        if not os.path.exists(category_path):
            print(f"Path does not exist: {category_path}")
            continue
        
        if image_names:
            selected_files = [
                image_name for image_name in image_names \
                    if image_name[3:5] == category[:2]
            ]
        else:
            image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
            selected_files = random.sample(image_files, min(num_images, len(image_files)))
        
        for file_name in selected_files:
            image_path = os.path.join(category_path, file_name)
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image)
            image_data.append((image_tensor, label, file_name))
    
    return image_data


def visualize_image(image_data_item):
    """
    Visualize an image from the image_data.
    
    Args:
    - image_data_item (tuple): A tuple containing (image_tensor, label, image_id).
    """
    
    image_tensor, label, image_id = image_data_item
    
    # Convert the tensor to a NumPy array and transpose the dimensions
    # Tensor shape: [C, H, W] -> NumPy array shape: [H, W, C]
    image_array = image_tensor.numpy().transpose((1, 2, 0))
    
    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.title(f"Label: {ID_2_CLASS_DICT.get(label)}, ID: {image_id}")
    plt.imshow(image_array)
    plt.axis('off')  # Hide the axis
    plt.show()


def count_images(data_dir="data", type="Training"):
    """
    Count the number of images for each category in the specified directory.
    
    Args:
    - data_dir (str): Root directory containing the data.
    - type (str): Either "Training" or "Testing".
    
    Returns:
    - dict: A dictionary with categories as keys and the number of images as values.
    """

    assert type in {"Training", "Testing"}, "`type` must be in {'Training', 'Testing'}"

    data_path = os.path.join(data_dir, type)
    categories = CLASS_2_ID_DICT.keys()
    image_counts = {}

    for category in categories:
        category_path = os.path.join(data_path, category)
        
        if not os.path.exists(category_path):
            print(f"Path does not exist: {category_path}")
            image_counts[category] = 0
            continue
        
        image_files = [f for f in os.listdir(category_path) if f.endswith('.jpg')]
        image_counts[category] = len(image_files)
    
    return image_counts