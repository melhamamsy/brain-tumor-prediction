import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.image import CLASS_2_ID_DICT, ID_2_CLASS_DICT


def create_confusion_matrix(misclassified_images, data_counts):
    n_classes = len(CLASS_2_ID_DICT)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Initialize the diagonal with the correct counts
    for class_name, count in data_counts.items():
        class_id = CLASS_2_ID_DICT[class_name]
        confusion_matrix[class_id, class_id] = count

    # Update the confusion matrix with misclassified examples
    for actual_class, _, predicted_class in misclassified_images:
        actual_id = CLASS_2_ID_DICT[actual_class]
        predicted_id = CLASS_2_ID_DICT[predicted_class]
        confusion_matrix[actual_id, actual_id] -= 1
        confusion_matrix[actual_id, predicted_id] += 1
    
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, title='Confusion Matrix Heatmap'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=[ID_2_CLASS_DICT[i] for i in range(len(ID_2_CLASS_DICT))],
                yticklabels=[ID_2_CLASS_DICT[i] for i in range(len(ID_2_CLASS_DICT))])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()