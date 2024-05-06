from torchvision import datasets


def extract_gesture_names():
    """
    Extracts the gesture names from the ImageFolder dataset.
    
    Returns:
        dict: A dictionary mapping class indices to gesture names.
    """
    # Load the dataset
    dataset = datasets.ImageFolder(root="data")

    # Get the class-to-index mapping
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    # Clean up the class names
    idx_to_class = {k: v.replace("(1)", "").capitalize() for k, v in idx_to_class.items()}

    return idx_to_class
