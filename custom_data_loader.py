import os
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split


def load_images_from_folder(folder_path, img_size=(28, 28)):
    """
    Loads images from a folder structure where each subfolder represents a label.
    Args:
        folder_path (str): Path to the main dataset folder.
        img_size (tuple): Target size to resize images.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays of images and corresponding labels.
    """
    print('Started loading training and testing images...')
    images = []
    labels = []
    for label in sorted(os.listdir(folder_path)):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            for img_file in os.listdir(label_path):
                img_path = os.path.join(label_path, img_file)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize(img_size)  # Resize to target size
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    images.append(img_array)
                    labels.append(int(label))  # Use the folder name as the label
                except Exception as e:
                    print(f"Failed to process image {img_path}: {e}")
    print('Loading images complete!')
    return np.array(images), np.array(labels)


def create_tf_datasets(images, labels, batch_size=32, validation_split=0.2):
    """
    Splits the data into training and validation sets and creates TensorFlow datasets.
    Args:
        images (np.ndarray): Array of image data.
        labels (np.ndarray): Array of labels corresponding to the images.
        batch_size (int): Batch size for the datasets.
        validation_split (float): Proportion of data to use for validation.
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation TensorFlow datasets.
    """
    # Split into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=validation_split, random_state=123)

    # Add a channel dimension (required for grayscale images in TensorFlow)
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)

    # Convert to TensorFlow datasets
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))

    # Shuffle and batch the datasets
    train_ds = train_ds.shuffle(buffer_size=1000).batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    return train_ds, val_ds


def load_custom_dataset(folder_path, img_size=(28, 28), batch_size=32, validation_split=0.2):
    """
    Main function to load and prepare a custom dataset for digit recognition.
    Args:
        folder_path (str): Path to the dataset folder.
        img_size (tuple): Target size for resizing images.
        batch_size (int): Batch size for the datasets.
        validation_split (float): Proportion of data to use for validation.
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Training and validation TensorFlow datasets.
    """
    images, labels = load_images_from_folder(folder_path, img_size=img_size)
    train_ds, val_ds = create_tf_datasets(images, labels, batch_size=batch_size, validation_split=validation_split)
    return train_ds, val_ds
