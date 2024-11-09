from PIL import Image
import numpy as np


def preprocess_image(image_path, img_size=(28, 28)):
    """
    Loads and preprocesses an image for prediction.
    Args:
        image_path (str): Path to the image file.
        img_size (tuple): Target size for resizing the image.
    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize(img_size)  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model, image_path):
    """
    Predicts the class of a given image using a trained model.
    Args:
        model (tf.keras.Model): Trained Keras model.
        image_path (str): Path to the image to be predicted.
    Returns:
        int: Predicted class label.
    """
    processed_image = preprocess_image(image_path)

    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=-1)[0]
    confidence = np.max(predictions, axis=-1)[0]

    print(f"Predicted Class: {predicted_class} with confidence {confidence:.2f}")
    return predicted_class
