import tensorflow as tf


def build_model(input_shape, num_classes):
    """
    Builds a simple neural network model for image classification.
    Args:
        input_shape (tuple): Shape of the input images, e.g., (28, 28, 1).
        num_classes (int): Number of output classes for classification.
    Returns:
        tf.keras.Model: Compiled Keras model ready for training.
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, train_ds, val_ds, epochs=5):
    """
    Trains the given model on the provided dataset.
    Args:
        model (tf.keras.Model): Compiled Keras model to be trained.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        epochs (int): Number of epochs for training.
    Returns:
        tf.keras.callbacks.History: History object containing training metrics.
    """
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return history


def evaluate_model(model, val_ds):
    """
    Evaluates the model on the validation dataset.
    Args:
        model (tf.keras.Model): Trained Keras model.
        val_ds (tf.data.Dataset): Validation dataset.
    Returns:
        Tuple[float, float]: Loss and accuracy on the validation dataset.
    """
    loss, accuracy = model.evaluate(val_ds)
    print(f"\nValidation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    return loss, accuracy
