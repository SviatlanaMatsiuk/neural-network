import custom_data_loader
import model_trainer
from digit_prediction import predict_image

dataset_path = 'training_dataset'

train_ds, val_ds = custom_data_loader.load_custom_dataset(
    folder_path=dataset_path,
    img_size=(28, 28),
    batch_size=32,
    validation_split=0.2
)

input_shape = (28, 28, 1)
num_classes = 10
model = model_trainer.build_model(input_shape=input_shape, num_classes=num_classes)

epochs = 5
history = model_trainer.train_model(model, train_ds, val_ds, epochs=epochs)

model_trainer.evaluate_model(model, val_ds)

image_path = 'handwritten_6.png'

predicted_class = predict_image(model, image_path)
print(f"The model predicts this digit as: {predicted_class}")