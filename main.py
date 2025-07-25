import os
import numpy as np
from preprocessing import load_images_from_folder, preprocess_images, encode_labels
from models.dummy_model import DummyModel
import yaml

# Cargar configuración desde config.yaml
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

data_path = config["data_path"]
img_height = config["img_height"]
img_width = config["img_width"]

# Cargar imágenes y etiquetas
print("Cargando imágenes...")
images, labels = load_images_from_folder(data_path, img_height, img_width)
images = preprocess_images(images)
labels_encoded, class_names = encode_labels(labels)

# Cargar modelo dummy
model = DummyModel()

# Hacer predicciones
print("Realizando predicciones...\n")
predictions = model.predict(images)

# Mostrar resultados
for i in range(len(images)):
    predicted_class_index = np.argmax(predictions[i])
    predicted_class = class_names[predicted_class_index]
    confidence = predictions[i][predicted_class_index]
    real_class = labels[i]
    print(f"Clase real: {real_class} -> Predicción: {predicted_class} (confianza: {confidence:.2f})")
