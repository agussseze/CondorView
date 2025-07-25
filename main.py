import os
import yaml
import numpy as np
from preprocessing import load_and_preprocess_images
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ============================
# 1. Cargar configuración
# ============================
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# 2. Cargar imágenes preprocesadas

images = np.load(os.path.join(config["paths"]["output_dir"], "preprocessed_images.npy"))

# Para este ejemplo usaremos etiquetas dummy (falsas)
# En un caso real, debes cargarlas desde un archivo anotado con clases (por ejemplo, incendios, inundaciones, etc.)
labels = np.random.randint(0, 2, len(images))  
labels = to_categorical(labels, num_classes=2)


# 3. Dividir en entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=config["train"]["random_seed"]
)


# 4. Definir el modelo CNN

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(
        config["preprocessing"]["img_height"], config["preprocessing"]["img_width"], 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# 5. Entrenar el modelo

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=config["train"]["epochs"],
    batch_size=config["train"]["batch_size"]
)


# 6. Guardar el modelo entrenado

model.save(os.path.join(config["paths"]["model_dir"], "condorview_model.h5"))

print("Modelo entrenado y guardado correctamente.")
