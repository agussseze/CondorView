import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_images_from_folder(folder, img_height, img_width):
    images = []
    labels = []
    class_names = os.listdir(folder)
    for class_name in class_names:
        class_folder = os.path.join(folder, class_name)
        if not os.path.isdir(class_folder):
            continue
        for filename in os.listdir(class_folder):
            img_path = os.path.join(class_folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (img_width, img_height))
                images.append(img)
                labels.append(class_name)
    return np.array(images), np.array(labels)

def preprocess_images(images):
    return images.astype("float32") / 255.0

def encode_labels(labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    labels_categorical = to_categorical(labels_encoded)
    return labels_categorical, le.classes_
