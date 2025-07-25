import os
import yaml
import cv2
import numpy as np
from glob import glob

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_images_from_folder(folder_path, image_size):
    image_paths = glob(os.path.join(folder_path, '*'))
    images = []
    for path in image_paths:
        try:
            img = cv2.imread(path)
            img = cv2.resize(img, tuple(image_size))
            images.append(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return np.array(images)

def normalize_images(images):
    return images.astype('float32') / 255.0

def save_preprocessed_data(images, output_path):
    os.makedirs(output_path, exist_ok=True)
    np.save(os.path.join(output_path, 'preprocessed_images.npy'), images)
    print(f"Preprocessed data saved to {output_path}")

def main():
    config = load_config()

    data_path = config['data']['raw_path']
    output_path = config['data']['processed_path']
    image_size = config['preprocessing']['image_size']

    print(f"Loading images from {data_path}...")
    images = load_images_from_folder(data_path, image_size)
    print(f"{len(images)} images loaded.")

    print("Normalizing images...")
    images = normalize_images(images)

    print("Saving preprocessed data...")
    save_preprocessed_data(images, output_path)

if __name__ == "__main__":
    main()
