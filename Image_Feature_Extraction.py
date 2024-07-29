import os
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# Directory paths
IMAGE_DIR = r'D:\RASA\IMAGE-CAPTIONING\archive\Images'
IMAGE_SIZE = (224, 224)  # Target size for the images

def preprocess_image(image_path, target_size=IMAGE_SIZE):
    """Preprocess the image for feature extraction."""
    image = Image.open(image_path).convert('RGB').resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)
    return image_array

def extract_features(model, image_folder, target_size=IMAGE_SIZE):
    """Extract features from all images in the folder."""
    image_paths = [os.path.join(image_folder, img_name) for img_name in os.listdir(image_folder)]
    features = []
    image_files = []

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        preprocessed_image = preprocess_image(img_path, target_size)
        feature = model.predict(preprocessed_image)
        features.append(feature.flatten())
        image_files.append(img_name)

    return np.array(features), image_files

def main():
    # Load the VGG16 model
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    # Extract features
    features, image_files = extract_features(model, IMAGE_DIR)

    # Save the features and image file names
    np.save('features.npy', features)
    np.save('image_files.npy', image_files)

    # Print some intermediate results
    print(f"Number of images: {len(image_files)}")
    print(f"Feature shape: {features.shape}")

if __name__ == '__main__':
    main()

