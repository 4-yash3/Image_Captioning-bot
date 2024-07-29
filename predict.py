import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from PIL import Image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model

# Define the paths
MODEL_PATH = 'image_captioning_model.keras'
TOKENIZER_PATH = 'tokenizer.npy'
FEATURES_PATH = 'features.npy'
MAX_LENGTH_PATH = 'max_length.npy'

def load_tokenizer(tokenizer_path):
    word_index = np.load(tokenizer_path, allow_pickle=True).item()
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    tokenizer.index_word = {index: word for word, index in word_index.items()}
    print(f"Tokenizer index_word mapping: {tokenizer.index_word}")
    return tokenizer


def preprocess_image(image_path):
    """Preprocess the image for feature extraction."""
    image = Image.open(image_path)
    image = image.resize((224, 224))  # Resize as per model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)  # Preprocess for ResNet50
    return image

def extract_features(image):
    """Extract features from the image using a feature extractor model."""
    # Load a pre-trained ResNet50 model with average pooling
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
    
    features = feature_extractor.predict(image)
    
    # Ensure feature shape is (2048,)
    if features.shape[1] != 2048:
        raise ValueError(f"Expected feature shape (2048,) but got {features.shape[1]}")
    
    # Convert features to (4096,)
    model_input = tf.keras.Input(shape=(2048,))
    x = Dense(4096, activation='relu')(model_input)
    feature_4096 = tf.keras.Model(inputs=model_input, outputs=x).predict(features)
    
    return feature_4096

def generate_caption(model, tokenizer, photo, max_length):
    """Generate a caption for a given image."""
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=40)  # Ensure this matches model training
        yhat = model.predict([photo, sequence], verbose=0)
        
        # Debug print statements
        print(f"Input sequence: {sequence}")
        print(f"Model prediction: {yhat}")

        # Verify yhat shape
        print(f"Model output shape: {yhat.shape}")

        yhat = np.argmax(yhat)
        
        # Debug print the predicted token ID and corresponding word
        word = tokenizer.index_word.get(yhat, None)
        print(f"Predicted token ID: {yhat}, Word: {word}")
        
        if word is None:
            print("No valid word found, breaking loop.")
            break
        in_text += ' ' + word
        if word == 'endseq':
            print("End token found, breaking loop.")
            break
    
    return in_text



def main(image_path):
    # Load tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    
    # Load the model
    model = load_model(MODEL_PATH)
    
    # Print model summary to verify output shape
    model.summary()
    
    # Preprocess the image
    image = preprocess_image(image_path)
    feature = extract_features(image)
    
    # Load max_length
    max_length = np.load(MAX_LENGTH_PATH).item()

    # Predict the caption
    caption = generate_caption(model, tokenizer, feature, max_length)
    
    print(f"Predicted Caption: {caption}")







if __name__ == '__main__':
    image_path = 'D:/RASA/IMAGE-CAPTIONING/archive/Images/41999070_838089137e.jpg'  # Replace with your image path
    main(image_path)
