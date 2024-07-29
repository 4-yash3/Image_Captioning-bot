import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate, Layer
from tensorflow.keras.models import Model, load_model
import tensorflow as tf

# Load the tokenizer, captions, and features
TOKENIZER_PATH = 'tokenizer.npy'
CAPTIONS_PATH = 'captions.npy'
FEATURES_PATH = 'features.npy'

# Recreate the tokenizer
def load_tokenizer(tokenizer_path):
    word_index = np.load(tokenizer_path, allow_pickle=True).item()
    tokenizer = Tokenizer()
    tokenizer.word_index = word_index
    return tokenizer

# Define the model
def define_model(vocab_size, max_length):
    # Feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder model
    decoder1 = Concatenate()([fe2, se3])  # Ensure inputs are passed as a list
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def main():
    # Load tokenizer, captions, and features
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    padded_captions = np.load(CAPTIONS_PATH)
    features = np.load(FEATURES_PATH)
    
    # Determine max_length
    max_length = padded_captions.shape[1]
    
    # Define and compile the model
    vocab_size = len(tokenizer.word_index) + 1
    model = define_model(vocab_size, max_length)
    
    # Print model summary
    model.summary()

    # Save the model in the Keras format
    model_file = 'image_captioning_model.keras'
    model.save(model_file)
    print(f"Model saved to {model_file}")

    # Load the model
    model = load_model(model_file)
    print("Model loaded successfully.")

if __name__ == '__main__':
    main()
