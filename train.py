import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Add
from tensorflow.keras.models import Model

# Load the tokenizer, captions, and features
TOKENIZER_PATH = 'tokenizer.npy'
FEATURES_PATH = 'features.npy'
IMAGE_FILES_PATH = 'image_files.npy'
CAPTION_FILE = r'D:\RASA\IMAGE-CAPTIONING\archive\captions.txt'
MAX_LENGTH_PATH = 'max_length.npy'

def load_captions(filename):
    """Load captions from a file."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    captions_dict = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        image, caption = line.split(',', 1)
        if image not in captions_dict:
            captions_dict[image] = []
        captions_dict[image].append(caption.strip())
    return captions_dict

def load_tokenizer(tokenizer_path):
    word_index = np.load(tokenizer_path, allow_pickle=True).item()
    tokenizer = Tokenizer(num_words=10000)  # Limit the vocabulary size
    tokenizer.word_index = word_index
    return tokenizer

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
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def data_generator(tokenizer, max_length, captions_dict, features, image_files, vocab_size, batch_size):
    while True:
        X1, X2, y = [], [], []
        for i, img_name in enumerate(image_files):
            if img_name not in captions_dict:
                continue
            for caption in captions_dict[img_name]:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for j in range(1, len(seq)):
                    in_seq, out_seq = seq[:j], seq[j]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[i])
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) >= batch_size:
                        yield (np.array(X1), np.array(X2)), np.array(y)
                        X1, X2, y = [], [], []

def main():
    # Load tokenizer, captions, and features
    tokenizer = load_tokenizer(TOKENIZER_PATH)
    features = np.load(FEATURES_PATH).astype('float32')
    image_files = np.load(IMAGE_FILES_PATH)
    
    # Load captions dictionary
    captions_dict = load_captions(CAPTION_FILE)
    all_captions = [caption for captions in captions_dict.values() for caption in captions]

    # Determine max_length
    max_length = max(len(seq) for seq in tokenizer.texts_to_sequences(all_captions))

    # Save max_length
    np.save(MAX_LENGTH_PATH, np.array([max_length]))

    # Define and compile the model
    vocab_size = min(len(tokenizer.word_index) + 1, 10000)  # Limit the vocabulary size
    model = define_model(vocab_size, max_length)
    
    # Create the data generator
    batch_size = 32  # Adjusted batch size for memory optimization
    steps_per_epoch = len(all_captions) // batch_size
    train_generator = data_generator(tokenizer, max_length, captions_dict, features, image_files, vocab_size, batch_size)
    
    # Define the dataset
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: train_generator,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 4096), dtype=tf.float32),
                tf.TensorSpec(shape=(None, max_length), dtype=tf.float32)
            ),
            tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
        )
    )
    
    # Train the model using the dataset
    model.fit(dataset, epochs=20, steps_per_epoch=steps_per_epoch, verbose=2)
    
    # Save the model in the .keras format
    model.save('imagecaptioning_model.keras', save_format='keras')
    print(f"Model saved to image_captioning_model.keras")

if __name__ == '__main__':
    main()
