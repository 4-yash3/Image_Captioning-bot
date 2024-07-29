import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Directory paths
CAPTION_FILE = r'D:\RASA\IMAGE-CAPTIONING\archive\captions.txt'

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

def preprocess_captions(captions):
    """Tokenize and pad captions."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return tokenizer, padded_sequences, max_length

def main():
    # Load and preprocess captions
    captions_dict = load_captions(CAPTION_FILE)
    all_captions = [caption for captions in captions_dict.values() for caption in captions]
    tokenizer, padded_captions, max_length = preprocess_captions(all_captions)

    # Save the tokenizer and captions
    np.save('tokenizer.npy', tokenizer.word_index)
    np.save('captions.npy', padded_captions)

    # Print some intermediate results
    print(f"Number of captions: {len(all_captions)}")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    print(f"Max caption length: {max_length}")
    print(f"Example padded caption: {padded_captions[0]}")

if __name__ == '__main__':
    main()
