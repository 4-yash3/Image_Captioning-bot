from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Directory to save uploaded images
UPLOAD_FOLDER = r'D:\RASA\IMAGE-CAPTIONING\archive\Images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        f = request.files['file']
        if f.filename == '':
            return "No selected file", 400
        if f:
            # Secure filename
            filename = secure_filename(f.filename)
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            f.save(image_path)
            
            # Preprocess image and generate caption
            feature = extract_features(image_path)
            caption = generate_caption(feature)
            
            return render_template('result.html', caption=caption)

def extract_features(image_path):
    # Implement your feature extraction logic
    pass

def generate_caption(feature):
    # Implement your caption generation logic
    pass

if __name__ == '__main__':
    app.run(debug=True)


