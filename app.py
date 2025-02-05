from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
import os

model_path = "best_model.h5"

if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}. Ensure the correct path.")

app = Flask(__name__)
CORS(app)

# Load your pre-trained model here
# model = tf.keras.models.load_model('model.h5')

# Emotion labels (update these according to your model's classes)
EMOTIONS = ['angry', 'happy', 'neutral', 'sad', 'surprise', 'love']

def base64_to_image(base64_string):
    # Remove the data URL prefix if present
    if 'data:image' in base64_string:
        base64_string = re.sub('^data:image/.+;base64,', '', base64_string)
    
    # Decode base64 string to bytes
    img_data = base64.b64decode(base64_string)
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    
    # Decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load face cascade classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None
    
    # Process the first face found
    (x, y, w, h) = faces[0]
    roi = gray[y:y + h, x:x + w]
    
    # Resize to match your model's input size (adjust size as needed)
    roi = cv2.resize(roi, (48, 48))
    
    # Normalize and prepare for model
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    
    return roi

@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Get image data from request
        data = request.json
        image_data = data['image']
        
        # Convert base64 to image
        image = base64_to_image(image_data)
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        if processed_image is None:
            return jsonify({'error': 'No face detected'}), 400
        
        # For testing without model
        # Replace this with actual model prediction
        # emotion = np.random.choice(EMOTIONS)
        
        # When you have your model:
        prediction = model.predict(processed_image)
        emotion = EMOTIONS[np.argmax(prediction[0])]

        return jsonify({
            'emotion': emotion,
            'confidence': 0.95  # Replace with actual confidence when using model
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)