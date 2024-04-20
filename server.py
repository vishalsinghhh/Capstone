from flask import Flask, jsonify, request
import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

app = Flask(__name__)

model = load_model('model.h5')  # Replace with your model path

# List of class names
classes = ['Fair_Light', 'Medium_Tan', 'Dark_Deep']

# Mapping dictionary for descriptive skin tone labels
descriptive_labels = {
    'Fair_Light': 'Fair / Light',
    'Medium_Tan': 'Medium / Tan',
    'Dark_Deep': 'Dark / Deep'
}

# Load the MTCNN face detection model
mtcnn = MTCNN()

# Route to handle POST request with image
@app.route('/api/upload', methods=['POST'])
def upload_image():
    # Check if the POST request contains a file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    # Get the file from the request
    file = request.files['file']
    
    # Check if the file has a filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Read the image from the file
    nparr = np.fromstring(file.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Predict skin tone
    try:
        faces = mtcnn.detect_faces(image)
        if len(faces) > 0:
            largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
            x, y, w, h = largest_face['box']
            detected_face = image[y:y+h, x:x+w]
            
            # Resize the detected face to the desired input shape
            detected_face = cv2.resize(detected_face, (120, 90))
            
            # Preprocess the detected face for classification
            detected_face = tf.keras.applications.mobilenet_v2.preprocess_input(detected_face[np.newaxis, ...])
            
            # Predict the class of the face
            predictions = model.predict(detected_face)
            predicted_class_idx = np.argmax(predictions)
            predicted_class = classes[predicted_class_idx]
            
            # Get the descriptive label from the mapping dictionary
            descriptive_label = descriptive_labels[predicted_class]
            
            # Return the prediction
            return jsonify({'predicted_skin_tone': descriptive_label})
        else:
            return jsonify({'error': 'No face detected in the uploaded image'}), 400
    except Exception as e:
        return jsonify({'error': f'Error detecting faces: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
