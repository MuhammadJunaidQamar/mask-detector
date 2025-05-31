import os
from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load model and face detector
model = tf.keras.models.load_model('mask_model.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_mask(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    predictions = []
    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        resized_face = cv2.resize(face_roi, (224, 224))
        normalized_face = resized_face / 255.0
        prediction = model.predict(np.expand_dims(normalized_face, axis=0))
        label = 'Mask' if np.argmax(prediction) == 0 else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
        
        # Draw bounding box and label
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        predictions.append(label)
    
    # Save processed image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    return output_path, predictions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        output_path, predictions = predict_mask(filepath)
        return render_template('index.html', 
                             original=filepath,
                             processed=output_path,
                             predictions=predictions)
    return redirect(request.url)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)