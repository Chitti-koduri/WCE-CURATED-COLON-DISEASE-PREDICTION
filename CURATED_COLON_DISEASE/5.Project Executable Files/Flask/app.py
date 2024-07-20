import os
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load your model
model = load_model('model_1/Vgg.h5')

print('Model Loaded!')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')  # Assuming you have an about.html template

@app.route('/predict')
def predict():
    return render_template('details.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')  # Assuming you have a contact.html template

@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        f = request.files['image']
        
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        
        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)
        
        # Read and preprocess the image
        img = load_img(filepath, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # Normalize the image
        
        # Make predictions using the model
        predictions = model.predict(x)
        class_index = np.argmax(predictions, axis=1)[0]
        class_names = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']
        result_text = f"Prediction: {class_names[class_index]} with confidence {predictions[0][class_index]:.2f}"
        
        return render_template('result.html', result=result_text)
    
    return redirect(url_for('predict'))  # Redirect to the predict page if not POST

if __name__ == '__main__':
    app.run(debug=True, port=5000)