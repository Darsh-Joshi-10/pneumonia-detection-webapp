from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model(os.path.join('model', 'pneumonia_detection_model.h5'))

# Define the path to save uploaded images
UPLOAD_FOLDER = 'static/uploads/'
PNEUMONIA_FOLDER = 'static/pneumonia_images/'
NORMAL_FOLDER = 'static/normal_images/'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PNEUMONIA_FOLDER'] = PNEUMONIA_FOLDER
app.config['NORMAL_FOLDER'] = NORMAL_FOLDER

# Create directories if they don't exist
os.makedirs(PNEUMONIA_FOLDER, exist_ok=True)
os.makedirs(NORMAL_FOLDER, exist_ok=True)

# Function to make prediction
def predict_pneumonia(img_path):
    img = load_img(img_path, target_size=(150, 150))  # Load the image and resize it
    img = img_to_array(img)  # Convert the image to an array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize the image
    prediction = model.predict(img)
    return 'Pneumonia' if prediction > 0.5 else 'Normal'

# Route for the home page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an empty file
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Make prediction
            prediction = predict_pneumonia(filepath)

            # Determine the correct folder based on prediction
            if prediction == 'Pneumonia':
                save_folder = app.config['PNEUMONIA_FOLDER']
            else:
                save_folder = app.config['NORMAL_FOLDER']

            # Move the file to the appropriate folder
            new_filepath = os.path.join(save_folder, file.filename)
            os.rename(filepath, new_filepath)

            # Render the result page
            return render_template('result.html', prediction=prediction, img_path=new_filepath)
    return render_template('index.html')

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
