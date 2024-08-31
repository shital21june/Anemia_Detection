from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained CNN model
cnn_model = load_model('cnn_model_normalized.h5')

# Threshold probability value for classification
threshold = 0.5  # Adjust as needed

@app.route('/')
def index():
    return render_template('index.html')

# Function to extract features from an eye image
def extract_features(image):
    # Define coordinates for the lower palpebral conjunctiva region
    x1, y1, x2, y2 = 43, 2115, 2807, 3400  # Adjust these coordinates as needed

    # Extract lower palpebral conjunctiva region using the defined coordinates
    lower_palpebral_conjunctiva = image[y1:y2, x1:x2]

    # Calculate mean intensity of red and green components within the lower palpebral conjunctiva region
    red_intensity = np.mean(lower_palpebral_conjunctiva[:, :, 2])  # Index 2 corresponds to red channel
    green_intensity = np.mean(lower_palpebral_conjunctiva[:, :, 1])  # Index 1 corresponds to green channel

    # Calculate the difference between mean red and green intensities
    intensity_difference = red_intensity - green_intensity

    return [red_intensity, green_intensity, intensity_difference]

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    image_file = request.files['image']
    
    # Read the image file
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Define width and height
    width, height = 224, 224
    
    # Extract features from the image
    features = extract_features(image)
    
    # Resize image to match the input size of the CNN model
    resized_image = cv2.resize(image, (width, height))
    
    # Normalize pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0
    
    # Expand dimensions to make it suitable for prediction
    input_image = np.expand_dims(normalized_image, axis=0)
    
    # Make prediction using the CNN model
    prediction = cnn_model.predict(input_image)
    
    # Check if the probability of the class exceeds the threshold
    if prediction[0][0] > threshold:
        predicted_class = "Anemia"
    else:
        predicted_class = "No Anemia"
    
    return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
