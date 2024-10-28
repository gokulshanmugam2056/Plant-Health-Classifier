from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dummy dataset for model training
def load_data():
    X = np.random.rand(100, 4096)  # 100 samples, each having 64x64 image features flattened
    y = np.random.randint(2, size=100)  # Binary classification (0 or 1)
    return X, y

# Train the model
def train_model():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    return model

# Global model variable
model = train_model()

# Function to extract features from the input image
def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (64, 64))  # Resize to 64x64
    features = resized.flatten() / 255.0  # Normalize pixel values
    return features.reshape(1, -1)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to classify an uploaded image
@app.route('/classify', methods=['POST'])
def classify_image():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    try:
        # Read the image file
        image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Validate the image
        if image is None:
            return jsonify({"error": "Invalid image format"}), 400

        # Extract features
        features = extract_features(image)

        # Predict class and probability
        prediction = model.predict(features)
        prediction_prob = model.predict_proba(features)

        # Prepare the result
        result = {
            'prediction': int(prediction[0]),  # Return the class (0 or 1)
            'confidence': float(np.max(prediction_prob))  # Confidence score
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
