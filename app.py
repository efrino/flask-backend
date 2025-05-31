from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and scaler
model_loaded = False
scaler_loaded = False
model_error = ""
scaler_error = ""

try:
    model = tf.keras.models.load_model('model/stunting_model.h5')
    model_loaded = True
except Exception as e:
    model_error = str(e)

try:
    mean = np.load('model/scaler_mean.npy')
    std = np.load('model/scaler_std.npy')
    scaler_loaded = True
except Exception as e:
    scaler_error = str(e)

def preprocess(sex, age, height, weight):
    # Encode gender sesuai tim ML: Laki-laki=0, Perempuan=1
    sex_encoded = 0 if sex.lower() == 'laki-laki' else 1

    # Urutan input: [age, gender_encoded, height, weight]
    input_data = np.array([[float(age), sex_encoded, float(height), float(weight)]], dtype=np.float32)

    # Normalisasi
    scaled = (input_data - mean) / std
    return scaled

@app.route('/')
def home():
    return jsonify({
        'status': 'API is running',
        'model_loaded': model_loaded,
        'scaler_loaded': scaler_loaded,
        'model_error': model_error if not model_loaded else None,
        'scaler_error': scaler_error if not scaler_loaded else None,
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded or not scaler_loaded:
        return jsonify({'error': 'Model or scaler not loaded properly'}), 500

    data = request.json
    sex = data.get('sex')
    age = data.get('age')
    height = data.get('height')
    weight = data.get('weight')

    if sex is None or age is None or height is None or weight is None:
        return jsonify({'error': 'All fields (sex, age, height, weight) are required'}), 400

    try:
        X = preprocess(sex, age, height, weight)
        preds = model.predict(X)

        print("Preds shape:", preds.shape)
        print("Preds:", preds)

        if preds.size == 0:
            return jsonify({'error': 'Prediction output is empty'}), 500

        if len(preds.shape) == 2:
            class_idx = int(np.argmax(preds, axis=1)[0])
            confidence = float(preds[0][class_idx])
        elif len(preds.shape) == 1:
            class_idx = int(np.argmax(preds))
            confidence = float(preds[class_idx])
        else:
            return jsonify({'error': 'Unexpected prediction output shape'}), 500

        labels = ['Severely Stunted', 'Stunted', 'Normal', 'Tall']
        recommendations = [
            'Segera konsultasikan ke petugas kesehatan untuk penanganan stunting.',
            'Segera konsultasikan ke petugas kesehatan untuk penanganan stunting.',
            'Terus pantau tumbuh kembang anak.',
            'Terus pantau tumbuh kembang anak dengan asupan gizi seimbang.'
        ]

        return jsonify({
            'result': labels[class_idx],
            'score': confidence,
            'recommendation': recommendations[class_idx]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
