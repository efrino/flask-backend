from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os

from predict_stunting import predict_stunting

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    html = """
    <!DOCTYPE html>
    <html lang="id">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
      <title>GrowSmart Flask API</title>
      <style>
        body {
          font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
          background: #f9f9f9;
          color: #333;
          margin: 0;
          padding: 0;
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          height: 100vh;
        }
        .container {
          text-align: center;
          background: white;
          padding: 2rem;
          border-radius: 1rem;
          box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        h1 {
          color: #0077cc;
          margin-bottom: 1rem;
        }
        code {
          background: #eee;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
          display: block;
          margin: 0.5rem auto;
          max-width: 500px;
          text-align: left;
        }
        footer {
          margin-top: 2rem;
          font-size: 0.9rem;
          color: #777;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>🚀 GrowSmart Flask API</h1>
        <p>API siap digunakan!</p>
        <p>Gunakan endpoint berikut:</p>
        <code>POST /api/predict</code>
        <code>POST /api/predictions <span style="color: #888;">(dengan token Bearer)</span></code>
        <p>dengan body JSON seperti:</p>
        <code>{<br> "age": 24,<br> "gender": "Laki-laki"<span style="color: #888;">(atau Perempuan)</span>,<br> "height": 83,<br> "weight": 10.5<br>}</code>
        <footer>GrowSmart Flask API &copy; 2025</footer>
      </div>
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "success",
        "message": "GrowSmart Flask API aktif dan responsif.",
        "available_endpoints": [
            {
                "method": "POST",
                "endpoint": "/api/predict",
                "auth": "Tidak perlu login"
            },
            {
                "method": "POST",
                "endpoint": "/api/predictions",
                "auth": "Diperlukan token Bearer (user login)"
            }
        ],
        "required_body_format": {
            "age": "Usia anak dalam bulan (misal: 24)",
            "gender": "Jenis kelamin, pilih antara 'Laki-laki' atau 'Perempuan'",
            "height": "Tinggi badan anak dalam cm (misal: 83)",
            "weight": "Berat badan anak dalam kg (misal: 10.5)"
        },
        "example_body": {
            "age": 24,
            "gender": "Laki-laki",
            "height": 83,
            "weight": 10.5
        }
    })
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    gender = data.get('gender') or data.get('sex')  # Support 'sex' or 'gender'
    age = data.get('age')
    height = data.get('height')
    weight = data.get('weight')

    # Validasi awal
    if gender is None or age is None or height is None or weight is None:
        return jsonify({'error': 'Semua field (gender, age, height, weight) wajib diisi'}), 400

    try:
        # Panggil fungsi utama prediksi
        result = predict_stunting(
            age=float(age),
            gender=gender,
            height=float(height),
            weight=float(weight)
        )

        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
