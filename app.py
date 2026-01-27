from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('best_auto_mpg_model.pkl')

@app.route('/')
def home():
    return send_file('simple_app.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array(data['data'])
        predictions = model.predict(input_data)
        return jsonify({
            'predictions': predictions.tolist(),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model': 'loaded'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)