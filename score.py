import json
import joblib
import numpy as np
import os

def init():
    global model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'best_auto_mpg_model.pkl')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        data = json.loads(raw_data)
        input_data = np.array(data["data"])
        
        predictions = model.predict(input_data)
        
        result = {
            "predictions": predictions.tolist(),
            "status": "success",
            "model": "Auto MPG Predictor"
        }
        return json.dumps(result)
        
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "status": "error"
        })