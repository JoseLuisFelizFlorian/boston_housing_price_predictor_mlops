import os
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template


# Definición de la ruta base (el directorio raíz del proyecto donde reside app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Rutas de Archivos Específicos
SCALER_PATH = os.path.join(BASE_DIR, 'artefacts', 'scaler.pkl')      
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_linear_regression.pkl')

# Features (necesarias para la validación del input)
FEATURE_COLUMNS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


# --- Inicialización de la Aplicación ---
app = Flask(__name__)

# --- Cargar Artefactos Globalmente ---
try:

    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(f"StandardScaler cargado exitosamente desde: {SCALER_PATH}")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo de Regresión Lineal cargado exitosamente desde: {MODEL_PATH}")

except FileNotFoundError:
    print("ERROR: No se pudieron encontrar los archivos de artefactos.")
    print(f"Buscando scaler en: {SCALER_PATH}")
    print(f"Buscando modelo en: {MODEL_PATH}")
    print("Asegúrese de ejecutar el pipeline y que las carpetas 'artefacts/' y 'models/' existan.")
    scaler = None
    model = None


# ---Web app para prueba de api ---
@app.route('/app', methods=['GET'])
def home():
    """Sirve la plantilla HTML principal con el formulario."""
    return render_template('index.html')

# --- Ruta de Salud (Health Check) ---
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "API de Predicción de Precios de Boston Housing está lista.",
        "endpoint_predict": "/predict (POST)",
        "features_required": FEATURE_COLUMNS, # Esto lista todas las 13 columnas
        "documentacion": "Consulte el archivo input_example.json o la documentación Swagger/Redoc para más detalles."
    }), 200 

# --- Ruta de Predicción ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({"error": "Modelos no cargados. Revisar logs del servidor."}), 500

    try:
        data = request.json
        input_data = [data.get(col) for col in FEATURE_COLUMNS]
        
        if None in input_data:
            missing_cols = [col for col in FEATURE_COLUMNS if data.get(col) is None]
            return jsonify({"error": "Faltan features requeridas.", "missing": missing_cols}), 400

        features_array = np.array([input_data], dtype=np.float64)
        
        # Preprocesamiento (Escalado) usando el scaler cargado
        features_scaled = scaler.transform(features_array)
        
        # Predicción
        prediction_scaled = model.predict(features_scaled)
        
        predicted_price = float(prediction_scaled[0])
        
        return jsonify({
            "predicted_price_in_usd": round(predicted_price, 2),
            "units": "USD"
        })

    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return jsonify({"error": f"Error interno del servidor durante la predicción: {str(e)}"}), 500

# --- Bloque de Prueba ---
if __name__ == '__main__':
    # Para la prueba local (no usar en producción)
    app.run(host='0.0.0.0', port=5000, debug=True)