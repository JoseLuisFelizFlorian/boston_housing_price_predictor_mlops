import os
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime

# Importamos todas las constantes definidas que proviene del archivo src\config.py
from .config import (
    PROCESSED_DIR,          # Usado para cargar los datos (.npy)
    MODEL_PATH,             # Usado para guardar el modelo .pkl
    REPORTS_DIR,            # Usado para guardar el reporte de métricas
    FEATURE_COLUMNS         # Usado para etiquetar los coeficientes en el reporte
)

# --- Definición de la Función de Entrenamiento ---
def train_and_save_model():
    """
    Carga los datos procesados, entrena el modelo de Regresión Lineal, 
    persiste el modelo y guarda un reporte detallado de las métricas.
    """
    print("Iniciando fase de entrenamiento...")
    
    # Cargar datos procesados
    X_train_scaled = np.load(os.path.join(PROCESSED_DIR, 'X_train_scaled.npy'))
    X_test_scaled = np.load(os.path.join(PROCESSED_DIR, 'X_test_scaled.npy'))
    y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
    y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

    print(f"Datos de entrenamiento cargados. X_train shape: {X_train_scaled.shape}")
    
    # Inicializar y Entrenar el Modelo
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    print("Modelo de Regresión Lineal entrenado exitosamente.")

    # Evaluación Detallada y Generación de Reporte (Trazabilidad MLOps)
    y_pred = model.predict(X_test_scaled)
    
    # Métricas estándar
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Métricas Requeridas
    rmse = np.sqrt(mse) 
    intercept = model.intercept_ 
    coefficients = model.coef_ 
    
    # Crear carpeta de reportes si no existe
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, 'metrics_summary.txt')
    
    # Escribir el Reporte
    with open(report_path, 'w') as f:
        f.write(f"========= REPORTE DE ENTRENAMIENTO MLOPS =========\n")
        f.write(f"Fecha de Ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo Entrenado: Regresión Lineal\n\n")
        f.write(f"--- RENDIMIENTO EN EL SET DE PRUEBA ---\n")
        f.write(f"R2 Score: {r2:.4f}\n")
        f.write(f"Mean Squared Error (MSE): {mse:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n\n")
        
        f.write(f"--- ANÁLISIS DEL MODELO ---\n")
        f.write(f"INTERCEPTO (Bias): {intercept:.4f}\n")
        f.write("COEFICIENTES POR FEATURE:\n")
        
        # Guardar coeficientes con sus etiquetas (usa FEATURE_COLUMNS)
        for feature, coef in zip(FEATURE_COLUMNS, coefficients):
            f.write(f"  {feature:<10}: {coef:.6f}\n")
        
        f.write(f"\n===================================================\n")
        
    print(f"Rendimiento del modelo en Test: R2={r2:.4f}, RMSE={rmse:.2f}")
    print(f"Reporte de métricas persistido en: {report_path}")

    # Persistir el modelo
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"Modelo persistido en: {MODEL_PATH}")


# --- Bloque de Prueba (Opcional, si se ejecuta directamente) ---
if __name__ == '__main__':
    train_and_save_model()