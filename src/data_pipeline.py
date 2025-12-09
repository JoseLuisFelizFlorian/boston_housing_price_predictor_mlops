import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from urllib.request import urlretrieve

# Importamos todas las constantes definidas que proviene del archivo src\config.py
from .config import (
    RANDOM_STATE, 
    TARGET_COLUMN, 
    COLUMNS_NAMES, 
    RAW_DATA_URL, 
    RAW_DIR, 
    PROCESSED_DIR, 
    ARTEFACTS_DIR, 
    SCALER_PATH
)

# --- Funciones de Utilidad (Crear Directorios) ---
def create_directories():
    """ 
    Asegura que las carpetas de datos y artefactos existan.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(ARTEFACTS_DIR, exist_ok=True)


# --- Obtención y Carga de Datos ---
def fetch_and_load_data(url):

    """
    Descarga los datos si no existen (crea una copia raw para consulta).
    Carga los datos para el tratamiento DIRECTAMENTE desde la URL
    y usa la lógica de reconstrucción de matriz (np.hstack) del notebook.
    """
    raw_data_path = os.path.join(RAW_DIR, 'boston_housing_raw.txt')
    
    # Asegura la existencia del archivo raw (Consulta/Trazabilidad)
    if not os.path.exists(raw_data_path):
        print(f"Descargando y guardando copia raw para consulta en: {raw_data_path}")
        urlretrieve(url, raw_data_path) 
    else:
        print("La copia raw de los datos ya existe (para consulta).")

    # Cargar y Reconstruir datos (Lógica EXACTA del Notebook)
    print("Cargando y reconstruyendo datos directamente desde la URL (506, 14).")


    try:
        # Leer los datos crudos desde la URL. Esto produce (1012, 11) 
        # debido al formato irregular de 2 líneas por registro.
        raw_df = pd.read_csv(
            url, 
            sep=r"\s+", 
            skiprows=22, 
            header=None,
        )
        
        # Reconstrucción de la Matriz (np.hstack) - Lógica del Notebook
        
        # Características (CRIM a LSTAT)
        data_features = np.hstack([
            # 11 columnas de las filas impares (index 0, 2, 4...) -> CRIM a PTRATIO
            raw_df.values[::2, :], 
            
            # 2 columnas de las filas pares (index 1, 3, 5...) -> B y LSTAT
            raw_df.values[1::2, :2] 
        ])
        
        # Variable Objetivo (PRICE) - Columna 3 (index 2) de las filas pares
        data_target = raw_df.values[1::2, 2].reshape(-1, 1)
        
        # Ensamblar Features y Target en el DataFrame final
        data_final = np.hstack([data_features, data_target])
        df = pd.DataFrame(data_final, columns=COLUMNS_NAMES)

    except Exception as e:
        # Fallback a la copia local si la URL falla, usando la misma lógica
        print(f"ADVERTENCIA: Fallo al cargar de la URL ({e}). Cargando desde la copia local (raw.txt).")
        raw_df = pd.read_csv(
            raw_data_path, 
            sep=r"\s+", 
            skiprows=22, 
            header=None,
        )
        data_features = np.hstack([
            raw_df.values[::2, :], 
            raw_df.values[1::2, :2] 
        ])
        data_target = raw_df.values[1::2, 2].reshape(-1, 1)
        data_final = np.hstack([data_features, data_target])
        df = pd.DataFrame(data_final, columns=COLUMNS_NAMES)
    
    # Verificación final de dimensiones
    if df.shape != (506, 14):
         raise ValueError(f"Fallo en el ensamblaje final. Se esperaban (506, 14), pero se obtuvieron {df.shape}.")
    
    # Crear DataFrame con los nombres de columna definidos en config.py
    df.columns=COLUMNS_NAMES
    
    print(f"Datos cargados. Tamaño final: {df.shape}")
    return df


# --- Limpieza y División de Datos ---
def clean_and_split_data():
    """
    Ejecuta la limpieza, división y escalado de datos.
    """
    
    # Preparación de la estructura
    create_directories()
    
    # Obtener datos
    data = fetch_and_load_data(RAW_DATA_URL)
    
    # Separar target (y) de features (X)
    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]
    
    # División de Datos (80% Train, 20% Test, usando RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=RANDOM_STATE
    )
    
    # Estandarización / Escalado (StandardScaler)
    print("Ajustando StandardScaler a los datos de entrenamiento...")
    scaler = StandardScaler()
    
    # Ajustar (fit) solo con los datos de entrenamiento
    scaler.fit(X_train)
    
    # Transformar los sets de entrenamiento y prueba
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Escalado completado.")
    
    # Persistencia de Artefactos MLOps (Scaler y Datos Procesados)
    
    # Guardar el scaler entrenado (CRÍTICO para el despliegue)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"StandardScaler persistido en: {SCALER_PATH}")
    
    # Guardar los sets de datos escalados como arrays NumPy para un acceso rápido
    np.save(os.path.join(PROCESSED_DIR, 'X_train_scaled.npy'), X_train_scaled)
    np.save(os.path.join(PROCESSED_DIR, 'X_test_scaled.npy'), X_test_scaled)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train.values)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test.values)
    print(f"Sets de datos procesados (.npy) guardados en: {PROCESSED_DIR}")


# --- Bloque de Prueba (Opcional, si se ejecuta directamente) ---
if __name__ == '__main__':
    # Esto permite probar el script individualmente
    clean_and_split_data()