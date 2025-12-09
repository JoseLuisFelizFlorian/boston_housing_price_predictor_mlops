import os

# --- CONSTANTES GLOBALES ---

# Semilla aleatoria para reproducibilidad
RANDOM_STATE = 42 

# Nombre de la columna objetivo (Target)
TARGET_COLUMN = 'PRICE' 

# URL de los datos brutos de Boston Housing (Datos Originales)
RAW_DATA_URL = "http://lib.stat.cmu.edu/datasets/boston"

# Nombres de las columnas
COLUMNS_NAMES = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 
    'PTRATIO', 'B', 'LSTAT', TARGET_COLUMN
]

# Lista de columnas Feature (excluyendo el Target para el entrenamiento)
FEATURE_COLUMNS = [col for col in COLUMNS_NAMES if col != TARGET_COLUMN]

# --- RUTAS DE ARCHIVOS ---

# Directorio base (la carpeta raíz del proyecto)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Rutas de Directorios
RAW_DIR = os.path.join(BASE_DIR, 'dataset', '01_raw')
INTERIM_DIR = os.path.join(BASE_DIR, 'dataset', '02_interim')
PROCESSED_DIR = os.path.join(BASE_DIR, 'dataset', '03_processed')
ARTEFACTS_DIR = os.path.join(BASE_DIR, 'artefacts')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')


# Rutas de Archivos Específicos
RAW_DATA_PATH = os.path.join(RAW_DIR, 'boston_housing_raw.txt')
INTERIM_DATA_PATH = os.path.join(INTERIM_DIR, 'boston_housing_interim.csv')
PROCESSED_CSV_PATH = os.path.join(PROCESSED_DIR, 'boston_housing_final.csv')

# Rutas de Artefactos MLOps
SCALER_PATH = os.path.join(ARTEFACTS_DIR, 'scaler.pkl')
MODEL_PATH = os.path.join(MODELS_DIR, 'model_linear_regression.pkl')