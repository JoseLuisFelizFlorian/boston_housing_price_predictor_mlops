import sys

# Importamos las funciones principales de los módulos
from .data_pipeline import clean_and_split_data
from .train_and_save import train_and_save_model

# -- Ejecuta Todos los Pipiline ---
def run_full_pipeline():
    """
    Ejecuta el pipeline completo de MLOps:
    1. Obtención, preprocesamiento y persistencia de datos (scaler y .npy).
    2. Entrenamiento, evaluación y persistencia del modelo.
    """
    try:
        print("=============================================")
        print("      INICIANDO PIPELINE MLOPS COMPLETO      ")
        print("=============================================")
        
        # Ejecutar la fase de Datos y Preprocesamiento
        print("\n--- FASE 1: EJECUTANDO DATA PIPELINE ---")
        clean_and_split_data()
        
        # Ejecutar la fase de Entrenamiento y Persistencia
        print("\n--- FASE 2: EJECUTANDO ENTRENAMIENTO Y GUARDADO ---")
        train_and_save_model()
        
        print("\n=============================================")
        print("     PIPELINE MLOPS COMPLETO EXITOSO       ")
        print("=============================================")
        
    except Exception as e:
        print(f"\n=============================================")
        print(f"     ERROR CRÍTICO EN EL PIPELINE  ")
        print(f"=============================================")
        print(f"Error: {e}")
        # Terminar el programa con un código de error
        sys.exit(1)


# --- Bloque de ejecución principal ---
if __name__ == '__main__':
    run_full_pipeline()