# 🏠 Boston Housing Price Predictor (MLOps End-to-End)

Este proyecto implementa una solución completa de Machine Learning Operacional (MLOps) para predecir el precio de casas en Boston. La solución incluye un pipeline de entrenamiento, una API RESTful con Flask/Gunicorn, contenerización con Docker y una interfaz web simple para pruebas.

---

## 1. Arquitectura y Componentes Clave

El flujo de trabajo MLOps abarca desde la experimentación hasta el despliegue contenerizado. 

| Componente | Herramienta | Propósito |
| :--- | :--- | :--- |
| **Pipeline ML** | Scikit-learn, Pandas | Entrenar un modelo de Regresión Lineal y serializar el modelo y el *scaler* (preprocesamiento). |
| **Servidor API** | **Flask** (Framework), **Gunicorn** (Servidor WSGI) | Crear los *endpoints* `/predict` (API) y `/app` (Web UI). |
| **Contenerización** | **Docker** | Empaquetar la aplicación y sus dependencias para garantizar la portabilidad e independencia del entorno. |
| **Despliegue** | **Heroku** | Plataforma de *hosting* para la API. |
| **Gestión de Código** | **Git/GitHub** | Control de versiones y flujo de trabajo. |

## 2. Configuración y Dependencias

### 2.1. Librerías Requeridas y Métodos Clave

Este proyecto utiliza el siguiente conjunto de librerías Python, cruciales para el modelado y el servicio de la API.

| Librería | Propósito Principal | Métodos Clave Utilizados |
| :--- | :--- | :--- |
| **`pandas`** | Manejo y estructuración de datos. | `pd.read_csv()`, `df.iloc`, `df.drop()`, etc. |
| **`numpy`** | Operaciones numéricas y manejo de *arrays*. | `np.array()` |
| **`scikit-learn`** | Modelado y Preprocesamiento. | `LinearRegression()`, `train_test_split()`, `StandardScaler()`, `fit()`, `transform()`, `predict()`, `mean_squared_error()`. |
| **`pickle`** | Serialización de objetos Python. | `pickle.dump()` (guardar modelo/scaler), `pickle.load()` (cargar en `app.py`). |
| **`flask`** | Framework de API y Web UI. | `Flask()`, `@app.route()`, `request.get_json()`, `jsonify()`, `render_template()`. |
| **`gunicorn`** | Servidor de producción (WSGI). | Configurado vía `Procfile` y `Dockerfile`. |
| **`seaborn`, `matplotlib`** | **Visualización de datos (Solo Notebook).** | Funciones de *plotting* (`sns.heatmap()`, `plt.scatter()`, etc.). |

### 2.2. Entorno de Desarrollo (Requisitos de la Máquina Local)

Para trabajar localmente y utilizar todas las funcionalidades (Docker, pruebas de API), necesitas tener instalados:

* **[Descarga]** **Python 3.10+**
* **[Descarga]** **Docker Desktop:** Necesario para construir y ejecutar el contenedor localmente.
* **[Descarga]** **Postman (Opcional):** Para probar los *endpoints* de la API sin usar el *frontend* web.

---

## 3. Despliegue en Producción (Heroku)

El proyecto está desplegado continuamente en Heroku.

* **URL Base de la API:** `https://boston-housing-predictor-api-7fdf2917a21e.herokuapp.com`
* **Estado:** Operacional (sujeto a los límites de la cuenta de Heroku).

---

## 4. Etapa Inicial: Desarrollo y Experimentación (Jupyter Notebook)

El desarrollo inicial del modelo (EDA, limpieza, entrenamiento) se realizó en un **Jupyter Notebook**.

### 4.1. Rol del Jupyter Notebook

1.  **Exploración de Datos (EDA):** Uso intensivo de `pandas` y las librerías de visualización (`seaborn`, `matplotlib`).
2.  **Entrenamiento y Evaluación:** Definición del modelo, entrenamiento y cálculo de métricas.
3.  **Serialización:** El Notebook fue el punto donde se utilizó `pickle.dump()` para guardar el modelo (`artefacts/model.pkl`) y el *scaler* (`artefacts/scaler.pkl`), los cuales son consumidos por la API en `app.py`.

> **Nota:** Las librerías de desarrollo/visualización se **eliminaron del `requirements.txt` final** para producción para reducir el tamaño y mejorar la seguridad del contenedor Docker.

---

## 5. Uso de la API (Endpoints)

La API opera en la URL de Heroku o en el puerto `8080` dentro del contenedor (mapeado típicamente al `5000` localmente).

### A. Endpoint de la Web (Frontend)

* **Ruta:** `/app`
* **Método:** `GET`
* **Propósito:** Servir el formulario HTML (`index.html`) para probar la predicción visualmente.
* **Ejemplo Producción (Web UI):** `https://boston-housing-predictor-api-7fdf2917a21e.herokuapp.com/app`
* **Ejemplo Local:** `http://localhost:5000/app`

### B. Endpoint de Predicción (API)

* **Ruta:** `/predict`
* **Método:** `POST`
* **Propósito:** Recibir los datos de las 13 características y devolver el precio predicho.
* **Ejemplo Producción (API Call):** `https://boston-housing-predictor-api-7fdf2917a21e.herokuapp.com/predict`
* **Ejemplo Local:** `http://localhost:5000/predict` (Usado con Postman o el JavaScript del frontend).

**Estructura de la Solicitud (JSON Body):**

```json
{
    "CRIM": 0.03,
    "ZN": 20.0,
    "INDUS": 2.0,
    "CHAS": 0.0,
    "NOX": 0.4,
    "RM": 7.0,
    "AGE": 30.0,
    "DIS": 6.0,
    "RAD": 3.0,
    "TAX": 200.0,
    "PTRATIO": 15.0,
    "B": 390.0,
    "LSTAT": 5.0
}
