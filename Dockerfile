# 1. Imagen base: Usamos una versión oficial de Python
FROM python:3.10-slim

# 2. Directorio de trabajo: /app
WORKDIR /app

# 3. Copiar requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto del código y artefactos
# Esto incluye: app.py, Procfile, artefacts/, y models/
COPY . /app

# 5. Comando de inicio del servidor con Gunicorn
# Heroku mapeará el puerto 8080 a su puerto de Internet
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1"]