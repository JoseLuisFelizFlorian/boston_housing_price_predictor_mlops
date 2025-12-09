# Imagen base: Usa una versión oficial de Python mas simplifica
FROM python:3.10-slim

# Directorio de trabajo: /app
WORKDIR /app

# Copiar requerimientos e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código y artefactos
# Esto incluye: app.py, Procfile, artefacts/, y models/
COPY . /app

# Comando de inicio del servidor con Gunicorn
# Heroku mapeará el puerto 8080 a su puerto de Internet
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080", "--workers", "1"]