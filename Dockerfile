# --- Imagen base ligera con Python 3.10 ---
FROM python:3.10-slim

# --- Instalar dependencias del sistema necesarias para OpenCV, Mediapipe y PyTorch ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# --- Directorio de trabajo ---
WORKDIR /app
COPY requirements.txt /app

# --- Actualizar pip y instalar dependencias de Python ---
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# --- Copiar toda la aplicación ---
COPY . /app

# --- Puerto y comando de inicio ---
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
