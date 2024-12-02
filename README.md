# Proyecto CBIR
Proyecto CBIR para clase

## Estructura

1. `preprocessor/`: Contiene scripts para tareas como redimensionar, normalizar y realizar aumentaciones de datos.
2. `feature_extraction/`: Scripts para implementar los distintos métodos de extracción de características.
3. `faiss_index/`: Incluye scripts para crear y manejar índices FAISS, y evaluar su rendimiento.
4. `ui/`: Carpeta dedicada a la interfaz de usuario proporcionada por los profesores.
5. `main.py`: Archivo de punto de entrada del proyecto que coordina las etapas principales.
6.  `requirements.txt`: Para listar todas las dependencias del proyecto (librerías como FAISS, OpenCV, etc.).
7. `config.py`: Archivo centralizado para configurar rutas, hiperparámetros, y otros valores globales.


## Ejecución

### 1. Crear y activar el entorno virtual

Crear:

- Windows:
 `python -m venv cbir_env`

Activar:

- Windows: 

`cbir_env\Scripts\activate`


### 2. Instalar librerías


### 3. Ejecutar la aplicación
El comando se debe ejecutar desde el directorio principal

` .\cbir_env\Scripts\python.exe -m streamlit run ui/app.py`

## Imágenes
Las imágenes fueron obtenidas manualmente de Google.

Enlace a las imágenes: [COMPLETAR]
