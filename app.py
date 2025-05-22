from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from uuid import uuid4
import time
from PIL import Image
import numpy as np
import tensorflow as tf

# --- Configuración inicial ---
app = Flask(__name__)

# Carpeta donde guardaremos las subidas
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear la carpeta si no existe
if not os.path.isdir(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Cargar el modelo entrenado
modelo = tf.keras.models.load_model('modelo_entrenado.keras')

# Tamaño de imagen esperado por el modelo
TAMANO_IMG = (224, 224)

# Lista de clases en el orden de entrenamiento
CLASES = [
    "Alondra pechirroja",
    "Baudo Oropendola - Psarocolius cassini",
    "Cacique de montaña (boliviano) - Cacicus chrysonotus chrysonotus",
    "Cacique de pico amarillo",
    "Cacique de rabadilla amarilla - Cacicus cela",
    "Cacique de rabadilla escarlata - Cacicus uropygialis",
    "Oropéndola cabeza castaña - Psarocolius wagleri",
    "Oropéndola crestada - Psarocolius decumanus",
    "Oropéndola de lomo rojizo - Psarocolius angustifrons -",
    "Pradero Tortillaconchile"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    confianza = None
    nombre_archivo = None
    timestamp = int(time.time())

    if request.method == 'POST':
        archivo = request.files.get('imagen')
        if archivo and archivo.filename:
            # Generar nombre único
            ext = os.path.splitext(secure_filename(archivo.filename))[1]
            nombre_archivo = f"{uuid4().hex}{ext}"
            ruta_completa = os.path.join(UPLOAD_FOLDER, nombre_archivo)
            archivo.save(ruta_completa)

            # Preprocesar imagen
            img = Image.open(ruta_completa).convert('RGB')
            img = img.resize(TAMANO_IMG)
            array = np.expand_dims(np.array(img) / 255.0, axis=0)

            # Predicción
            resultado = modelo.predict(array)
            idx = np.argmax(resultado)
            prediccion = CLASES[idx]
            confianza = round(float(np.max(resultado)) * 100, 2)

            # Nuevo timestamp para bust cache
            timestamp = int(time.time())

    return render_template(
        'index.html',
        prediccion=prediccion,
        confianza=confianza,
        imagen=nombre_archivo,
        timestamp=timestamp
    )

if __name__ == '__main__':
    app.run(debug=True)
