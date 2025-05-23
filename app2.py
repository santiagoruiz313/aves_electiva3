import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from uuid import uuid4
import time

# Parámetros
TAMANO_IMG = (224, 224)

# Clases
CLASES = [
    "Alondra pechirroja",
    "Baudo Oropendola - Psarocolius cassini",
    "Black-crested Warbler - Myiothlypis nigrocristata",
    "Buff-rumped Warbler - Myiothlypis fulvicauda",
    "Cacique de montaña (boliviano) - Cacicus chrysonotus chrysonotus",
    "Cacique de pico amarillo",
    "Cacique de rabadilla amarilla - Cacicus cela",
    "Cacique de rabadilla escarlata - Cacicus uropygialis",
    "Canada Warbler - Cardellina canadensis",
    "Citrine Warbler - Myiothlypis luteoviridis",
    "Golden-crowned Warbler - Basileuterus culicivorus",
    "Oropéndola cabeza castaña - Psarocolius wagleri",
    "Oropéndola crestada - Psarocolius decumanus",
    "Oropéndola de lomo rojizo - Psarocolius angustifrons",
    "Pradero Tortillaconchile",
    "Slate-throated Redstart - Myioborus miniatus",
    "Three-striped Warbler - Basileuterus tristriatus"
]

# Información de aves
BIRD_INFO = {
    'Alondra pechirroja': 'La alondra pechirroja es un ave pequeña y territorial que se destaca por su canto melódico al amanecer.',
    'Baudo Oropendola - Psarocolius cassini': 'Ave llamativa de plumaje negro con pico amarillo. Habita selvas húmedas.',
    'Black-crested Warbler - Myiothlypis nigrocristata': 'Pequeño pájaro de bosques andinos, con cresta negra.',
    'Buff-rumped Warbler - Myiothlypis fulvicauda': 'Habita orillas de ríos. Tiene rabadilla amarilla y cantos agudos.',
    'Cacique de montaña (boliviano) - Cacicus chrysonotus chrysonotus': 'Habita montañas húmedas. Es negro con partes amarillas.',
    'Cacique de pico amarillo': 'Reconocido por su plumaje oscuro y pico amarillo brillante.',
    'Cacique de rabadilla amarilla - Cacicus cela': 'Ave gregaria con plumaje negro y parte inferior amarilla.',
    'Cacique de rabadilla escarlata - Cacicus uropygialis': 'Tiene una mancha roja en la espalda. Muy vocal.',
    'Canada Warbler - Cardellina canadensis': 'Ave migratoria con pecho amarillo y collar negro.',
    'Citrine Warbler - Myiothlypis luteoviridis': 'Plumaje amarillo-verdoso. Activo en el sotobosque andino.',
    'Golden-crowned Warbler - Basileuterus culicivorus': 'Se distingue por su corona amarilla.',
    'Oropéndola cabeza castaña - Psarocolius wagleri': 'Plumaje negro, cabeza marrón y pico amarillo.',
    'Oropéndola crestada - Psarocolius decumanus': 'Cresta notable y plumaje negro brillante.',
    'Oropéndola de lomo rojizo - Psarocolius angustifrons': 'Espalda marrón rojiza. Forma colonias ruidosas.',
    'Pradero Tortillaconchile': 'Especie poco documentada, canto distintivo.',
    'Slate-throated Redstart - Myioborus miniatus': 'Garganta oscura, partes inferiores rojizas. Muy activo.',
    'Three-striped Warbler - Basileuterus tristriatus': 'Muy vocal. Tiene rayas en la cabeza y pecho amarillo.'
}

# Cargar el modelo
modelo = tf.keras.models.load_model('model_VGG16_v4_os.keras')

# Interfaz de usuario
st.set_page_config(page_title="Clasificador de Aves", layout="centered")
st.title("Clasificación de Aves 🐦")
st.write("Sube una imagen de un ave y obtén las 3 predicciones más probables con su información.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
lat = st.text_input("Latitud (opcional)")
lon = st.text_input("Longitud (opcional)")

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen subida", use_column_width=True)
    img = image.resize(TAMANO_IMG)
    array = np.expand_dims(np.array(img) / 255.0, axis=0)

    resultado = modelo.predict(array)[0]
    idxs = np.argsort(resultado)[-3:][::-1]
    top_k = [(CLASES[i], round(float(resultado[i]) * 100, 2)) for i in idxs]
    
    st.subheader("Top 3 predicciones")
    for i, (name, prob) in enumerate(top_k):
        st.markdown(f"**{i+1}. {name}** — {prob}%")
        st.markdown(f"*{BIRD_INFO.get(name, 'Información no disponible.')}*")
    
    st.caption(f"🗺️ Coordenadas ingresadas: {lat}, {lon}")
