# app.py
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.datasets import fashion_mnist
from PIL import Image
import pandas as pd

# 1. Definimos los nombres de clase
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

# 2. Función para cargar tu modelo Pickle
@st.cache_resource
def load_model_pickle():
    # Asume que 'modelo.pkl' está en la misma carpeta que app.py
    with open("Image_classification.pkl", "rb") as f:
        modelo = pickle.load(f)
    return modelo

model = load_model_pickle()

# 3. Cargar el set de prueba (opcional, para mostrar ejemplos)
@st.cache_data
def load_test_data():
    (_, _), (test_images, test_labels) = fashion_mnist.load_data()
    # Normalizamos y aplanamos igual que en entrenamiento
    test_images_norm = test_images / 255.0
    test_images_flat = test_images_norm.reshape((test_images_norm.shape[0], 28 * 28))
    return test_images, test_images_flat, test_labels

original_test_images, test_images_flat, test_labels = load_test_data()

# 4. Título y descripción
st.title("👗 Fashion-MNIST Demo con Streamlit (modelo Pickle)")
st.write("""
    Esta app carga un modelo que fue serializado en Pickle y permite:
    - Ver una imagen de prueba al azar.
    - Mostrar la predicción del modelo y compararla con la etiqueta real.
    - Ver las probabilidades de cada clase.
""")

# 5. Sidebar: selección de índice o aleatorio
st.sidebar.header("Opciones")
usar_indice = st.sidebar.checkbox("Seleccionar índice manual", value=False)

if usar_indice:
    indice_imagen = st.sidebar.number_input(
        "Índice (0–9999):", min_value=0, max_value=9999, value=0, step=1
    )
else:
    if st.sidebar.button("Mostrar imagen aleatoria"):
        indice_imagen = np.random.randint(0, 10000)
        st.sidebar.success(f"Índice aleatorio: {indice_imagen}")
    else:
        indice_imagen = 0

# 6. Mostrar imagen de prueba
imagen = original_test_images[indice_imagen]
etiqueta_real = test_labels[indice_imagen]

st.subheader(f"Imagen de prueba (índice = {indice_imagen})")
img_pil = Image.fromarray(imagen)
st.image(img_pil, width=150, caption=f"Etiqueta real: {CLASS_NAMES[etiqueta_real]}")

# 7. Predicción
imagen_norm = imagen / 255.0
imagen_flat = imagen_norm.reshape((1, 28 * 28))  # en forma (1, 784)

# Aquí asumimos que tu modelo Pickle acepta exactamente ese vector de entrada
probabilidades = model.predict(imagen_flat)  # shape (1,10)
pred_idx = np.argmax(probabilidades, axis=1)[0]
etiqueta_pred = CLASS_NAMES[pred_idx]

st.subheader("Predicción del modelo")
st.write(f"- **Clase predicha:** {etiqueta_pred}")
st.write(f"- **Etiqueta real:** {CLASS_NAMES[etiqueta_real]}")
st.write(f"- **¿Coincide?** {'✅ Sí' if pred_idx == etiqueta_real else '❌ No'}")

# 8. Mostrar probabilidades
st.subheader("Probabilidades por clase")
probs = probabilidades.flatten()
df_probs = pd.DataFrame({
    "Clase": CLASS_NAMES,
    "Probabilidad": np.round(probs, 4)
}).sort_values(by="Probabilidad", ascending=False)

st.table(df_probs)

# 9. (Opcional) Subir tu propia imagen
st.sidebar.subheader("🔍 Cargar tu propia imagen")
uploaded_file = st.sidebar.file_uploader(
    "Sube una imagen 28×28 en escala de grises (png/jpg)", type=["png", "jpg", "jpeg"]
)
if uploaded_file is not None:
    img_user = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.sidebar.image(img_user, caption="Tu imagen (28×28)", width=100)

    arr = np.array(img_user) / 255.0
    arr_flat = arr.reshape((1, 28 * 28))
    probs_user = model.predict(arr_flat)
    idx_user = np.argmax(probs_user, axis=1)[0]
    etiqueta_user = CLASS_NAMES[idx_user]

    st.sidebar.write(f"**Predicción:** {etiqueta_user}")
    df_user = pd.DataFrame({
        "Clase": CLASS_NAMES,
        "Probabilidad": np.round(probs_user.flatten(), 4)
    }).sort_values(by="Probabilidad", ascending=False)
    st.sidebar.table(df_user)
