"""App Udenar: login facial + clasificador de lugares del campus."""
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

BASE = Path(__file__).parent
MODELS = BASE / "models"
CLASSIFIER_PATH = MODELS / "modelo_clasificador.keras"
FACES_PATH = MODELS / "modelo_faces.keras"
PERSONS_PATH = MODELS / "embeddings.pkl"   # lista de nombres (la guardó tu notebook)
CLASSES_PATH = MODELS / "clases.json"      # nombres de lugares (ver README)

FACE_THRESHOLD = 0.60   # confianza mínima para aceptar el login
DEBUG = True            # muestra diagnóstico en el login (pon False al terminar)

st.set_page_config(page_title="Udenar · Clasificador", page_icon="🏛️")


# ---------- Carga de modelos (una sola vez) ----------
@st.cache_resource
def load_keras(path: str):
    import tensorflow as tf
    return tf.keras.models.load_model(path)


@st.cache_resource
def load_detector():
    from mtcnn import MTCNN
    return MTCNN()


@st.cache_data
def load_persons():
    with open(PERSONS_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_classes():
    with open(CLASSES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------- Utilidades ----------
def get_image(key: str):
    """Cámara o archivo; devuelve una imagen PIL RGB o None."""
    tab_cam, tab_file = st.tabs(["📷 Cámara", "🖼️ Subir foto"])
    with tab_cam:
        cam = st.camera_input("Toma la foto", key=f"cam_{key}")
    with tab_file:
        up = st.file_uploader("Elige una imagen", type=["jpg", "jpeg", "png"], key=f"up_{key}")
    src = cam or up
    return Image.open(src).convert("RGB") if src else None


def crop_face(img: Image.Image):
    """Detecta el rostro más grande y lo devuelve listo para el modelo (1,100,100,3)."""
    arr = np.array(img)  # RGB
    faces = load_detector().detect_faces(arr)
    if not faces:
        return None
    f = max(faces, key=lambda d: d["box"][2] * d["box"][3])
    x, y, w, h = f["box"]
    x, y = max(x, 0), max(y, 0)
    face = cv2.resize(arr[y:y + h, x:x + w], (100, 100)) / 255.0
    return face.reshape(1, 100, 100, 3).astype("float32")


# ---------- Pantalla de login ----------
def login_page():
    st.title("🔐 Ingreso con rostro")
    st.write("Toma o sube una foto donde se vea tu cara.")
    img = get_image("login")
    if img is None:
        return
    face = crop_face(img)
    if face is None:
        st.error("No se encontró ningún rostro. Intenta con mejor luz y de frente.")
        if DEBUG:
            st.caption(f"Tamaño de la imagen recibida: {img.size}")
        return
    pred = load_keras(str(FACES_PATH)).predict(face, verbose=0)[0]
    persons = load_persons()
    if DEBUG:
        with st.expander("Diagnóstico", expanded=True):
            st.image(face[0], caption="Rostro que recibe el modelo (100x100)", width=150)
            for i in np.argsort(pred)[::-1][:3]:
                st.write(f"{persons[i]}: {pred[i]:.1%}")
    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    if conf >= FACE_THRESHOLD:
        st.session_state.user = persons[idx]
        st.rerun()
    else:
        st.error(f"No te reconocí (confianza {conf:.0%}). Intenta de nuevo.")


# ---------- Pantalla del clasificador ----------
def classifier_page():
    st.title("🏛️ ¿Qué lugar de la Udenar es?")
    st.write("Toma o sube una foto de un lugar del campus.")
    img = get_image("clasif")
    if img is None:
        return
    st.image(img, use_container_width=True)

    model = load_keras(str(CLASSIFIER_PATH))
    clases = load_classes()
    x = np.expand_dims(np.array(img.resize((224, 224))) / 255.0, 0).astype("float32")
    pred = model.predict(x, verbose=0)[0]
    if len(pred) != len(clases):
        st.error(f"El modelo tiene {len(pred)} clases pero clases.json tiene {len(clases)}.")
        return

    top = np.argsort(pred)[::-1][:3]
    st.success(f"**{clases[top[0]]}** — {pred[top[0]]:.0%}")
    for i in top[1:]:
        st.write(f"{clases[i]}: {pred[i]:.0%}")
        st.progress(float(pred[i]))


# ---------- Main ----------
missing = [p.name for p in (CLASSIFIER_PATH, FACES_PATH, PERSONS_PATH, CLASSES_PATH) if not p.exists()]
if missing:
    st.error("Faltan archivos en la carpeta `models/`: " + ", ".join(missing))
    st.stop()

if "user" not in st.session_state:
    login_page()
else:
    with st.sidebar:
        st.write(f"👤 **{st.session_state.user}**")
        if st.button("Cerrar sesión"):
            del st.session_state["user"]
            st.rerun()
    classifier_page()
