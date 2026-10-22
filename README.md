# Clasificador de lugares de la Udenar con login facial

Aplicación web hecha con Streamlit que combina dos modelos de aprendizaje profundo: un login por reconocimiento facial y un clasificador de imágenes que identifica lugares del campus de la Universidad de Nariño (Udenar). La persona inicia sesión con su rostro y luego puede tomar o subir una foto para saber qué lugar del campus es.

## Características
- Clasificación de lugares de la Udenar (por ejemplo, biblioteca o baños) a partir de una foto, con el porcentaje de confianza y los tres lugares más probables.
- Reconocimiento de rostros para el inicio de sesión: MTCNN detecta el rostro y una red neuronal convolucional (CNN) identifica a la persona registrada.
- Entrada por cámara o por subida de archivos (JPG y PNG), por lo que funciona también desde el celular.
- Umbral de confianza configurable para aceptar o rechazar el ingreso.

## Modelos utilizados
| Modelo | Arquitectura | Entrada | Archivo |
|---|---|---|---|
| Clasificador de lugares | MobileNetV2 con transfer learning | 224×224 | `modelo_clasificador.keras` |
| Reconocimiento facial | MTCNN + CNN propia | 100×100 | `modelo_faces.keras` |

Los nombres de las personas registradas están en `embeddings.pkl` y los nombres de los lugares en `clases.json`.

## Requisitos
- Python 3.10, 3.11 o 3.12 (se recomienda 3.11). TensorFlow todavía no se instala en Python 3.14.
- Las dependencias están en `requirements.txt` (Streamlit, TensorFlow, MTCNN, OpenCV, NumPy y Pillow).
- Usa la misma versión de TensorFlow con la que se entrenaron los modelos para evitar errores al cargar los archivos `.keras`.

## Instalación
```bash
git clone https://github.com/TU_USUARIO/NOMBRE_DEL_REPO.git
cd NOMBRE_DEL_REPO
python -m venv venv
```

Activa el entorno virtual:
```bash
# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Archivos necesarios
Antes de ejecutar la app, la carpeta `models/` debe contener estos cuatro archivos:
- `modelo_clasificador.keras`
- `modelo_faces.keras`
- `embeddings.pkl`
- `clases.json`

El archivo `clases.json` se genera una sola vez desde el notebook de entrenamiento:
```python
import json
clases = sorted(train_generator.class_indices, key=train_generator.class_indices.get)
with open('clases.json', 'w', encoding='utf-8') as f:
    json.dump(clases, f, ensure_ascii=False)
```

## Uso
```bash
streamlit run app.py
```

Se abrirá el navegador en `http://localhost:8501`.

1. Inicia sesión tomando o subiendo una foto donde se vea tu rostro.
2. Toma o sube una foto de un lugar del campus.
3. Consulta el lugar predicho y su porcentaje de confianza.

## Estructura
- `app.py`: aplicación principal (login facial y clasificador).
- `models/`: modelos entrenados, lista de personas registradas y nombres de las clases.
- `requirements.txt`: dependencias del proyecto.
- `README.md`: documentación del proyecto.

## Despliegue
La app puede publicarse en [Streamlit Community Cloud](https://share.streamlit.io) o en Hugging Face Spaces. Al desplegar, selecciona Python 3.11 en la configuración avanzada y usa un repositorio privado si los modelos incluyen datos de personas.

## Limitaciones
- El reconocimiento facial se entrenó con un número reducido de personas e imágenes, por lo que puede fallar con cambios de luz, cámara o ángulo. Para un sistema de seguridad real se recomienda usar un modelo de embeddings faciales como FaceNet o ArcFace.
- El clasificador de lugares depende de la cantidad y variedad de fotos de entrenamiento por lugar.

## Autora
Camila Preciado
