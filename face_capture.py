import os
import cv2
import dlib
import time
from recognizer import es_evasor

# Cargar el detector y los modelos de Dlib
detector_rostros = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')

def guardar_imagen(img, ruta_carpeta, tipo_persona):
    """Guarda la imagen capturada en la carpeta especificada."""
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    nombre_archivo = f"{tipo_persona}_{timestamp}.jpg"
    ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
    cv2.imwrite(ruta_completa, img)
    print(f"Imagen guardada: {ruta_completa}")
    return ruta_completa

def obtener_descriptor_facial(img):
    """Genera el descriptor facial de la imagen capturada."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros_detectados = detector_rostros(gray, 1)

    if len(rostros_detectados) > 0:
        rostro = rostros_detectados[0]
        shape = predictor(gray, rostro)
        descriptor_facial = facerec.compute_face_descriptor(img, shape)
        return descriptor_facial
    return None

def capturar_foto(cap, tipo_persona, ruta_carpeta):
    """Captura una foto desde la cámara y devuelve el descriptor facial."""
    ret, img = cap.read()
    if ret:
        ruta_foto = guardar_imagen(img, ruta_carpeta, tipo_persona)
        descriptor_facial = obtener_descriptor_facial(img)
        cap.release()
        cv2.destroyAllWindows()
        return descriptor_facial
    return None

def ejecutar_comparacion(descriptor1, descriptor2, base_pagadores):
    """Compara los descriptores faciales con la base de datos de pagadores."""
    if descriptor1 and es_evasor(descriptor1, base_pagadores):
        print("Persona 1: EVASOR DETECTADO")
    else:
        print("Persona 1: PAGADOR DETECTADO")

    if descriptor2 and es_evasor(descriptor2, base_pagadores):
        print("Persona 2: EVASOR DETECTADO")
    else:
        print("Persona 2: PAGADOR DETECTADO")

if __name__ == "__main__":
    # Inicializar las cámaras
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)

    # Cargar la base de datos de pagadores
    from db_loader import cargar_base_de_datos
    base_pagadores = cargar_base_de_datos('base_de_datos/pagadores')

    # Capturar fotos desde ambas cámaras
    print("Capturando foto desde la primera cámara...")
    descriptor1 = capturar_foto(cap1, "persona1", 'data/fotos')

    print("Capturando foto desde la segunda cámara...")
    descriptor2 = capturar_foto(cap2, "persona2", 'data/fotos')

    # Ejecutar la comparación después de cerrar ambas cámaras
    ejecutar_comparacion(descriptor1, descriptor2, base_pagadores)
