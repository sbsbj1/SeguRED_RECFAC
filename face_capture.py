import os
import cv2
import dlib
import time
import math

# Cargar el detector de rostros y el predictor de puntos faciales
detector_rostros = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')


# Función para guardar la imagen en una carpeta
def guardar_imagen(img, ruta_carpeta, tipo_persona):
    """
    Guarda la imagen en la carpeta especificada con el tipo de persona (pagador o evasor).
    """
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    nombre_archivo = f"{tipo_persona}_{timestamp}.jpg"
    ruta_completa = os.path.join(ruta_carpeta, nombre_archivo)
    cv2.imwrite(ruta_completa, img)


# Función para capturar rostro desde la cámara
def capturar_rostro_desde_camara(cap, lista_pagadores, es_comparacion=False):
    """
    Captura un frame de la cámara y devuelve el descriptor facial del rostro detectado.
    """
    ret, img = cap.read()

    if not ret:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros_detectados = detector_rostros(gray, 1)

    if len(rostros_detectados) > 0:
        for rostro in rostros_detectados:
            # Extraer la región del rostro detectado
            x, y, w, h = (rostro.left(), rostro.top(), rostro.width(), rostro.height())
            rostro_img = img[y:y+h, x:x+w]

            # Convertir el rostro a escala de grises y redimensionar a 200x200 píxeles
            rostro_img_gris = cv2.cvtColor(rostro_img, cv2.COLOR_BGR2GRAY)
            rostro_redimensionado = cv2.resize(rostro_img_gris, (200, 200))

            # Mostrar rectángulos verdes para pagadores y rojos para posibles evasores
            shape = predictor(gray, rostro)
            descriptor_facial = facerec.compute_face_descriptor(img, shape)

            if es_comparacion:
                es_pagador = comparar_con_pagadores(descriptor_facial, lista_pagadores)
                if es_pagador:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Cuadro verde
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Cuadro rojo
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Cuadro verde (pagadores)

            # Guardar imagen del rostro recortado y procesado
            return descriptor_facial, rostro_redimensionado

    return None, img


# Función para comparar con los pagadores
def comparar_con_pagadores(descriptor_facial, lista_pagadores):
    """
    Compara el descriptor facial con la lista de pagadores.
    """
    for descriptor_pagador in lista_pagadores:
        # Calcula la distancia euclidiana entre los dos descriptores faciales
        distancia = math.sqrt(sum((a - b) ** 2 for a, b in zip(descriptor_facial, descriptor_pagador)))

        # Si la distancia es menor a un umbral, consideramos que es el mismo rostro
        if distancia < 0.6:
            return True  # El rostro coincide con un pagador

    return False  # El rostro no coincide con ningún pagador
