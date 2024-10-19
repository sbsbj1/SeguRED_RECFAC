import cv2
import dlib
import os
import numpy as np

# Cargar el detector de rostros y el predictor
detector_rostros = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')

# Carpetas para almacenar imágenes
IMG_PATH_PAGADORES = 'base_de_datos/pagadores'
IMG_PATH_EVASORES = 'base_de_datos/evasores'

# Almacenaremos el último descriptor capturado para compararlo
ultimo_descriptor_pagador = None
ultimo_descriptor_evasor = None


def guardar_imagen(img, ruta, tipo):
    """
    Guarda la imagen redimensionada en la carpeta correspondiente.
    """
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    filename = f"{tipo}_{len(os.listdir(ruta))}.jpg"
    cv2.imwrite(os.path.join(ruta, filename), img)


def capturar_rostro_desde_camara(cap, lista_pagadores, es_comparacion=False):
    """
    Captura un frame de la cámara, redimensiona los rostros a 200x200 píxeles.
    Si es_comparacion=True, compara el rostro con los registrados en la lista de pagadores.
    """
    global ultimo_descriptor_pagador, ultimo_descriptor_evasor

    ret, img = cap.read()

    if not ret:
        return None, img

    # Detectar los rostros
    rostros_detectados = detector_rostros(img, 1)
    if len(rostros_detectados) > 0:
        for rostro in rostros_detectados:
            shape = predictor(img, rostro)
            descriptor_facial = facerec.compute_face_descriptor(img, shape)

            # Redimensionar la imagen a 200x200 píxeles
            x, y, w, h = (rostro.left(), rostro.top(), rostro.width(), rostro.height())
            rostro_recortado = cv2.resize(img[y:y + h, x:x + w], (200, 200))

            # Comparar similitud con el último rostro capturado
            if es_comparacion:
                es_pagador = comparar_con_pagadores(descriptor_facial, lista_pagadores)
                if es_pagador:
                    if ultimo_descriptor_pagador is None or comparar_similitud(descriptor_facial,
                                                                               ultimo_descriptor_pagador) > 0.5:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Cuadro verde
                        guardar_imagen(rostro_recortado, IMG_PATH_PAGADORES, 'pagador')
                        ultimo_descriptor_pagador = descriptor_facial  # Actualizar el último descriptor de pagador
                else:
                    if ultimo_descriptor_evasor is None or comparar_similitud(descriptor_facial,
                                                                              ultimo_descriptor_evasor) > 0.5:
                        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Cuadro rojo
                        guardar_imagen(rostro_recortado, IMG_PATH_EVASORES, 'evasor')
                        ultimo_descriptor_evasor = descriptor_facial  # Actualizar el último descriptor de evasor
            else:
                if ultimo_descriptor_pagador is None or comparar_similitud(descriptor_facial,
                                                                           ultimo_descriptor_pagador) > 0.5:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Cuadro verde
                    guardar_imagen(rostro_recortado, IMG_PATH_PAGADORES, 'pagador')
                    ultimo_descriptor_pagador = descriptor_facial  # Actualizar el último descriptor de pagador

        return descriptor_facial, img

    return None, img


def comparar_similitud(descriptor1, descriptor2):
    """
    Compara dos descriptores faciales y devuelve un valor de similitud (distancia).
    Un valor mayor a 0.2 se considera lo suficientemente diferente para capturar.
    """
    distancia = np.linalg.norm(np.array(descriptor1) - np.array(descriptor2))
    return distancia


def comparar_con_pagadores(descriptor_facial, lista_pagadores, umbral=0.6):
    """
    Compara el descriptor facial con los registrados en la lista de pagadores.
    Devuelve True si la distancia es menor al umbral.
    """
    for descriptor_pagador in lista_pagadores:
        distancia = np.sqrt(np.sum((np.array(descriptor_facial) - np.array(descriptor_pagador)) ** 2))
        if distancia < umbral:
            return True
    return False


def registrar_pagadores():
    """
    Captura los rostros de los pagadores y los guarda en la carpeta correspondiente.
    """
    cap_puerta = cv2.VideoCapture(0)  # Cámara de la puerta principal
    lista_pagadores = []

    # Crear una ventana de captura y ajustar su tamaño
    cv2.namedWindow('Cámara Puerta Principal - Registro de Pagadores', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cámara Puerta Principal - Registro de Pagadores', 640, 480)

    while True:
        descriptor_facial, img = capturar_rostro_desde_camara(cap_puerta, lista_pagadores, es_comparacion=False)

        # Mostrar la captura de la cámara de la puerta
        if img is not None:
            cv2.imshow('Cámara Puerta Principal - Registro de Pagadores', img)

        # Presiona 'q' para cambiar a la cámara de control o 'Esc' para salir
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == 27:  # Código ASCII para 'Esc'
            cap_puerta.release()
            cv2.destroyAllWindows()
            return  # Salir del sistema

    cap_puerta.release()
    cv2.destroyAllWindows()

    # Cambiar a la cámara de control después de registrar los pagadores
    comparar_pasajeros(lista_pagadores)


def comparar_pasajeros(lista_pagadores):
    """
    Compara los rostros capturados en la cámara general con los pagadores registrados.
    """
    cap_general = cv2.VideoCapture(0)  # Cámara general dentro del bus

    # Crear una ventana de captura para la cámara general y ajustar su tamaño
    cv2.namedWindow('Cámara General - Control de Pasajeros', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Cámara General - Control de Pasajeros', 640, 480)

    while True:
        descriptor_facial, img = capturar_rostro_desde_camara(cap_general, lista_pagadores, es_comparacion=True)

        # Mostrar la captura de la cámara general
        if img is not None:
            cv2.imshow('Cámara General - Control de Pasajeros', img)

        # Presiona 'Esc' para salir
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Código ASCII para 'Esc'
            cap_general.release()
            cv2.destroyAllWindows()
            return  # Salir del sistema


if __name__ == '__main__':
    print("Simulación de registro de pagadores y comparación en la cámara de control.")
    registrar_pagadores()
