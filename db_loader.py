import os
import cv2
import dlib

# Cargar los modelos necesarios de Dlib
detector_rostros = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('data/dlib_face_recognition_resnet_model_v1.dat')


def obtener_descriptores_facial(img):
    """
    Genera el descriptor facial de un rostro detectado en la imagen.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rostros_detectados = detector_rostros(gray, 1)

    if len(rostros_detectados) > 0:
        rostro = rostros_detectados[0]
        shape = predictor(gray, rostro)
        descriptor_facial = facerec.compute_face_descriptor(img, shape)
        return np.array(descriptor_facial)
    return None


def cargar_base_de_datos(folder='base_de_datos'):
    """
    Carga la base de datos de rostros de pasajeros pagados y genera sus descriptores faciales.
    """
    database = {}
    for file in os.listdir(folder):
        if file.endswith('.jpg') or file.endswith('.png'):
            filepath = os.path.join(folder, file)
            img = cv2.imread(filepath)
            rostro_descriptores = obtener_descriptores_facial(img)
            if rostro_descriptores is not None:
                database[file] = rostro_descriptores
    return database
