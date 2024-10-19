import numpy as np

def comparar_con_base_de_datos(rostro_descriptor, base_de_datos, umbral=0.6):
    """
    Compara el descriptor del rostro capturado con los descriptores faciales en la base de datos.
    """
    for nombre, descriptor in base_de_datos.items():
        # Calcular la distancia euclidiana entre el descriptor capturado y el de la base de datos
        distancia = np.linalg.norm(descriptor - rostro_descriptor)
        if distancia < umbral:
            return nombre, distancia
    return None, None
