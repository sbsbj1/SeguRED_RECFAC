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

def comparar_con_pagadores(descriptor_facial, lista_pagadores, umbral=0.6):
    """
    Compara el descriptor facial con los registrados en la lista de pagadores.
    Devuelve True si la distancia es menor al umbral, lo que significa que la persona pagó.
    """
    for descriptor_pagador in lista_pagadores:
        distancia = np.sqrt(np.sum((np.array(descriptor_facial) - np.array(descriptor_pagador)) ** 2))
        if distancia < umbral:
            return True  # El rostro coincide con un pagador
    return False  # El rostro no coincide con ningún pagador