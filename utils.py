import os
import time

def limpiar_imagenes_antiguas(folder, max_time):
    """
    Elimina las imágenes de una carpeta si exceden un tiempo máximo de almacenamiento.
    """
    current_time = time.time()
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            file_creation_time = os.path.getctime(filepath)
            if current_time - file_creation_time > max_time:
                print(f"Eliminando imagen antigua: {filepath}")
                os.remove(filepath)
