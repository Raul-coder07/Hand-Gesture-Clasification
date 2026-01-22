import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split
from google.colab import drive

# 1. Montar Drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

ruta_base = '/content/drive/MyDrive/a_DATASET/Dataset_EMG_complet'

# --- CONFIGURACIÓN ---
datasets_a_cargar = [5,6]  # dataset

# Mapeo de etiquetas: Convertir texto 'G1' a número 0, 'G2' a 1, etc.
class_map = {'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, 'G5': 4, 'G6': 5, 'G7':6}

# Listas temporales para acumular datos
X_temp = []
y_temp = []

print("=== CARGANDO DATOS (Datasets: 2, 3, 4, 7) ===")

for i in datasets_a_cargar:
    nombre_archivo = f'dataset_{i}_.h5'
    ruta_completa = os.path.join(ruta_base, nombre_archivo)

    if os.path.exists(ruta_completa):
        print(f"-> Leyendo: {nombre_archivo} ...")

        with h5py.File(ruta_completa, 'r') as f:
            # Iterar sobre las clases (G1, G2...)
            for nombre_clase in f.keys():
                # Verificar si la clase está en nuestro mapa (por seguridad)
                if nombre_clase in class_map:
                    label_num = class_map[nombre_clase]
                    grupo = f[nombre_clase]

                    # Iterar sobre cada muestra dentro de la clase
                    for nombre_muestra in grupo.keys():
                        data = grupo[nombre_muestra][()] # Cargar matriz (200, 4)

                        # Asegurar las demensiones
                        if data.shape == (200, 4):
                            X_temp.append(data)
                            y_temp.append(label_num)
                        else:
                            print(f"   [Ignorado] {nombre_muestra} tiene forma incorrecta: {data.shape}")
    else:
        print(f"X ERROR: No se encontró {nombre_archivo}")

# --- CONCATENAR TODOS LOS DATASET ---
print("\nConvirtiendo listas a Arreglos NumPy...")
X_total = np.array(X_temp)
y_total = np.array(y_temp)

print(f"Dimensiones totales cargadas:")
print(f"  X (Datos): {X_total.shape} -> (Muestras, Filas, Canales)")
print(f"  y (Etiquetas): {y_total.shape}")

# --- DIVISIÓN ESTRATIFICADA (70% Train - 30% Test) ---
print("\nRealizando división estratificada (70/30)...")

# stratify=y_total asegura que si hay 100 muestras de G1 en total,
# 70 vayan a train y 30 a test, manteniendo el balance.
X_train, X_test, y_train, y_test = train_test_split(
    X_total, y_total,
    test_size=0.30,
    random_state=42,
    stratify=y_total
)

print("\n=== ¡DATOS LISTOS PARA ENTRENAR! ===")
print(f"ENTRENAMIENTO (70%): X={X_train.shape}, y={y_train.shape}")
print(f"TESTEO (30%):        X={X_test.shape},  y={y_test.shape}")

# Verificación rápida de clases en Test
unique, counts = np.unique(y_test, return_counts=True)
print("\nDistribución de clases en el set de Test (debe ser equilibrada):")
for u, c in zip(unique, counts):
    print(f"  Clase {u}: {c} muestras")