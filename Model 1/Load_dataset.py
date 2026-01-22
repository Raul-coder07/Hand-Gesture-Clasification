from google.colab import drive
drive.mount('/content/drive')

import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical


file_paths = [
    "/content/drive/MyDrive/Dataset_EMG_complet/dataset_5_.h5",
    "/content/drive/MyDrive/Dataset_EMG_complet/dataset_6_.h5"
]

X_list = []
y_list = []


for file_path in file_paths:
    with h5py.File(file_path, "r") as f:

        grupos = list(f.keys())   # Ej: ['G1','G2','G3','G4','G5','G6','G7']

        for idx, g in enumerate(grupos):
            grupo = f[g]

            # Iteramos todos los datasets del grupo
            for key in grupo.keys():
                data = np.array(grupo[key])   # (200,4)

                X_list.append(data)
                y_list.append(idx)           # Etiqueta según grupo


X = np.array(X_list)        # (N, 200, 4)
y = np.array(y_list)        # (N,)


print("\n📊 Cantidad total de muestras por clase:")
for i in range(len(grupos)):
    count = np.sum(y == i)
    print(f"{grupos[i]}: {count}")

print("Total muestras:", len(y))

# One hot
num_classes = 7
y = to_categorical(y, num_classes=num_classes)

#Mezclar
perm = np.random.permutation(len(X))
X = X[perm]
y = y[perm]

# Dividi 80/20
split = int(0.8 * len(X))

X_train = X[:split]
Y_train = y[:split]

X_test = X[split:]
Y_test = y[split:]


print("\nShapes finales:")
print("X:", X.shape)
print("y:", y.shape)
print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)
print("X_test:", X_test.shape)
print("Y_test:", Y_test.shape)
