# === CELDA: Graficar señal original junto con su RMS ===
import numpy as np
import matplotlib.pyplot as plt

# 1. Determinar el índice de la muestra
if 'SAMPLE_INDEX' in globals():
    idx = int(SAMPLE_INDEX)
else:
    # Usamos X_train (tus datos crudos recién cargados)
    if 'X_train' in globals():
        idx = np.random.randint(0, X_train.shape[0])
    else:
        raise RuntimeError("No se encontró X_train. Ejecuta la celda de carga de datos primero.")

# 2. Elegir la fuente de datos para la gráfica
# Preferimos X_train para ver la señal en su escala original
if 'X_train' in globals():
    source_array = X_train
    source_name = "Original"
else:
    source_array = X_kfold_s # Si no, la estandarizada
    source_name = "Estandarizada"

# 3. Extraer la muestra y sus dimensiones
sample = source_array[idx]
T, C = sample.shape

# Canal a graficar (puedes cambiarlo de 0 a 5)
channel_to_plot = 0

# 4. Calcular envolvente RMS para esta muestra específica
# La función espera (N, T, C), así que redimensionamos a (1, T, C)
sample_batch = sample.reshape((1, T, C)).astype(np.float32)

# Usamos la función ya definida en la celda anterior
# Si ENVELOPE_WIN no está definido, usamos 25 por defecto
win_size = ENVELOPE_WIN if 'ENVELOPE_WIN' in globals() else 25
env_sample = compute_envelope_rms(sample_batch, win=win_size)[0]

# 5. Graficación
plt.figure(figsize=(12, 5))

# Graficar señal bruta
plt.plot(sample[:, channel_to_plot],
         label=f'Señal {source_name} - Canal {channel_to_plot}',
         color='steelblue', alpha=0.5)

# Graficar envolvente
plt.plot(env_sample[:, channel_to_plot],
         label=f'Envolvente RMS (win={win_size})',
         color='orangered', linewidth=2)

# Estética de la gráfica
plt.title(f'Visualización de Señal EMG y su Envolvente (Muestra {idx})', fontsize=12)
plt.xlabel('Tiempo (muestras)')
plt.ylabel('Amplitud')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# 6. Información adicional en consola
print(f"Mostrando muestra índice: {idx}")
print(f"Forma de la muestra: {sample.shape}")
if 'y_train' in locals():
    print(f"Clase de esta muestra: {y_train[idx]} (Agarre {y_train[idx] + 1})")