import time
import numpy as np
import tensorflow as tf
import os

# ============================
# 0. Configuración
# ============================
# Se asume que los archivos están en una carpeta llamada "dataset_2"
MODEL_PATH = "dataset_2/modelo_dataset_2.h5"
X_TEST_PATH = "dataset_2/X_test_dataset_2.npy"
Y_TEST_PATH = "dataset_2/Y_test_dataset_2.npy"

OUT_TIMES_NPY = "keras_inference_times_dataset_2.npy"

# ============================
# 1. Cargar modelo Keras y datos
# ============================
print(f"Cargando modelo desde: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
# model.summary()  # Descomentar si se quiere ver el resumen

print("Cargando datos de test...")
X_test = np.load(X_TEST_PATH)
Y_test = np.load(Y_TEST_PATH)

print("Modelo y datos cargados correctamente.")
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)

# ============================
# 2. Definir mapeo de clases
# ============================
clases = {
    0: "G7",
    1: "G6",
    2: "G5",
    3: "G4",
    4: "G3",
    5: "G2",
    6: "G1"
}

# ============================
# 3. Función para medir inferencia muestra a muestra
# ============================
def medir_inferencia_keras(model, X, Y, max_samples=None, verbose=True):
    """
    Itera muestra a muestra usando el modelo Keras y mide tiempos.
    Devuelve diccionario con estadísticas y pred/true arrays.
    """
    n = len(X) if max_samples is None else min(len(X), max_samples)
    tiempos_ms = np.zeros(n, dtype=np.float64)
    y_pred_idx = np.zeros(n, dtype=np.int32)
    y_true_idx = np.zeros(n, dtype=np.int32)

    # Warm-up: una pasada rápida con la primera muestra para inicializar grafos
    if n > 0:
        sample_w = np.expand_dims(X[0], axis=0).astype(np.float32)
        _ = model.predict(sample_w, batch_size=1, verbose=0)

    for i in range(n):
        sample = np.expand_dims(X[i], axis=0).astype(np.float32)  # (1, timesteps, channels)

        # Etiqueta real
        real_idx = int(np.argmax(Y[i]))
        y_true_idx[i] = real_idx
        real_class = clases.get(real_idx, str(real_idx))

        # Medir inferencia
        t0 = time.perf_counter()
        preds = model.predict(sample, batch_size=1, verbose=0)
        t1 = time.perf_counter()

        # Procesar salida
        pred_idx = int(np.argmax(preds, axis=1)[0])
        y_pred_idx[i] = pred_idx
        pred_class = clases.get(pred_idx, str(pred_idx))

        elapsed_ms = (t1 - t0) * 1000.0
        tiempos_ms[i] = elapsed_ms

        if verbose:
            print(f"Muestra {i:04d}: Tiempo={elapsed_ms:7.3f} ms | Pred={pred_class} (idx {pred_idx}) | Real={real_class} (idx {real_idx})")

    # Estadísticas
    total_ms = tiempos_ms.sum()
    mean_ms = tiempos_ms.mean()
    median_ms = np.median(tiempos_ms)
    std_ms = tiempos_ms.std()
    acc = np.mean(y_pred_idx == y_true_idx)

    print("\n=== Resumen de inferencia (Dataset 2) ===")
    print(f"N muestras procesadas : {n}")
    print(f"Tiempo total (s)      : {total_ms/1000.0:.4f} s")
    print(f"Tiempo total (ms)     : {total_ms:.3f} ms")
    print(f"Tiempo promedio (ms)  : {mean_ms:.3f} ms")
    print(f"Mediana (ms)          : {median_ms:.3f} ms")
    print(f"Desviación estándar   : {std_ms:.3f} ms")
    print(f"Accuracy (muestras)   : {acc*100:.2f}%")

    return {
        "tiempos_ms": tiempos_ms,
        "total_ms": total_ms,
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "std_ms": std_ms,
        "y_pred_idx": y_pred_idx,
        "y_true_idx": y_true_idx,
        "accuracy": acc
    }

# ============================
# 4. Ejecutar medición
# ============================
# Se ejecuta sobre dataset_2
results = medir_inferencia_keras(model, X_test, Y_test, max_samples=None, verbose=True)

# ============================
# 5. Guardar tiempos a disco
# ============================
np.save(OUT_TIMES_NPY, results["tiempos_ms"])
print(f"\nTiempos guardados en {OUT_TIMES_NPY}")