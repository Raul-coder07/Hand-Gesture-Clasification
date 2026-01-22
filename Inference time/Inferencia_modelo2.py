import time
import numpy as np
import tensorflow as tf
import h5py

# ============================
# 0. Configuración de Rutas
# ============================
# Se asume que todos los archivos están dentro de la carpeta "dataset_2"
MODEL_PATH = "dataset_2/modelo_dataset_2.keras"
DATA_PATH  = "dataset_2/datos_test_dataset_2.h5"  # Archivo H5 con X e y
MEAN_PATH  = "dataset_2/mean_dataset_2.npy"
STD_PATH   = "dataset_2/std_dataset_2.npy"
OUT_TIMES_NPY = "keras_inference_times_dataset_2.npy"

ENVELOPE_WIN = 25

# ============================
# 1. Cargar Parámetros y Modelo
# ============================
print(f"Cargando modelo desde: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Cargando parámetros de estandarización...")
mean_ch = np.load(MEAN_PATH).astype(np.float32)
std_ch  = np.load(STD_PATH).astype(np.float32)

print(f"Cargando datos de prueba desde: {DATA_PATH}")
with h5py.File(DATA_PATH, 'r') as f:
    X_test_raw = np.array(f['X']).astype(np.float32)
    Y_test     = np.array(f['y']).astype(np.int32)

print(f"Datos cargados correctamente. X_raw: {X_test_raw.shape}")

# ============================
# 2. Funciones de Preprocesamiento (Optimizado para 1 muestra)
# ============================
def get_envelope_single(X_sample, win=ENVELOPE_WIN):
    """
    Versión optimizada de compute_envelope_rms para una sola muestra (1, T, C)
    """
    T, C = X_sample.shape[1], X_sample.shape[2]
    kernel = np.ones(win, dtype=np.float32) / win
    out = np.zeros_like(X_sample)
    pad = win // 2

    for ch in range(C):
        sq = X_sample[0, :, ch]**2
        # Aplicar padding reflect para mantener consistencia con el entrenamiento
        s = np.pad(sq, pad_width=pad, mode='reflect')
        conv = np.convolve(s, kernel, mode='valid')
        out[0, :, ch] = np.sqrt(np.maximum(conv[:T], 0.0))
    return out

# ============================
# 3. Medición Detallada
# ============================
def medir_rendimiento_completo(model, X, Y, mean, std):
    n = len(X)
    tiempos_feat = []  # Estandarización + Envelope
    tiempos_inf  = []  # Solo el paso por la red neuronal
    predicciones = []

    print(f"\nProcesando {n} muestras una a una (Dataset 2)...")

    # Warm-up (Importante para GPU/TF Graph)
    if n > 0:
        dummy_x = np.expand_dims(X[0], axis=0)
        _ = model([dummy_x, dummy_x], training=False)

    for i in range(n):
        sample_raw = np.expand_dims(X[i], axis=0) # (1, 200, 4)

        # --- INICIO FEATURE EXTRACTION ---
        t0 = time.perf_counter()
        
        # 1. Estandarizar
        sample_s = (sample_raw - mean) / std
        
        # 2. Calcular Envelope
        sample_env = get_envelope_single(sample_s)
        
        t1 = time.perf_counter()
        # --- FIN FEATURE EXTRACTION ---

        # --- INICIO INFERENCIA ---
        t2 = time.perf_counter()
        # El modelo híbrido recibe [raw_in, env_in]
        output = model([sample_s, sample_env], training=False)
        t3 = time.perf_counter()
        # --- FIN INFERENCIA ---

        tiempos_feat.append((t1 - t0) * 1000.0)
        tiempos_inf.append((t3 - t2) * 1000.0)
        predicciones.append(np.argmax(output.numpy()))

        if i % 500 == 0 and i > 0:
            print(f"Progreso: {i}/{n}...")

    return np.array(tiempos_feat), np.array(tiempos_inf), np.array(predicciones)

# Ejecutar medición
t_feat, t_inf, y_pred = medir_rendimiento_completo(model, X_test_raw, Y_test, mean_ch, std_ch)

# ============================
# 4. Reporte de Resultados
# ============================
accuracy = np.mean(y_pred == Y_test) * 100
total_per_sample = t_feat + t_inf

print("\n" + "="*40)
print(f"REPORTES DE TIEMPOS (dataset_2) (ms)")
print("="*40)
print(f"Accuracy final:          {accuracy:.2f}%")
print("-" * 40)
print(f"FEATURE EXTRACTION (RMS): {np.mean(t_feat):.4f} ms")
print(f"INFERENCIA MODELO:       {np.mean(t_inf):.4f} ms")
print(f"TOTAL POR MUESTRA:       {np.mean(total_per_sample):.4f} ms")
print("-" * 40)
print(f"Desviación Estándar Tot: {np.std(total_per_sample):.4f} ms")
print("="*40)

# Guardar resultados detallados
np.save(OUT_TIMES_NPY, {"feat": t_feat, "inf": t_inf, "total": total_per_sample})
print(f"\nResultados guardados en {OUT_TIMES_NPY}")