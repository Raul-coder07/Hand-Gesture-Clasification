import time
import numpy as np
import tensorflow as tf
import h5py

# ============================
# 0. Configuración
# ============================
MODEL_PATH = "dataset_2/modelo_dataset_2.tflite"
DATA_PATH  = "dataset_2/datos_test_dataset_2.h5"
MEAN_PATH, STD_PATH = "dataset_2/mean_dataset_2.npy", "dataset_2/std_dataset_2.npy"
ENVELOPE_WIN = 25

# 1. Cargar recursos
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details() # Tendrá 2 entradas
output_details = interpreter.get_output_details()

mean, std = np.load(MEAN_PATH).astype(np.float32), np.load(STD_PATH).astype(np.float32)
with h5py.File(DATA_PATH, 'r') as f:
    X_raw, Y_true = np.array(f['X']).astype(np.float32), np.array(f['y']).astype(np.int32)

def get_envelope_single(X_sample, win=ENVELOPE_WIN):
    T, C = X_sample.shape[1], X_sample.shape[2]
    kernel = np.ones(win, dtype=np.float32) / win
    out = np.zeros_like(X_sample)
    for ch in range(C):
        s = np.pad(X_sample[0,:,ch]**2, win//2, mode='reflect')
        conv = np.convolve(s, kernel, mode='valid')
        out[0,:,ch] = np.sqrt(np.maximum(conv[:T], 0.0))
    return out

# 2. Ejecución
tiempos_feat, tiempos_inf, predicciones = [], [], []

for i in range(len(X_raw)):
    # Feature Extraction
    t0 = time.perf_counter()
    s_raw = (np.expand_dims(X_raw[i], 0) - mean) / std
    s_env = get_envelope_single(s_raw)
    t1 = time.perf_counter()
    
    # Inferencia TFLite (Doble entrada)
    t2 = time.perf_counter()
    # Mapear entradas según el orden del modelo
    interpreter.set_tensor(input_details[0]['index'], s_raw)
    interpreter.set_tensor(input_details[1]['index'], s_env)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    t3 = time.perf_counter()
    
    tiempos_feat.append((t1-t0)*1000); tiempos_inf.append((t3-t2)*1000)
    predicciones.append(np.argmax(output))

print(f"\nRESULTADOS MODELO 2 (TFLite Híbrido):")
print(f"Total Latencia: {np.mean(tiempos_feat) + np.mean(tiempos_inf):.4f} ms")