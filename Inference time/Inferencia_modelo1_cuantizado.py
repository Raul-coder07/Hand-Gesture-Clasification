import time
import numpy as np
import tensorflow as tf

# ============================
# 0. Configuración
# ============================
MODEL_PATH = "dataset_2/modelo_dataset_2.tflite"
X_TEST_PATH = "dataset_2/X_test_dataset_2.npy"
Y_TEST_PATH = "dataset_2/Y_test_dataset_2.npy"
OUT_TIMES_NPY = "tflite_inference_times_dataset_2.npy"

# ============================
# 1. Cargar Intérprete TFLite y Datos
# ============================
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

X_test = np.load(X_TEST_PATH).astype(np.float32)
Y_test = np.load(Y_TEST_PATH)

print(f"Modelo TFLite cargado. Input shape esperado: {input_details[0]['shape']}")

# ============================
# 2. Medición de Inferencia
# ============================
def medir_tflite(X, Y):
    n = len(X)
    tiempos = []
    preds = []
    
    for i in range(n):
        sample = np.expand_dims(X[i], axis=0) # (1, T, C)
        
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], sample)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        t1 = time.perf_counter()
        
        tiempos.append((t1 - t0) * 1000.0)
        preds.append(np.argmax(output_data))
        
    return np.array(tiempos), np.array(preds)

t_exec, y_pred = medir_tflite(X_test, Y_test)
acc = np.mean(y_pred == np.argmax(Y_test, axis=1)) * 100

print(f"\nRESULTADOS MODELO 1 (TFLite):")
print(f"Accuracy: {acc:.2f}% | Latencia Media: {np.mean(t_exec):.4f} ms")

np.save(OUT_TIMES_NPY, t_exec)