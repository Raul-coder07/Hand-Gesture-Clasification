
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, initializers, regularizers
from tensorflow.keras.utils import to_categorical

#COMPATIBILIDAD CON CODIGO ANTERIOR
X_kfold_raw = X_train  # Tu set de entrenamiento previo
y_kfold_int = y_train  # Tus etiquetas de entrenamiento
X_holdout_raw = X_test # Tu set de test previo
y_holdout_int = y_test # Tus etiquetas de test
# Como no tenemos un tercer set externo real, usaremos X_test como dummy para que no falle
X_test_public = X_test
y_test_int_from_labels_file = None


# ==========================================
# CONFIGURACIÓN Y PREPROCESAMIENTO
# ==========================================

# 1. One-hot encoding
num_classes = int(np.max(y_kfold_int) + 1)
y_kfold_hot = to_categorical(y_kfold_int, num_classes=num_classes)
y_holdout_hot = to_categorical(y_holdout_int, num_classes=num_classes)

if y_test_int_from_labels_file is not None:
    y_test_hot = to_categorical(y_test_int_from_labels_file, num_classes=num_classes)
else:
    y_test_hot = None

print("\n=== Shapes Detectados ===")
print("  X_kfold_raw:", X_kfold_raw.shape)
print("  y_kfold_hot:", y_kfold_hot.shape)

# 2. Semillas
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

# 3. Parámetros Globales
BATCH_SIZE = 64
EPOCHS = 120
ENVELOPE_WIN = 25
DROPOUT = 0.30
L2 = 1e-5

# 4. Utilidades: Envelope y Estandarización
def compute_envelope_rms(X, win=ENVELOPE_WIN):
    N, T, C = X.shape
    if win <= 1: return np.abs(X)

    kernel = np.ones(win, dtype=np.float32) / win
    out = np.zeros_like(X, dtype=np.float32)
    pad = win // 2

    for ch in range(C):
        sq = X[:,:,ch].astype(np.float32)**2
        # Función auxiliar para aplicar convolución por fila
        def conv_row(row):
            s = np.pad(row, pad_width=pad, mode='reflect')
            conv = np.convolve(s, kernel, mode='valid')
            return np.sqrt(np.maximum(conv, 0.0))

        # Aplicar a todo el batch de ese canal
        out[:,:,ch] = np.apply_along_axis(conv_row, 1, sq)

    # Recorte o ajuste final para asegurar shape (a veces la convalidación varía por 1 frame)
    if out.shape[1] != T:
        out = out[:, :T, :]
    return out

def channel_standardize_train_test(X_train_local, X_test_local):
    means = X_train_local.mean(axis=(0,1), keepdims=True)
    stds  = X_train_local.std(axis=(0,1), keepdims=True) + 1e-8
    X_train_s = (X_train_local - means)/stds
    X_test_s  = (X_test_local  - means)/stds
    return X_train_s.astype(np.float32), X_test_s.astype(np.float32), means, stds

# 5. Aplicar Preprocesamiento
print("\nProcesando datos (Standardization + Envelope)...")
# A. Estandarizar
X_kfold_s, X_holdout_s, mean_ch, std_ch = channel_standardize_train_test(X_kfold_raw, X_holdout_raw)
X_test_s = (X_test_public - mean_ch)/std_ch
X_test_s = X_test_s.astype(np.float32)

# B. Calcular Envelope
X_env_kfold = compute_envelope_rms(X_kfold_s, win=ENVELOPE_WIN)
X_env_holdout = compute_envelope_rms(X_holdout_s, win=ENVELOPE_WIN)
X_env_test  = compute_envelope_rms(X_test_s,  win=ENVELOPE_WIN)

print("Datos listos para el modelo:")
print("  Raw Input:", X_kfold_s.shape)
print("  Env Input:", X_env_kfold.shape)

# ==========================================
# DEFINICIÓN DEL MODELO (SIMPLIFICADO)
# ==========================================
KERNEL_INIT = initializers.GlorotUniform(seed=SEED)
BIAS_INIT   = initializers.Zeros()

def sep_res_block_1d(x, filters, kernel=3, l2=L2):
    shortcut = x
    x = layers.SeparableConv1D(filters, kernel, padding='same',
                               depthwise_initializer=KERNEL_INIT,
                               pointwise_initializer=KERNEL_INIT,
                               bias_initializer=BIAS_INIT,
                               depthwise_regularizer=regularizers.l2(l2),
                               pointwise_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)

    x = layers.SeparableConv1D(filters, kernel, padding='same',
                               depthwise_initializer=KERNEL_INIT,
                               pointwise_initializer=KERNEL_INIT,
                               bias_initializer=BIAS_INIT,
                               depthwise_regularizer=regularizers.l2(l2),
                               pointwise_regularizer=regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != x.shape[-1]:
        shortcut = layers.Conv1D(filters, 1, padding='same', kernel_initializer=KERNEL_INIT)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('gelu')(x)
    return x

def build_hybrid_two_input(timesteps, channels, num_classes,
                           # === PARÁMETROS FIJOS (Sin lógica condicional) ===
                           conv_f1=28,
                           conv_f2=56,
                           env_f=16,
                           dense_proj=160,
                           proj_units=160,
                           dropout=DROPOUT):

    # 1. Rama RAW
    inp_raw = layers.Input(shape=(timesteps, channels), name='raw_in')
    x = layers.SeparableConv1D(conv_f1, 3, padding='same',
                               depthwise_initializer=KERNEL_INIT,
                               pointwise_initializer=KERNEL_INIT,
                               bias_initializer=BIAS_INIT)(inp_raw)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)

    x = layers.SeparableConv1D(conv_f1, 3, padding='same',
                               depthwise_initializer=KERNEL_INIT,
                               pointwise_initializer=KERNEL_INIT,
                               bias_initializer=BIAS_INIT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.MaxPool1D(2)(x)

    x = layers.SeparableConv1D(conv_f2, 3, padding='same',
                               depthwise_initializer=KERNEL_INIT,
                               pointwise_initializer=KERNEL_INIT,
                               bias_initializer=BIAS_INIT)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    raw_feat = layers.Dense(dense_proj, activation='gelu')(x)
    raw_feat = layers.Dropout(dropout)(raw_feat)

    # 2. Rama ENVELOPE
    inp_env = layers.Input(shape=(timesteps, channels), name='env_in')
    e = layers.SeparableConv1D(env_f, 3, padding='same',
                               depthwise_initializer=KERNEL_INIT,
                               pointwise_initializer=KERNEL_INIT,
                               bias_initializer=BIAS_INIT)(inp_env)
    e = layers.BatchNormalization()(e)
    e = layers.Activation('gelu')(e)
    e = layers.MaxPool1D(2)(e)
    e = layers.GlobalAveragePooling1D()(e)

    env_feat = layers.Dense(64, activation='gelu')(e)
    env_feat = layers.Dropout(dropout)(env_feat)

    # 3. Fusión
    fused = layers.Concatenate()([raw_feat, env_feat])
    proj = layers.Dense(proj_units, activation='gelu')(fused)
    proj = layers.Dropout(dropout)(proj)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(proj)

    model = Model(inputs=[inp_raw, inp_env], outputs=out, name='hybrid_student_compact')
    return model

# === EJEMPLO DE CONSTRUCCIÓN ===
timesteps = X_kfold_s.shape[1]
channels  = X_kfold_s.shape[2]

model = build_hybrid_two_input(timesteps, channels, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
