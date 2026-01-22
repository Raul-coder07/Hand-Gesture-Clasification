# === CELDA B: K-Fold training y Selección del Mejor Modelo ===
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json, os, time
import tensorflow as tf
import numpy as np

# 1. Configuración de K-Fold
N_SPLITS = 7
labels_int = y_kfold_int
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

ckpt_paths = []
fold_scores = []
oof_preds = np.zeros((len(X_kfold_s), num_classes), dtype=np.float32)
histories = []

# Variables para rastrear el mejor fold
best_fold_acc = 0.0
best_fold_idx = -1
best_fold_path = ""

# Calcular el tiempo de entrenamiento
start_time = time.time()

fold = 0
for train_idx, val_idx in skf.split(X_kfold_s, labels_int):
    fold += 1
    print(f"\n--- Iniciando Fold {fold}/{N_SPLITS} ---")

    X_tr_raw, X_val_raw = X_kfold_s[train_idx], X_kfold_s[val_idx]
    X_tr_env, X_val_env = X_env_kfold[train_idx], X_env_kfold[val_idx]
    y_tr, y_val = y_kfold_hot[train_idx], y_kfold_hot[val_idx]

    model = build_hybrid_two_input(
        timesteps=X_tr_raw.shape[1],
        channels=X_tr_raw.shape[2],
        num_classes=num_classes
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    ckpt = f"hybrid_fold{fold}.weights.h5"
    cp = ModelCheckpoint(ckpt, monitor='val_accuracy', save_best_only=True,
                         save_weights_only=True, verbose=1)
    es = EarlyStopping(monitor='val_accuracy', patience=12, verbose=1)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

    train_ds = tf.data.Dataset.from_tensor_slices(((X_tr_raw, X_tr_env), y_tr)) \
                  .shuffle(20000, seed=SEED) \
                  .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(((X_val_raw, X_val_env), y_val)) \
                .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                        callbacks=[cp, es, rlrop], verbose=2)
    histories.append(history.history)

    if os.path.exists(ckpt):
        model.load_weights(ckpt)

    _, acc_val = model.evaluate(val_ds, verbose=0)
    print(f"Resultado Fold {fold}: Accuracy = {acc_val:.4f}")

    # Guardar predicciones OOF
    preds_val = model.predict((X_val_raw, X_val_env), batch_size=BATCH_SIZE, verbose=0)
    oof_preds[val_idx] = preds_val

    ckpt_paths.append(ckpt)
    fold_scores.append(float(acc_val))

    # ACTUALIZAR EL MEJOR FOLD
    if acc_val > best_fold_acc:
        best_fold_acc = acc_val
        best_fold_idx = fold
        best_fold_path = ckpt

# FIN DEL CRONÓMETRO
end_time = time.time()
total_time = end_time - start_time

print("\n" + "="*40)
print(f"TIEMPO TOTAL DE ENTRENAMIENTO: {total_time/60:.2f} minutos")
print("="*40)

# ==========================================
# SELECCIÓN DEL MEJOR MODELO Y GUARDADO EN DRIVE
# ==========================================
print(f"\n=== MEJOR FOLD DETECTADO: Fold {best_fold_idx} (Acc: {best_fold_acc:.4f}) ===")
print("Usando únicamente los pesos de este fold para Holdout y Test.")

# Cargar el mejor modelo
best_model = build_hybrid_two_input(X_holdout_s.shape[1], X_holdout_s.shape[2], num_classes)
best_model.load_weights(best_fold_path)

# 1. Predicción sobre Holdout (Set interno 10%)
print("\nEvaluando mejor modelo en Holdout...")
preds_holdout = best_model.predict((X_holdout_s, X_env_holdout), batch_size=BATCH_SIZE, verbose=1)
holdout_acc = (preds_holdout.argmax(axis=1) == y_holdout_hot.argmax(axis=1)).mean()
print(f"Accuracy en Holdout (Mejor Fold): {holdout_acc:.4f}")

# 2. Predicción sobre Test Externo
print("\nGenerando predicciones sobre Test Externo con el mejor modelo...")
preds_test_ext = best_model.predict((X_test_s, X_env_test), batch_size=BATCH_SIZE, verbose=1)

np.save("preds_test_external_best_fold.npy", preds_test_ext)
print("Predicciones guardadas en: preds_test_external_best_fold.npy")

# Guardar logs locales
np.save("oof_preds_hybrid.npy", oof_preds)
with open("hybrid_ckpts.json", "w") as f:
    json.dump(ckpt_paths, f)

# ---------------------------------------------------------
# NUEVO: GUARDAR MODELO COMPLETO EN GOOGLE DRIVE
# ---------------------------------------------------------
drive_path = '/content/drive/MyDrive/a_DATASET'

# Asegurar que la carpeta exista
if not os.path.exists(drive_path):
    os.makedirs(drive_path)
    print(f"Directorio creado: {drive_path}")

# Nombre del archivo final
model_filename = "best_model_JC_Myoweare.keras" # Formato recomendado por Keras 3
full_save_path = os.path.join(drive_path, model_filename)

print(f"\nGuardando el mejor modelo completo en: {full_save_path} ...")
best_model.save(full_save_path)
print("Modelo guardado exitosamente. Listo para descargar y probar inferencia local.")