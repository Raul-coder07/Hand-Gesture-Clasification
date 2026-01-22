
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# reproducibilidad
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


if 'X_train_2d' not in globals() or 'Y_train' not in globals():
    raise NameError("No se encontraron X_train_2d / y_train_onehot. Asegúrate de haber hecho el reshape antes.")

X_all = np.asarray(X_train_2d)
Y_all = np.asarray(Y_train)

if X_all.ndim != 4:
    raise ValueError(f"X_train_2d debe tener forma (N, tiempo, canales, 1), pero tiene {X_all.shape}.")
if Y_all.ndim != 2:
    raise ValueError("Y_train debe ser one-hot (N, n_classes).")

input_shape = X_all.shape[1:]  # (200, 4, 1)
num_classes = Y_all.shape[1]

print("Input shape:", input_shape, "Num classes:", num_classes)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def build_model(input_shape, num_classes):
    x = keras.Input(shape=input_shape)

    # Bloque 1
    y = layers.Conv2D(128, (3, 3), padding="same", activation=mish)(x)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling2D((2, 1))(y)
    y = layers.Dropout(0.4)(y)

    # Bloque 2
    y = layers.Conv2D(128, (5, 3), padding="same", activation=mish)(y)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling2D((2, 1))(y)
    y = layers.Dropout(0.4)(y)

    # Bloque 3
    y = layers.Conv2D(256, (7, 3), padding="same", activation=mish)(y)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling2D((2, 1))(y)
    y = layers.Dropout(0.4)(y)

    # Bloque 4
    y = layers.Conv2D(256, (15, 3), padding="same", activation=mish)(y)
    y = layers.BatchNormalization()(y)
    y = layers.MaxPooling2D((2, 1))(y)
    y = layers.Dropout(0.4)(y)

    # Capas densas
    y = layers.Flatten()(y)
    y = layers.Dense(256, activation=mish)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dense(128, activation=mish)(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.4)(y)

    output = layers.Dense(num_classes, activation="softmax", kernel_initializer="glorot_uniform")(y)

    model = keras.Model(inputs=x, outputs=output)
    model.compile(
        optimizer=optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

#K-Fold
num_folds = 7
kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
fold_accuracies = []
fold_val_accuracies = []
fold_histories = []
best_global_val_loss = np.inf
best_global_model_path = "mejor_modelo_kfold_2d.h5"

for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), 1):
    print(f"\n--- Fold {fold}/{num_folds} ---")
    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    Y_tr, Y_val = Y_all[train_idx], Y_all[val_idx]

    model = build_model(input_shape, num_classes)

    early_stopping = callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True, verbose=1)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)
    ckpt_path = f"best_model_fold_{fold}.h5"
    model_ckpt = callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0)

    history = model.fit(
        X_tr, Y_tr,
        validation_data=(X_val, Y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr, model_ckpt],
        verbose=2
    )

    fold_histories.append(history)

    train_acc = history.history.get("accuracy", [None])[-1]
    val_acc = max(history.history.get("val_accuracy", [0]))
    fold_accuracies.append(train_acc)
    fold_val_accuracies.append(val_acc)

    best_val_loss_fold = min(history.history.get("val_loss", [np.inf]))
    print(f"Fold {fold} -> train_acc: {train_acc:.4f}, best_val_acc: {val_acc:.4f}, best_val_loss: {best_val_loss_fold:.4f}")

    if best_val_loss_fold < best_global_val_loss:
        best_global_val_loss = best_val_loss_fold
        try:
            best_model = keras.models.load_model(ckpt_path, compile=False)
            best_model.save(best_global_model_path)
            print(f" Nuevo mejor modelo global guardado en {best_global_model_path}")
        except Exception as e:
            model.save(best_global_model_path)
            print(f" Error al guardar mejor modelo global: {e}")


fold_accuracies = np.array(fold_accuracies)
fold_val_accuracies = np.array(fold_val_accuracies)
print("\nK-Fold summary:")
print("Train acc:", fold_accuracies)
print("Val acc  :", fold_val_accuracies)
print("Mean train acc:", np.mean(fold_accuracies), "Std:", np.std(fold_accuracies))
print("Mean val acc:", np.mean(fold_val_accuracies), "Std:", np.std(fold_val_accuracies))
print(f"\n📌 Mejor modelo global guardado en: {best_global_model_path} (val_loss = {best_global_val_loss:.4f})")

# perdidas
cols = 3
rows = (num_folds + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
axes = axes.flatten()

for i in range(num_folds):
    ax = axes[i]
    h = fold_histories[i].history
    ax.plot(h["loss"], '--', label="Train Loss")
    ax.plot(h["val_loss"], '-', label="Val Loss")
    ax.set_title(f"Fold {i+1}")
    ax.legend()
    ax.grid(True)

for j in range(num_folds, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle("Pérdida por Fold (CNN 2D)", fontsize=16)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

model.summary()