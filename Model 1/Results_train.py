
import numpy as np

if "fold_accuracies" not in globals() or "fold_val_accuracies" not in globals():
    raise NameError("No se encontraron los resultados de K-Fold. Ejecuta primero el entrenamiento.")


train_accs = np.array(fold_accuracies, dtype=np.float32)
val_accs = np.array(fold_val_accuracies, dtype=np.float32)

print("=== Resultados por fold ===")
for i, (t, v) in enumerate(zip(train_accs, val_accs), 1):
    print(f"Fold {i}: train_acc={t:.4f}, val_acc={v:.4f}")

print("\n=== Estadísticas globales ===")
print(f"Train Acc  -> Media: {np.mean(train_accs):.4f},  Std: {np.std(train_accs):.4f}")
print(f"Val Acc    -> Media: {np.mean(val_accs):.4f},  Std: {np.std(val_accs):.4f}")