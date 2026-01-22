# === CELDA: Ajuste de Tamaño de Títulos y Números de Ejes ===
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURACIÓN DE TAMAÑOS (Ajusta aquí) ---
FONT_TITLE = 22       # Títulos principales
FONT_LABEL = 18       # Texto de 'Loss', 'Accuracy' y 'Epochs'
FONT_TICKS = 15       # <--- ESTE AGRANDA LOS NÚMEROS DE LOS EJES (0, 10, 20...)
LINE_WIDTH = 3.6      # Grosor de las curvas
# ----------------------------------------------

COL_TRAIN = '#ff7f0e'
COL_VAL   = '#1f77b4'

if 'histories' not in locals() or len(histories) == 0:
    raise RuntimeError("No se encontraron historiales.")

num_folds = len(histories)
fig, axes = plt.subplots(num_folds, 2, figsize=(16, 5 * num_folds), layout='constrained')

if num_folds == 1:
    axes = np.expand_dims(axes, axis=0)

for i, h in enumerate(histories):
    fold_num = i + 1

    # --- Gráfico de Pérdida ---
    ax_loss = axes[i, 0]
    ax_loss.plot(h['loss'], color=COL_TRAIN, linewidth=LINE_WIDTH, label='Train Loss')
    ax_loss.plot(h['val_loss'], color=COL_VAL, linewidth=LINE_WIDTH, linestyle='--', label='Val Loss')

    ax_loss.set_title(f'Model 2\nFold {fold_num}: Loss curve - Dataset 1 Myoware', fontweight='bold', fontsize=FONT_TITLE, pad=15)
    ax_loss.set_ylabel('Loss', fontsize=FONT_LABEL)
    ax_loss.set_xlabel('Epochs', fontsize=FONT_LABEL)

    # --- AGRANDAR NÚMEROS DE LOS EJES ---
    ax_loss.tick_params(axis='both', labelsize=FONT_TICKS)

    ax_loss.grid(True, alpha=0.3, linewidth=1.5)
    ax_loss.legend(fontsize=14) # También agrandé un poco la leyenda
    ax_loss.margins(x=0.01, y=0.05)

    # --- Gráfico de Precisión ---
    ax_acc = axes[i, 1]
    ax_acc.plot(h['accuracy'], color=COL_TRAIN, linewidth=LINE_WIDTH, label='Train Acc')
    ax_acc.plot(h['val_accuracy'], color=COL_VAL, linewidth=LINE_WIDTH, linestyle='--', label='Val Acc')

    ax_acc.set_title(f'Fold {fold_num}: Curva de Precisión', fontweight='bold', fontsize=FONT_TITLE, pad=15)
    ax_acc.set_ylabel('Accuracy', fontsize=FONT_LABEL)
    ax_acc.set_xlabel('Epochs', fontsize=FONT_LABEL)

    # --- AGRANDAR NÚMEROS DE LOS EJES ---
    ax_acc.tick_params(axis='both', labelsize=FONT_TICKS)

    ax_acc.grid(True, alpha=0.3, linewidth=1.5)
    ax_acc.legend(fontsize=14)
    ax_acc.margins(x=0.01, y=0.05)

plt.show()