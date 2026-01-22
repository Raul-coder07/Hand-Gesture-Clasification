import numpy as np
import matplotlib.pyplot as plt

def plot_first_sample_per_class(X_train, y_train, gestos):
    num_classes = len(gestos)
    num_channels = X_train.shape[2]


    colors = plt.cm.tab10(np.linspace(0, 1, num_channels))


    y_indices = np.argmax(y_train, axis=1)


    global_min = np.min(X_train)
    global_max = np.max(X_train)

    plt.figure(figsize=(22, 22))
    plt.suptitle("Primera muestra de cada clase (Train)", fontsize=20)

    for class_idx in range(num_classes):


        sample_index = np.where(y_indices == class_idx)[0][0]
        sample = X_train[sample_index]

        for ch in range(num_channels):
            ax = plt.subplot(num_classes, num_channels, class_idx * num_channels + ch + 1)
            ax.plot(sample[:, ch], color=colors[ch], linewidth=1.2)
            ax.set_title(f"{gestos[class_idx]}", fontsize=12)

            # MISMA ESCALA para todas las gráficas
            ax.set_ylim(global_min, global_max)

            if class_idx == num_classes - 1:
                ax.set_xlabel("Tiempo (muestras)")
            if ch == 0:
                ax.set_ylabel("Amplitud")

            ax.grid(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()



gestos = ["G1","G2","G3","G4","G5","G6","G7"]
plot_first_sample_per_class(X_train, Y_train, gestos)
