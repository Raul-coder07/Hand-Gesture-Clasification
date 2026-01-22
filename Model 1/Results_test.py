import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


model = tf.keras.models.load_model("mejor_modelo_kfold_2d.h5", compile=False)


model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


print(" Modelo cargado correctamente.")
print("X_test shape:", X_test.shape)
print("Y_test shape:", Y_test.shape)


gestos = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]


test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
print(f"\n📌 Keras -> Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")


y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(Y_test, axis=1)


print("\n=== Reporte de Clasificación ===\n")
print(classification_report(y_true, y_pred_classes, target_names=gestos, digits=4))


report_acc = accuracy_score(y_true, y_pred_classes)
print(f"📊 Accuracy (classification_report): {report_acc:.4f}")


cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=gestos, yticklabels=gestos)
plt.xlabel("Predicción")
plt.ylabel("Etiqueta real")
plt.title("Matriz de Confusión")
plt.show()