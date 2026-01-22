import tensorflow as tf


best_model = tf.keras.models.load_model("mejor_modelo_kfold_2d.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("modelo_EMG.tflite", "wb") as f:
    f.write(tflite_model)

print(" Modelo convertido a TFLite y guardado como modelo_EMG.tflite")
