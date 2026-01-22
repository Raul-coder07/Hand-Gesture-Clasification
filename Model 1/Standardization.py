import numpy as np

#Y_train=y_train_onehot
#Y_test=y_test_onehot

# Estandarizar usando media y std del TRAIN
mean_train = np.mean(X_train, axis=0, keepdims=True)
std_train  = np.std(X_train, axis=0, keepdims=True)


np.save('mean_train.npy', mean_train)
np.save('std_train.npy', std_train)


X_train = (X_train - mean_train) / std_train
X_test  = (X_test  - mean_train) / std_train

# Se expande la última dimensión para obtener shape (n, 200, 4, 1)
X_train_2d = np.expand_dims(X_train, axis=-1)
X_test  = np.expand_dims(X_test, axis=-1)

print("X_train_2d shape:", X_train_2d.shape)
print("X_test_2d shape:", X_test.shape)