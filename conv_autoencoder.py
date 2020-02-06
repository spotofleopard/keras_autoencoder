import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt

input = Input(shape=(28, 28, 1))
x = Conv2D(8, (3, 3), activation='relu', padding='same')(input)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input, decoded)
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()
x_train =  np.expand_dims(x_train.astype('float32') / 255, axis=3)
x_test = np.expand_dims(x_test.astype('float32') / 255, axis=3)

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir='logs')])

reconstructed_imgs = autoencoder.predict(x_test)

n = 10
fig, axes = plt.subplots(2, n,figsize=(20, 4))
for i in range(n):
    # first row: original
    axes[0][i].imshow(x_test[i].reshape(28, 28))
    axes[0][i].axis('off')

    # second row: reconstructed
    axes[1][i].imshow(reconstructed_imgs[i].reshape(28, 28))
    axes[1][i].axis('off')
plt.show()
