import tensorflow as tf
from tensorflow.keras.layers import Input,Flatten,Dense,Lambda,Reshape, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

latent_dim = 4
input = Input(shape=(28, 28, 1))
x = Conv2D(8, (3, 3), activation='relu', padding='same',name='enc/conv1')(input)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same',name='enc/conv2')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same',name='enc/conv3')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Flatten()(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
encoder = Model(input, [z_mean, z_log_var, z], name='encoder')

latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(16*4*4,activation='relu',name='D3')(latent_inputs)
x = Reshape((4,4,16))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same',name='dec/conv1')(x)
x = UpSampling2D((2, 2),name='dec/up1')(x)
x = Conv2D(12, (3, 3), activation='relu', padding='same',name='dec/conv2')(x)
x = UpSampling2D((2, 2),name='dec/up2')(x)
x = Conv2D(8, (3, 3), activation='relu',name='dec/conv3')(x)
x = UpSampling2D((2, 2),name='dec/up3')(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name='dec/conv4')(x)
decoder = Model(latent_inputs, decoded, name='decoder')

output=decoder(encoder(input)[2])
vae = Model(input,output)
print(vae.summary())

reconstruction_loss = binary_crossentropy(K.flatten(input), K.flatten(output))
reconstruction_loss *= 28*28
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)*(-0.5)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='Adam')

(x_train, _), (x_test, _) = mnist.load_data()
x_train =  np.expand_dims(x_train.astype('float32') / 255, axis=3)
x_test = np.expand_dims(x_test.astype('float32') / 255, axis=3)

vae.fit(x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, None),
                callbacks=[TensorBoard(log_dir='logs')])
vae.save_weights('tmp.tf')

decoder.load_weights('tmp.tf')
reconstructed_imgs = vae.predict(x_test)
generated_imgs=decoder.predict(np.random.randn(10,latent_dim))

n = 10
fig, axes = plt.subplots(3, n,figsize=(20, 4))
for i in range(n):
    # first row: original
    axes[0][i].imshow(x_test[i].reshape(28, 28))
    axes[0][i].axis('off')

    # second row: reconstructed
    axes[1][i].imshow(reconstructed_imgs[i].reshape(28, 28))
    axes[1][i].axis('off')

    # second row: generated from sampling noise
    axes[2][i].imshow(generated_imgs[i].reshape(28, 28))
    axes[2][i].axis('off')
plt.show()
