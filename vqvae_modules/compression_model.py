#compression_model
## VQ-VAE
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from vq_vae import *


def vq_vae(x_train_m, commitment_cost, embedding_dim, num_embeddings):
  input = Input(shape=(x_train_m.shape[1], x_train_m.shape[2],1))
  x = Conv2D(filters=32, kernel_size=(7,7),  strides=(2,2),padding="same", activation="relu")(input)

  # VQVAELayer.
  enc = Conv2D(embedding_dim, kernel_size=(1,1), strides=1, name="pre_vqvae")(x)
  enc_inputs = enc
  enc = VQVAELayer(embedding_dim, num_embeddings, commitment_cost, name="vqvae")(enc)
  x = Lambda(lambda enc: enc_inputs + K.stop_gradient(enc - enc_inputs), name="encoded")(enc)
  data_variance = np.var(x_train_m)
  loss = vq_vae_loss_wrapper(data_variance, commitment_cost, enc, enc_inputs)

  # Decoder.
  x = Conv2D(filters=16, kernel_size=(7,7), strides=2, padding = 'same')(x)
  x = UpSampling2D(size = (2,2))(x)
  x = Conv2D(filters=32, kernel_size=(7,7),  strides=1, padding = 'same')(x)
  x = UpSampling2D(size = (2,2))(x)
  x = Conv2D(filters=1, kernel_size=(7,7), padding = 'same')(x)

  # Autoencoder.
  return (Model(input, x),loss)