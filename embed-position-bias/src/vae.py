import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
from keras import backend as K
from tensorflow.keras.layers import Input, Dense, Lambda
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize



class Encoder(tf.keras.Model):
    """
    Encoder for a Variational AutoEncoder (VAE).

    This encoder maps inputs into a latent space using a fully connected network. It outputs
    the parameters (mean and log variance) of the assumed Gaussian distribution in the latent space.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The dimensionality of the hidden layer.
        latent_dim (int): The dimensionality of the latent space representation.

    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
    super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dense_1 = Dense(self.hidden_dim, activation='relu')
        self.dense_mu = Dense(self.latent_dim, activation='linear')
        self.dense_sigma = Dense(self.latent_dim, activation='linear')

    def call(self, x_input):
        hidden = self.dense_1(x_input)
        mu = self.dense_mu(hidden)
        zog_sigma = self.dense_sigma(hidden)
        eps = K.random_normal(shape=(self.latent_dim,), mean=0., stddev=0.1)
        z = mu + K.exp(log_sigma) * eps

        return mu, log_sigma, z

class Decoder(tf.keras.Model):
    """
    Decoder for a Variational AutoEncoder (VAE).

    This decoder maps points in latent space back to the original input space using a fully connected network.

    Attributes:
        input_dim (int): The dimensionality of the output data (same as the input dimensionality of the encoder).
        hidden_dim (int): The dimensionality of the hidden layer.
        latent_dim (int): The dimensionality of the latent space representation.

    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.dense_1 = Dense(self.hidden_dim, activation='relu')
        self.dense_output = Dense(self.input_dim, activation='sigmoid')

    def call(self, z):
        hidden = self.dense_1(z)
        output = self.dense_output(hidden)
        return output

class VAE(tf.keras.Model):
    """
    Variational AutoEncoder (VAE) model combining the Encoder and Decoder.

    The VAE model maps inputs to a latent space using the Encoder, and then maps the latent
    variables back to the input space using the Decoder.

    Attributes:
        input_dim (int): The dimensionality of the input data.
        hidden_dim (int): The dimensionality of the hidden layers.
        latent_dim (int): The dimensionality of the latent space representation.

    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dim, latent_dim)


    def call(self, x):
        mu, log_sigma, z = self.encode(x)
        x_decoded = self.decoder(z)
        return mu, log_sigma, x_decoded, z

    def encode(self, x):
        mu, log_sigma, z = self.encoder(x)
        return mu, log_sigma, z

    def decode(self, z):
        x_decoded = self.decoder(z)
        return x_decoded

def loss_function(label, predict, mu, log_sigma):
    """
    Computes the VAE loss function which is the sum of reconstruction loss and the KL divergence regularization term.

    Parameters:
        label (tensor): The true data (input).
        predict (tensor): The reconstructed data (output of the VAE).
        mu (tensor): The mean of the latent space distribution.
        log_sigma (tensor): The log variance of the latent space distribution.

    Returns:
        tensor: The calculated loss value.
    """
    reconstruction_loss = tf.keras.losses.binary_crossentropy(label, predict)
    reconstruction_loss *= 768
    kl_loss = 1 + log_sigma - K.square(mu) - K.exp(log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    return vae_loss

@tf.function
def train_step(x):
    """
    Performs one training step for the VAE model including the forward pass, loss calculation, and optimization step.

    Parameters:
        x (tensor): A batch of input data.

    Returns:
        tensor: The calculated batch loss.
    """
    loss = 0
    with tf.GradientTape() as tape:
        mu, log_sigma, x_reconstructed, z = vae(x, training=True)
        loss += loss_function(x, x_reconstructed, mu, log_sigma)
    batch_loss = (loss / len(x))
    variables = vae.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss



def get_embedding_from_VAE(df):
    """
    Trains the VAE model on the provided dataset and returns the embedding matrix representing the data in the latent space.

    Parameters:
        df (DataFrame): The input dataset to be transformed into an embedding.

    Returns:
        DataFrame: The embedding matrix corresponding to the input data.
    """
    df = normalize(df)
    x_train, x_test  = train_test_split(df, test_size=0.33, random_state=42)
    optimizer = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    BUFFER_SIZE = 5
    BATCH_SIZE = 5
    dataset = tf.data.Dataset.from_tensor_slices((x_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    steps_per_epoch = len(x_train) // BATCH_SIZE # 何個に分けるか

    EPOCHS = 300
    vae = VAE(input_dim=x_train.shape[-1], hidden_dim=15, latent_dim=15)
    for epoch in range(EPOCHS):
        for batch, x in enumerate(dataset):
            batch_loss = train_step(x)
            #if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.numpy()))

    mu, sigma, z = vae.encode(df) 
    z = z.numpy()
    embedding = pd.DataFrame(z)
    embedding.to_csv('embedding_matrix.csv')
    return embedding
