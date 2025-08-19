import tensorflow as tf

class Dynamics:
    def __init__(self, latent_size, action_size):
        self.latent_size = latent_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        inputs = tf.keras.layers.Input(shape=(self.latent_size + self.action_size,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        latent = tf.keras.layers.Dense(self.latent_size, activation='relu')(x)
        reward = tf.keras.layers.Dense(1)(x)
        done = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        return tf.keras.models.Model(inputs, [latent, reward, done])

    def predict(self, inputs):
        return self.model.predict(inputs, verbose=0)