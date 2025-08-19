import tensorflow as tf

class Reward:
    def __init__(self, latent_size):
        self.latent_size = latent_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(self.latent_size,)),
            tf.keras.layers.Dense(1)
        ])
        return model

    def predict(self, latent):
        return self.model.predict(latent, verbose=0)