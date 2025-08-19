import tensorflow as tf

class Actor:
    def __init__(self, latent_size, action_size):
        self.latent_size = latent_size
        self.action_size = action_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.latent_size,)),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])
        return model

    def predict(self, latent):
        return self.model.predict(latent, verbose=0)