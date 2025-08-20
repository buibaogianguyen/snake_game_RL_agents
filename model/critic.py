import tensorflow as tf

class Critic:
    def __init__(self, state_size):
        self.state_size = state_size
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        return model

    def predict(self, state):
        return self.model.predict(state, verbose=0)