class DreamerAgent:
    def __init__(self, state_size=13, action_size=3, latent_size=32):
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.gamma = 0.99
        self.lamb = 0.95

        self.learning_rate = 0.0001
        