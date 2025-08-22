# DQN and PPO agents on Snake Game

This project implements two reinforcement learning models - Deep-Q Network (DQN) and Proximal Policy Optimization (PPO) - to be trained on the classic Snake Game with 10x10 playable grids and rounded edges with TensorFlow.

5 Minute Demo of the DQN Agent's training

[![Watch the demo](https://img.youtube.com/vi/0oqSsnyiLy8/0.jpg)](https://youtu.be/0oqSsnyiLy8?si=tV5923hJvTZ2Ires)



# Navigation
- [DQN Research Paper - Playing Atari with Deep Reinforcement Learning](#research-paper-dqn)
- [PPO Research Paper - Proximal Policy Optimization Algorithms](#research-paper-ppo)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)

# Research Paper - Playing Atari with Deep Reinforcement Learning <a id="research-paper-dqn"></a>

This model is a Deep-Q Network reinforcement learning architecture proposed in the 2013 research paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602), authored by Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra, and Martin Riedmiller. DQN uses an epsilon-greedy exploration strategy with experience replay

# Research Paper - Proximal Policy Optimization Algorithms <a id="research-paper-ppo"></a>

This model is a Proximal Policy Optimization reinforcement learning architecture proposed in the 2017 research paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1707.06347), authored by John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. PPO uses an actor-critic technique and is stabilized by using a clipped surrogate objective.

# Requirements

```bash
tensorflow>=2.0
numpy
pygame
```

- See `requirements.txt` for a full list of dependencies.

# Project Structure

```bash
snake_game_RL_agent/
├── model/
│   ├── __init__.py
│   └── actor.py
│   └── critic.py
├── requirements.txt
├── dqn_agent.py
├── main_dqn.py
├── ppo_agent.py
└── main_ppo.py
```

# Setup

Clone the repository:
```cmd
git clone https://github.com/buibaogianguyen/snake_game_RL_agent.git
cd snake_game_RL_agent
```

Install dependencies:
```cmd
pip install -r requirements.txt
```

# Usage

To train the model using DQN, run:
```cmd
python main_dqn.py
```

To train the model using PPO, run:
```cmd
python main_ppo.py
```

Configurations:
- **Episodes**: Default is 1000 (adjustable in `main_dqn.py/main_ppo.py`).
- **Batch Size**: Currently set to 64, adjust as needed.
- **Render**: True by default. To see your agent play and learn, adjust to true in `main_dqn.py/main_ppo.py`, for efficient training, adjust to false.
  
# License

This project is licensed under the MIT License. See the `LICENSE` file for details (create one if needed).

# Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or bug fixes.
