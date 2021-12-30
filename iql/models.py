from typing import Optional
import jax


class IQLAgent:
    def __init__(actor_lr,
                 value_lr,
                 critic_lr,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None):

        self.actor_lr = actor_lr

