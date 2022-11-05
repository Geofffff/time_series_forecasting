from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class LocalLevelModel:
    var_eps: float
    var_eta: float
    initial_alpha: float

    def simulate(self, n: int):

        # Generate all the random variables ahead of time
        eps = np.random.normal(0, self.var_eps, n)
        eta = np.random.normal(0, self.var_eta, n)

        # The alpha values are the initial alpha + the cumulative sum of the etas
        alpha = self.initial_alpha + eta.cumsum()

        # The y values are just the alpha values + the epsilons
        y = alpha + eps

        return y, alpha


class LocalLevelKalman:
    def __init__(self, initial_alpha: float, initial_P: float, var_eps: float, var_eta: float):
        self._initial_alpha = initial_alpha
        self._initial_P = initial_P
        self._var_eps = var_eps
        self._var_eta = var_eta

        self._P = self._initial_P
        self._a = self._initial_alpha

        self._F = self._P + self._var_eps
        self._K = self._P / self._F
        self._a_next = self._a
        self._P_next = self._P
        self._t = 0

        self._y = None

    def observe(self, y: float):
        # Move to the next time step t -> t+1
        self._t += 1
        self._y = y
        self._a = self._a_next
        self._P = self._P_next

        # Update K and F
        self._F = self._P + self._var_eps
        self._K = self._P / self._F

        # Set v at time t (before moving to t+1)
        self._v = y - self._a  # Here a is really a_prev as it hasnt been updated (as is y)

        # Update a and P to a and P given y_t
        self._a += self._K * self._v
        self._P = self._P * (1 - self._K)

        # Get a_next and P_next
        self._a_next = self._a
        self._P_next = self._P + self._var_eta

    @property
    def a(self):
        return self._a

    def P(self):
        return self._P

