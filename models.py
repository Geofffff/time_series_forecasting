from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Model:
    pass


@dataclass(frozen=True)
class LocalLevelModel(Model):
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