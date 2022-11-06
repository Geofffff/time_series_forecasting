from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class KalmanHistory:
    # Calculate smoothed history if required -> convert to static and add history
    # Options for plotting and residuals
    # Summary stats
    # Allow for updating but also allow for it to return a frozen version eg. with smoothing
    # Perfomance evaluation with simulated data?
    pass


class LocalLevelKalman:
    def __init__(self, initial_alpha: float, initial_P: float, var_eps: float, var_eta: float, record: bool = False):
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
        self._v = None

        if record:
            self._history = pd.DataFrame(columns=["a", "y", "P"])
        else:
            self._history = None

    @property
    def a(self):
        return self._a

    @property
    def P(self):
        return self._P

    @property
    def history(self):
        if self._history is None:
            raise AttributeError("This KalmanFilter was not set up to record values")
        return self._history

    def tick(self, y: Optional[float] = None):
        # Move to the next time step t -> t+1
        self._t += 1
        self._y = y
        self._a = self._a_next
        self._P = self._P_next

        # Update K and F
        self._F = self._P + self._var_eps
        self._K = self._P / self._F

        if y is not None:
            self._v = y - self._a

            # Update a and P to a and P given y_t
            self._a += self._K * self._v
            self._P = self._P * (1 - self._K)

        # Get a_next and P_next
        self._a_next = self._a
        self._P_next = self._P + self._var_eta

        if isinstance(self._history, pd.DataFrame):
            new_entry = pd.Series([self._a, self._y, self._P], index=["a", "y", "P"], name=self._t)
            self._history.append(new_entry)

