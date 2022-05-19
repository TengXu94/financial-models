"""
Code is based on:
https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python 
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class MertonDiffusionModel:
    def __init__(self):

        self.S = 100  # current stock price
        self.T = 1  # time to maturity
        self.r = 0.02  # risk free rate
        self.m = 0  # meean of jump size
        self.v = 0.15  # standard deviation of jump
        self.lam = 1  # intensity of jump i.e. number of jumps per annum
        self.steps = 10000  # time steps
        self.Npaths = 3  # number of paths to simulate
        self.sigma = 0.2  # annual standard deviation , for weiner process

    def merton_jump_paths(self, S, T, r, sigma, lam, m, v, steps, Npaths) -> np.array:
        size = (steps, Npaths)
        dt = T / steps

        # N_t
        jump_probabilities = np.random.poisson(lam * dt, size=size)
        # Y_t
        jump_sizes = np.random.normal(m, v, size=size)

        # \sum_{j=1}^N_t
        poi_rv = np.multiply(jump_probabilities, jump_sizes).cumsum(axis=0)

        geo = np.cumsum(
            (
                (r - sigma**2 / 2 - lam * (m + v**2 * 0.5)) * dt
                + sigma * np.sqrt(dt) * np.random.normal(size=size)
            ),
            axis=0,
        )

        # S_t = S_0 * np.exp(geo_poi_rv)
        return np.exp(geo + poi_rv) * S

    def plot_merton_jump_paths(self):
        j = self.merton_jump_paths(
            self.S,
            self.T,
            self.r,
            self.sigma,
            self.lam,
            self.m,
            self.v,
            self.steps,
            self.Npaths,
        )

        plt.plot(j)
        plt.xlabel("Days")
        plt.ylabel("Stock Price")
        plt.title("Jump Diffusion Process")
        plt.show()


if __name__ == "__main__":
    merton = MertonDiffusionModel()
    merton.plot_merton_jump_paths()
