import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from config import GBM, CIR

np.random.seed(42)

def main():
    # drift = lambda x: MU * x
    # diffusion = lambda x: SIGMA * x
    # diffusion_dx = lambda x: SIGMA

    truncating_proba = 2 ** (-3 / 2) # this will need to be computed

    bla = np.array([3,7,3,3,3,7,1,8,7])
    un, counts = np.unique(bla, return_counts=True)

    gamma = 10_000 
    # alpha = 0
    # for g in range(gamma):
    #     z = coupled_sum_estimator(truncating_proba)
    #     alpha += z
        
    # alpha = alpha / gamma

    alpha = np.mean([coupled_sum_estimator(truncating_proba) for _ in range(gamma)])
    alpha_mc = monte_carlo_estimator()

    truncation_idx = np.random.geometric(p=truncating_proba)

    n = 2 ** truncation_idx 
    r = 8 
    # n_gross = n // r
    t = np.linspace(0, GBM.maturity, n + 1)
    dt = t[1:] - t[:-1]
    # t_gross = t[::r]
    dw = np.random.normal(loc=0, scale=np.sqrt(dt), size=n)
    # dw_gross = np.add.reduceat(dw, range(0, n, r))
    w = np.r_[0, np.cumsum(dw)]
    # w_gross = np.r_[0, np.cumsum(dw_gross)]
    # s = np.empty_like(w_gross)
    # s[0] = S0
    # for n in range(n_gross):
    #     s[n + 1] = milstein_step(s[n], t_gross[n + 1] - t_gross[n], dw_gross[n], drift, diffusion, diffusion_dy)

    # s_maturity = s[-1]
    

    s_true = GBM.s0 * np.exp((GBM.mu - GBM.sigma ** 2 / 2) * t + GBM.sigma * w)
    s_maturity = milstein_gbm(GBM.s0, dt, dw)
    # plt.plot(t, s_true)
    # plt.plot(t_gross, s)
    # plt.show()
    
    payoffs = np.empty(truncation_idx + 1)

    # for i in range(truncation_idx, -1, -1):
    #     s_maturity = milstein(S0, dw, dt, drift, diffusion, diffusion_dx)
    #     payoffs[i] = european(s_maturity)
    #     dt = dt[0::2] + dt[1::2]
    #     dw = dw[0::2] + dw[1::2]

    # coupled_sum_estimatorr = payoffs[0] + np.sum([(payoffs[i+1] - payoffs[i]) / (1 - truncating_proba) ** i for i in range(truncation_idx)])
    
    print("end of main")

def coupled_sum_estimator(
        truncating_proba: float,
        ) -> float:
    truncation_idx = np.random.geometric(p=truncating_proba)

    n = 2 ** truncation_idx 
    r = 8 
    t = np.linspace(0, GBM.maturity, n + 1)
    dt = t[1:] - t[:-1]
    dw = np.random.normal(loc=0, scale=np.sqrt(dt), size=n)
    payoffs = np.empty(truncation_idx + 1)

    for i in range(truncation_idx, -1, -1):
        s_maturity = milstein_gbm(GBM.s0, dt, dw)
        payoffs[i] = european(s_maturity)
        dt = dt[0::2] + dt[1::2]
        dw = dw[0::2] + dw[1::2]
        
    estimator = payoffs[0] + np.sum([(payoffs[i+1] - payoffs[i]) / (1 - truncating_proba) ** i for i in range(truncation_idx)])
    return estimator

def monte_carlo_estimator():
    n = 2 ** 10
    dt = GBM.maturity / n
    count = 10000
    dw_batch = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(count, n))
    s_maturity = milstein_gbm(GBM.s0, dt, dw_batch)
    payoff = european(s_maturity)
    mean = np.mean(payoff)
    std = np.std(payoff, ddof=1)
    return mean, std

def european(x):
    return np.exp(-GBM.mu * GBM.maturity) * np.maximum(x - GBM.strike, 0.0)

def milstein_step(
        x: float,
        dt: float,
        dw: float,
        drift: Callable,
        diffusion: Callable,
        diffusion_dx: Callable,
        ) -> float:
    diff = diffusion(x)
    return (
        x +
        drift(x) * dt + 
        diff * dw + 
        0.5 * diff * diffusion_dx(x) * (dw ** 2 - dt)
    )

def milstein(
        x0: float,
        dw: ArrayLike,
        dt: ArrayLike,
        drift: Callable,
        diffusion: Callable,
        diffusion_dx: Callable,
        ) -> float:
    x = np.empty(len(dt) + 1)
    x[0] = x0
    for n in range(len(dt)):
        x[n + 1] = milstein_step(x[n], dt[n], dw[n], drift, diffusion, diffusion_dx)
    return x[-1]

def milstein_gbm(x0, dt, dw):
    factors = 1 + GBM.mu * dt + GBM.sigma * dw + 0.5 * GBM.sigma ** 2 * (dw ** 2 - dt)
    return x0 * np.prod(factors, axis=-1)
        


if __name__ == '__main__':
    main()