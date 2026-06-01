import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from config import GBM, CIR

ALPHA_TRUE = 0.104505836

np.random.seed(50)

def main():
    drift_implicit_study_strong_order()
    plot_cir_trajectories()


    e_kt = np.exp(-CIR.kappa * CIR.maturity)
    mean_exact = CIR.v0 * e_kt + CIR.theta * (1 - e_kt)
    var_exact   = (CIR.v0 * CIR.sigma**2 / CIR.kappa * (e_kt - e_kt**2)
                   + CIR.theta * CIR.sigma**2 / (2*CIR.kappa) * (1 - e_kt)**2)

    # v_maturity = z_maturity ** 2
    
    budgets = np.array([20_000, 100_000, 500_000])
    R = 10
    results = np.array([[coupled_sum_mc(budget) for _ in range(R)] for budget in budgets])
    means, stds, sdes, costs, n_samples = results.transpose(2, 0, 1)
    mean_across_runs = means.mean(axis=1)   # shape: (len(budgets),)
    empirical_bias = mean_across_runs - ALPHA_TRUE
    std_across_runs  = means.std(axis=1)
    se_across_runs   = sdes.mean(axis=1)
    avg_work_per_replicate = costs / n_samples   # shape: (len(budgets), R)
    work_variance = (avg_work_per_replicate * stds ** 2).mean(axis=1)

    ci_left  = mean_across_runs - 1.96 * se_across_runs
    ci_right = mean_across_runs + 1.96 * se_across_runs
    rmse = np.sqrt(((means - ALPHA_TRUE) ** 2).mean(axis=1))  # shape: (len(budgets),)


        
    plt.errorbar(budgets, means, sdes)
    plt.show()
    budget = 500_000
        
        # compute confidence interval
    mean, std, sde, cost = coupled_sum_mc(budget)
    critical_value = 1.96 # for 95% confidence interval
    left = mean - critical_value * sde
    right = mean + critical_value * sde
    

    alpha_mc = standard_mc(budget)

    print("end of main")

def coupled_sum_sampler(truncation_idx: int, truncating_proba) -> float:

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


def coupled_sum_mc(budget):
    truncating_proba = 1 - 2 ** (-3 / 2) # I am not sure about the 1-...
    sample_cost = 0
    sample_mean = 0
    n_sample = 0
    sample_std = 0.0
    # doing coupled sum monte carlo
    while True:
        truncation_idx = np.random.geometric(p=truncating_proba)
        cost = 2 ** (truncation_idx + 1) - 1
        if sample_cost + cost > budget:
            break
        sample_cost += cost
        payoff_draw = coupled_sum_sampler(truncation_idx, truncating_proba)
        sample_mean += payoff_draw
        n_sample += 1
        sample_std += (payoff_draw - ALPHA_TRUE) ** 2
    sample_mean = sample_mean / n_sample
    sample_std = np.sqrt(sample_std / n_sample)
    
    standard_error = sample_std / np.sqrt(n_sample)
    
    return sample_mean, sample_std, standard_error, sample_cost, n_sample

def standard_mc(budget): # TODO: rethink how to use for other schemes than milstein
    zeta = 1
    n_steps = round(budget ** (1 / (2 * zeta + 1)))
    dt = GBM.maturity / n_steps
    n_paths = round(budget ** (2 * zeta / (2 * zeta + 1)))
    dw_batch = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    s_maturity = milstein_gbm(GBM.s0, dt, dw_batch)
    payoff = european(s_maturity)
    mean = np.mean(payoff)
    std = np.std(payoff, ddof=1)
    return mean, std

def plot_cir_trajectories(n_steps=2**8, n_paths=30):
    z0 = np.sqrt(CIR.v0)
    dt = CIR.maturity / n_steps
    t = np.linspace(0, CIR.maturity, n_steps + 1)

    dw_em = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    v_em = lamperti_drift_implicit_em_cir(z0, dw_em) ** 2

    dw_mil = np.random.normal(0, np.sqrt(dt), (n_paths, n_steps))
    v_milstein = drift_implicit_milstein_cir(CIR.v0, dw_mil)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(t, v_em.T)
    ax1.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('v')
    ax1.set_title('Lamperti Drift Implicit EM')

    ax2.plot(t, v_milstein.T)
    ax2.axhline(y=0.0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('v')
    ax2.set_title('Drift Implicit Milstein')

    plt.tight_layout()
    plt.show()

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

def lamperti_drift_implicit_em_cir(z0, dw):
    n_paths, n_steps = dw.shape
    dt = CIR.maturity / n_steps
    z = np.empty(shape=(n_paths, n_steps + 1))
    z[:, 0] = z0
    a = 1 + CIR.kappa * dt / 2
    c = CIR.kappa * dt / 2 * (CIR.theta - CIR.sigma ** 2 / (4 * CIR.kappa))
    for k in range(n_steps):
        b = z[:, k] + CIR.sigma / 2 * dw[:, k]
        disc = b ** 2 + 4 * a * c
        z[:, k + 1] = (b + np.sqrt(disc)) / (2 * a)
    return z

def drift_implicit_milstein_cir(v0, dw):
    n_paths, n_steps = dw.shape
    dt = CIR.maturity / n_steps
    v = np.empty(shape=(n_paths, n_steps + 1))
    v[:, 0] = v0
    for k in range(n_steps):
        v[:, k + 1] = 1 / (1 + CIR.kappa * dt) * (v[:, k] + CIR.kappa * CIR.theta * dt + CIR.sigma * np.sqrt(v[:, k]) * dw[:, k] + CIR.sigma ** 2 / 4 * (dw[:, k] ** 2 - dt))
    return v

def drift_implicit_study_strong_order():
    levels = range(4, 10)
    n_ref = 2 ** (max(levels) + 2)
    n_paths = 2000

    dw_fine = np.random.normal(0, np.sqrt(CIR.maturity / n_ref), (n_paths, n_ref))

    z0 = np.sqrt(CIR.v0)
    v_ref = drift_implicit_milstein_cir(CIR.v0, dw_fine)[:, -1]

    errors_em = []
    errors_milstein = []

    for l in levels:
        n = 2 ** l
        ratio = n_ref // n
        dw_coarse = dw_fine.reshape(n_paths, n, ratio).sum(axis=2)

        v_em = lamperti_drift_implicit_em_cir(z0, dw_coarse)[:, -1] ** 2
        errors_em.append(np.mean(np.abs(v_em - v_ref)))

        v_mil = drift_implicit_milstein_cir(CIR.v0, dw_coarse)[:, -1]
        errors_milstein.append(np.mean(np.abs(v_mil - v_ref)))

    h = np.array([CIR.maturity / 2**l for l in levels])
    slope_em,  _ = np.polyfit(np.log(h), np.log(errors_em),       1)
    slope_mil, _ = np.polyfit(np.log(h), np.log(errors_milstein), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, errors, slope, name in zip(
        axes,
        [np.array(errors_em), np.array(errors_milstein)],
        [slope_em, slope_mil],
        ['Lamperti Drift-Implicit EM', 'Drift-Implicit Milstein'],
    ):
        c = errors[0] / h[0] ** 0.5
        ax.loglog(h, errors,       'o-',  label=f'strong error (order≈{slope:.2f})')
        ax.loglog(h, c * h ** 0.5, '--',  label='O(√h)')
        ax.loglog(h, c * h ** 1.0, '--',  label='O(h)')
        ax.set_xlabel('h')
        ax.set_ylabel('E[|V_h(T) − V_ref(T)|]')
        ax.set_title(f'{name}\nestimated strong order: {slope:.2f}')
        ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Lamperti EM   strong order ≈ {slope_em:.3f}")
    print(f"Milstein CIR  strong order ≈ {slope_mil:.3f}")
    return slope_em, slope_mil

if __name__ == '__main__':
    main()