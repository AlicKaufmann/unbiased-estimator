import numpy as np
from typing import Callable
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from config import GBM, CIR

ALPHA_TRUE = 0.104505836

np.random.seed(50)

def main():
    coupled_sum_study()

    gbm_ref = ALPHA_TRUE
    comparison_study(
        budgets=np.array([20_000, 100_000, 500_000]),
        x0=GBM.s0, maturity=GBM.maturity,
        scheme=milstein_gbm, test_fn=european,
        true_val=gbm_ref, title='GBM',
    )

    drift_implicit_study_strong_order()
    plot_cir_trajectories()

    cir_ref, _, _, _, _ = coupled_sum_mc(5_000_000, CIR.v0, CIR.maturity, milstein_cir_scheme, cir_european)
    comparison_study(
        budgets=np.array([1_000, 10_000, 50_000]),
        x0=CIR.v0, maturity=CIR.maturity,
        scheme=milstein_cir_scheme, test_fn=cir_european,
        true_val=cir_ref, title='CIR',
    )

    print("end of main")

def coupled_sum_sampler(truncation_idx: int, truncating_proba, x0, maturity, scheme, test_fn) -> float:
    n = 2 ** truncation_idx
    t = np.linspace(0, maturity, n + 1)
    dt = t[1:] - t[:-1]
    dw = np.random.normal(loc=0, scale=np.sqrt(dt), size=n)
    payoffs = np.empty(truncation_idx + 1)

    for i in range(truncation_idx, -1, -1):
        payoffs[i] = test_fn(scheme(x0, dt, dw))
        dt = dt[0::2] + dt[1::2]
        dw = dw[0::2] + dw[1::2]

    estimator = payoffs[0] + np.sum([(payoffs[i+1] - payoffs[i]) / (1 - truncating_proba) ** i for i in range(truncation_idx)])
    return estimator


def coupled_sum_mc(budget, x0, maturity, scheme, test_fn):
    truncating_proba = 1 - 2 ** (-3 / 2)
    sample_cost = 0
    n_sample = 0
    mean = 0.0
    M2 = 0.0
    while True:
        truncation_idx = np.random.geometric(p=truncating_proba)
        cost = 2 ** (truncation_idx + 1) - 1
        if sample_cost + cost > budget:
            break
        sample_cost += cost
        draw = coupled_sum_sampler(truncation_idx, truncating_proba, x0, maturity, scheme, test_fn)
        n_sample += 1
        delta = draw - mean
        mean += delta / n_sample
        M2 += delta * (draw - mean)
    std = np.sqrt(M2 / n_sample)
    se = std / np.sqrt(n_sample)
    return mean, std, se, sample_cost, n_sample

def coupled_sum_study():
    budgets = np.array([20_000, 100_000, 500_000])
    R = 10
    results = np.array([[coupled_sum_mc(budget, GBM.s0, GBM.maturity, milstein_gbm, european) for _ in range(R)] for budget in budgets])
    means, stds, sdes, costs, _ = results.transpose(2, 0, 1)

    estimator_mean = means.mean(axis=1)
    estimator_se   = sdes.mean(axis=1)
    estimator_variance   = (stds ** 2).mean(axis=1)
    ci_left        = estimator_mean - 1.96 * estimator_se
    ci_right       = estimator_mean + 1.96 * estimator_se
    rmse           = np.sqrt(((means - ALPHA_TRUE) ** 2).mean(axis=1))
    bias           = estimator_mean - ALPHA_TRUE
    avg_cost       = costs.mean(axis=1)

    header = f"{'Budget':>10}  {'Mean':>10}  {'Variance':>10}  {'Bias':>10}  {'RMSE':>10}  {'95% CI':>28}  {'Avg cost':>10}"
    print(header)
    print('-' * len(header))
    for i, budget in enumerate(budgets):
        ci_str = f"[{ci_left[i]:.6f}, {ci_right[i]:.6f}]"
        print(f"{budget:>10}  {estimator_mean[i]:>10.6f}  {estimator_variance[i]:>10.2e}  {bias[i]:>10.2e}  {rmse[i]:>10.2e}  {ci_str:>28}  {avg_cost[i]:>10.0f}")

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.errorbar(budgets, estimator_mean, yerr=1.96 * estimator_se, fmt='o-', capsize=5, label='Mean ± 95% CI')
    ax1.axhline(ALPHA_TRUE, color='r', linestyle='--', label=f'True = {ALPHA_TRUE}')
    ax1.set_xscale('log')
    ax1.set_xlabel('Budget')
    ax1.set_ylabel('Estimated mean')
    ax1.set_title('Coupled-sum estimator')
    ax1.legend()

    ax2.loglog(budgets, rmse, 'o-', label='RMSE')
    ax2.set_xlabel('Budget')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs budget')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def standard_mc(budget, x0, maturity, scheme, test_fn):
    zeta = 1
    n_steps = round(budget ** (1 / (2 * zeta + 1)))
    n_paths = round(budget ** (2 * zeta / (2 * zeta + 1)))
    dt = maturity / n_steps
    dw_batch = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    payoff = test_fn(scheme(x0, dt, dw_batch))
    mean = np.mean(payoff)
    std = np.std(payoff, ddof=1)
    se = std / np.sqrt(n_paths)
    cost = n_steps * n_paths
    return mean, std, se, cost, n_paths

def comparison_study(budgets, x0, maturity, scheme, test_fn, true_val, title=''):
    R = 10

    results_cs  = np.array([[coupled_sum_mc(b, x0, maturity, scheme, test_fn) for _ in range(R)] for b in budgets])
    results_smc = np.array([[standard_mc(b, x0, maturity, scheme, test_fn)    for _ in range(R)] for b in budgets])

    def summarise(results):
        means, stds, sdes, costs, n_samples = results.transpose(2, 0, 1)
        mean      = means.mean(axis=1)
        se        = sdes.mean(axis=1)
        variance  = (stds ** 2).mean(axis=1)
        ci_left   = mean - 1.96 * se
        ci_right  = mean + 1.96 * se
        rmse      = np.sqrt(((means - true_val) ** 2).mean(axis=1))
        bias      = mean - true_val
        avg_work  = (costs / n_samples).mean(axis=1)
        work_var  = avg_work * variance
        return mean, se, variance, ci_left, ci_right, rmse, bias, work_var

    cs  = summarise(results_cs)
    smc = summarise(results_smc)
    mean_cs,  se_cs,  var_cs,  cil_cs,  cir_cs,  rmse_cs,  bias_cs,  wv_cs  = cs
    mean_smc, se_smc, var_smc, cil_smc, cir_smc, rmse_smc, bias_smc, wv_smc = smc

    for label, mean, se, var, cil, cir, rmse, bias, wv in [
        ('Coupled-sum', mean_cs,  se_cs,  var_cs,  cil_cs,  cir_cs,  rmse_cs,  bias_cs,  wv_cs),
        ('Standard MC', mean_smc, se_smc, var_smc, cil_smc, cir_smc, rmse_smc, bias_smc, wv_smc),
    ]:
        print(f"\n--- {title} {label} ---")
        header = f"{'Budget':>10}  {'Mean':>10}  {'Variance':>10}  {'Bias':>10}  {'RMSE':>10}  {'Work×Var':>10}  {'95% CI':>28}"
        print(header)
        print('-' * len(header))
        for i, b in enumerate(budgets):
            ci_str = f"[{cil[i]:.6f}, {cir[i]:.6f}]"
            print(f"{b:>10}  {mean[i]:>10.6f}  {var[i]:>10.2e}  {bias[i]:>10.2e}  {rmse[i]:>10.2e}  {wv[i]:>10.2e}  {ci_str:>28}")

    _, axes = plt.subplots(1, 3, figsize=(16, 4))

    for mean, se, label, fmt in [(mean_cs, se_cs, 'Coupled-sum', 'o-'), (mean_smc, se_smc, 'Standard MC', 's--')]:
        axes[0].errorbar(budgets, mean, yerr=1.96 * se, fmt=fmt, capsize=5, label=label)
    axes[0].axhline(true_val, color='r', linestyle=':', label=f'Ref = {true_val:.6f}')
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Budget')
    axes[0].set_ylabel('Estimated mean')
    axes[0].set_title(f'{title} Mean ± 95% CI')
    axes[0].legend()

    for rmse, label, fmt in [(rmse_cs, 'Coupled-sum', 'o-'), (rmse_smc, 'Standard MC', 's--')]:
        axes[1].loglog(budgets, rmse, fmt, label=label)
    axes[1].set_xlabel('Budget')
    axes[1].set_ylabel('RMSE')
    axes[1].set_title(f'{title} RMSE vs budget')
    axes[1].legend()

    for wv, label, fmt in [(wv_cs, 'Coupled-sum', 'o-'), (wv_smc, 'Standard MC', 's--')]:
        axes[2].loglog(budgets, wv, fmt, label=label)
    axes[2].set_xlabel('Budget')
    axes[2].set_ylabel('Work × Variance')
    axes[2].set_title(f'{title} Work–variance product')
    axes[2].legend()

    plt.tight_layout()
    plt.show()


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

def cir_european(v):
    return np.exp(-0.05 * CIR.maturity) * np.maximum(v - 0.25, 0.0)

def milstein_cir_scheme(v0, _dt, dw):
    if dw.ndim == 1:
        return drift_implicit_milstein_cir(v0, dw[np.newaxis, :])[:, -1][0]
    return drift_implicit_milstein_cir(v0, dw)[:, -1]

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