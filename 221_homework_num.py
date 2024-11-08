import numpy as np
from tabulate import tabulate
from scipy.stats import norm

def option_payoff(S, K, option_type):
    if option_type == 'call':
        return np.maximum(S - K, 0)
    elif option_type == 'put':
        return np.maximum(K - S, 0)
    else:
        raise ValueError('Option type must be either "call" or "put".')
    
def black_scholes(S0, K, option_type, T, r, q, sigma):
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError('Option type must be either "call" or "put".')

def plain_monte_carlo(S0, K, option_type, T, r, q, sigma, n_steps, n_simulation):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    payoff = np.zeros(n_simulation, dtype=float)
    
    for i in range(n_simulation):
        S = S0
        for _ in range(n_steps):
            S *= np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.random.normal() * sqrt_dt)
        payoff[i] = option_payoff(S, K, option_type)
    error_estimate = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_simulation)
    return np.exp(-r * T) * np.mean(payoff), error_estimate

def antithetic_variate(S0, K, option_type, T, r, q, sigma, n_steps, n_simulation):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    payoff_down = np.zeros(n_simulation, dtype=float)
    payoff_up = np.zeros(n_simulation, dtype=float)
    for i in range(n_simulation):
        S_up = S0
        S_down = S0
        for _ in range(n_steps):
            z = np.random.normal()
            S_up *= np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * z * sqrt_dt)
            S_down *= np.exp((r - q - 0.5 * sigma ** 2) * dt - sigma * z * sqrt_dt)
        payoff_down[i] = option_payoff(S_down, K, option_type)
        payoff_up[i] = option_payoff(S_up, K, option_type)
    payoff = (payoff_down + payoff_up) / 2
    error_estimate = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_simulation)
    return np.exp(-r * T) * np.mean(payoff), error_estimate

def control_variate(S0, K, option_type, T, r, q, sigma, n_steps, n_simulation):
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    payoff_init = np.zeros(n_simulation, dtype=float)
    f = np.zeros(n_simulation, dtype=float)
    
    for i in range(n_simulation):
        S = S0
        for _ in range(n_steps):
            S *= np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.random.normal() * sqrt_dt)
        f[i] = S
        payoff_init[i] = option_payoff(S, K, option_type)
        
    mu = S0 * np.exp((r - q) * T)
    beta_estimate = np.cov(payoff_init, f)[0][1] / np.var(f)
    payoff = payoff_init - beta_estimate * (f - mu)
    error_estimate = np.exp(-r * T) * np.std(payoff) / np.sqrt(n_simulation)
    return np.exp(-r * T) * np.mean(payoff), error_estimate
    
# PARAMETERS  
S0 = 100
K = 100
option_type = 'call'
T = 1
r = 0.06
q = 0.06
sigma = 0.35

# SIMULATION PARAMETERS
n_steps = 100
np.random.seed(110124)
n_simulation = 4000

# PRINT RESULTS
estimations = [
    ["Plain Monte Carlo", f"{plain_monte_carlo(S0, K, option_type, T, r, q, sigma, n_steps, n_simulation)[0]:.2f}"],
    ["Antithetic Variate", f"{antithetic_variate(S0, K, option_type, T, r, q, sigma, n_steps, n_simulation)[0]:.2f}"],
    ["Control Variate", f"{control_variate(S0, K, option_type, T, r, q, sigma, n_steps, n_simulation)[0]:.2f}"],
    ["Black-Scholes", f"{black_scholes(S0, K, option_type, T, r, q, sigma):.2f}"]
]

errors = [
    ["Plain Monte Carlo", f"{plain_monte_carlo(S0, K, option_type, T, r, q, sigma, n_steps, n_simulation)[1]:.2f}"],
    ["Antithetic Variate", f"{antithetic_variate(S0, K, option_type, T, r, q, sigma, n_steps, n_simulation)[1]:.2f}"],
    ["Control Variate", f"{control_variate(S0, K, option_type, T, r, q, sigma, n_steps, n_simulation)[1]:.2f}"]
]

print("### Option Price Estimations ###")
print(tabulate(estimations, headers=["Method", "Estimation"], tablefmt="grid"))

print("\n### Error Estimates ###")
print(tabulate(errors, headers=["Method", "Error"], tablefmt="grid"))
