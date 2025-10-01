#Imports
import numpy as np
import pandas as pd
from math import factorial
from matplotlib import pyplot as plt
import os
import math

# Read Data
## Folder where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "inspection_log.csv")

df = pd.read_csv(csv_path)
print(df.head())

# Parameters
c_i = 10 # Inspection cost (thousands euros)
c_minor = 0 # Repair cost for minor damage  (thousands euros)
c_moderate = 1 # Repair cost for moderate damage (thousands euros)
c_major = 10 # Repair cost for major damage (thousands euros)
penalty = 500 # Penalty cost (thousands euros)
penalty_threshold = 20 # Threshold of total severity score for penalty
days_in_month = 30 # Assume there are 30 days in every month
inspection_lambda = 0.5

# Task a
## Identify damage columns
damage_cols = [col for col in df.columns if "Damage" in col]

## Count damages per month (non-empty entries)
num_damages = df[damage_cols].apply(pd.to_numeric, errors='coerce').notna().sum(axis=1)

## Histogram of number of damages per month
df["YearMonth"] = df["Year"].astype(str) + "-" + df["Month"].astype(str)

plt.figure(figsize=(12,6))
plt.bar(df["YearMonth"], num_damages, color="skyblue", edgecolor="black")
plt.xticks(rotation=45)
plt.ylabel("Number of Damages")
plt.title("Number of Damages per Month-Year")
plt.tight_layout()
plt.show()

## Average monthly damages
X_bar = num_damages.mean()

## MLE and MoM estimate (is the same, check hand calculations)
lambda_hat = X_bar / days_in_month
print("Estimated arrival rate λ (damages/day):", lambda_hat) # = 0.07222222222222222

## Including calculating the standard error and confidence interval
n = len(num_damages)
se = math.sqrt(lambda_hat / (n*days_in_month))
ci_low = lambda_hat - 1.96*se
ci_high = lambda_hat + 1.96*se
print(f"95% CI: [{ci_low}, {ci_high}]") # = 95% CI: [0.059806978704476044, 0.08463746573996839]

# Task b
## Flatten all severity values (to make it a 1d digestible array), convert to numeric, drop NAN
all_severities = df[damage_cols].apply(pd.to_numeric, errors='coerce').values.flatten()
all_severities = pd.Series(all_severities).dropna()

## Making sure there are no errors/typos
print(all_severities.unique())

## Count each severity type
minor_count = (all_severities == 0.1).sum()
moderate_count = (all_severities == 1).sum()
major_count = (all_severities == 10).sum()
total_count = len(all_severities)

## Calculating the probabilities. The probability of arrival is the same as the empirical
## probability since the arrivals and severities are i.i.d. Also check hand calculations
p_minor = minor_count / total_count
p_moderate = moderate_count / total_count
p_major = major_count / total_count

print(f"Probability of minor damage: {p_minor:.3f}") # 51.5%
print(f"Probability of moderate damage: {p_moderate:.3f}") # 33.8%
print(f"Probability of major damage: {p_major:.3f}") # 14.6%

# task c (data-driven)
ecl = 1  # month

## Expected repair cost per cycle
mu = lambda_hat * days_in_month  # estimated mean damages per month
E_repair_per_damage = p_moderate * c_moderate + p_major * c_major + p_minor * c_minor#c_minor is 0 so it will eliminate itself
c_rep = mu * E_repair_per_damage

## Total severity per month
monthly_total_severity = df[damage_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

## Checking if it's even possible to cross the threshold
plt.figure(figsize=(12,6))
plt.plot(df["YearMonth"], monthly_total_severity, marker="o", label="Total Severity per Month")
plt.axhline(y=penalty_threshold, color="r", linestyle="--", label=f"Threshold = {penalty_threshold}")
plt.xticks(rotation=45)
plt.ylabel("Total Severity Score")
plt.title("Monthly Total Severity vs Safety Threshold")
plt.legend()
plt.tight_layout()
plt.show()

## Data-driven penalty probability
months_over_threshold = (monthly_total_severity > penalty_threshold).sum() # 2
total_months = len(monthly_total_severity)# 60
p_penalty_data = months_over_threshold / total_months

## Expected penalty cost
c_penalty_data = penalty * p_penalty_data

## Expected monthly cost
ecc_data = c_i + c_rep + c_penalty_data
g_data = ecc_data / ecl

print("\n--- Task C (data-driven) ---")
print(f"Empirical penalty probability = {p_penalty_data:.4f} ({p_penalty_data*100:.2f}%)")# Empirical penalty probability = 3.33%
print(f"Expected monthly maintenance cost (data-driven) = {g_data:.2f} thousand euros")# Expected monthly maintenance cost = 30.57 thousand euros

# task c (theory-based)
## Reusable theory-based probability of penalty function
def penalty_probability(mu, threshold=20, p_minor=p_minor, p_moderate=p_moderate, p_major=p_major):
    # Conservative truncation of Poisson distribution
    n_max = int(math.ceil(mu + 10 * math.sqrt(max(mu, 1.0))))
    p_penalty_theory = 0.0

    for n in range(0, n_max + 1):
        pN = math.exp(-mu) * (mu ** n) / math.factorial(n)
        if n == 0:
            Pr_gt = 0.0
        else:
            Pr_le = 0.0
            # enumerate possible (k minors, m moderates, q majors)
            for q in range(0, n + 1):
                for m in range(0, n - q + 1):
                    k = n - m - q
                    total_sev = 0.1 * k + 1.0 * m + 10.0 * q
                    if total_sev <= threshold + 1e-12:
                        coeff = math.factorial(n) / (
                            math.factorial(k) * math.factorial(m) * math.factorial(q)
                        )
                        Pr_le += coeff * (p_minor ** k) * (p_moderate ** m) * (p_major ** q)
            Pr_gt = 1.0 - Pr_le
        p_penalty_theory += pN * Pr_gt

    return p_penalty_theory

## Call the reusable function
p_penalty_theory = penalty_probability(mu, threshold=penalty_threshold)

## Expected penalty cost (theory-based)
c_penalty_theory = penalty * p_penalty_theory

## Expected monthly cost (theory-based)
ecc_theory = c_i + c_rep + c_penalty_theory
g_theory = ecc_theory / ecl

print("\n--- Task C (theory-based) ---")
print(f"Theoretical penalty probability = {p_penalty_theory:.4f} ({p_penalty_theory*100:.2f}%)")
print(f"Expected monthly maintenance cost (theory-based) = {g_theory:.2f} thousand euros")

# task d (data-driven)
## Candidate inspection intervals (days)
tau_values = np.arange(5, int(365/2) + 1, 1) # Start at 5 days, check every day for 365/2 days to make sure
                                          # we can have at least 2 inspections per day
monthly_costs_data = []
for tau in tau_values:
    cycle_costs = []
    cycle_length_months = max(1, int(round(tau / 30))) # make sure it's at least 30 to allow for computation so minimum tau is 30 days
    ## Split dataset into cycles of tau days
    for start in range(0, len(df), cycle_length_months):
        cycle = df.iloc[start:start + cycle_length_months]
        if cycle.empty:
            continue

        ## Inspection cost
        cost = c_i

        # Repair cost (from actual data)
        damages = cycle[damage_cols].apply(pd.to_numeric, errors="coerce").values.flatten()
        damages = pd.Series(damages).dropna()
        cost += (damages == 1).sum() * c_moderate
        cost += (damages == 10).sum() * c_major

        # Penalty
        total_severity = cycle[damage_cols].apply(pd.to_numeric, errors="coerce").sum().sum()
        if total_severity > penalty_threshold:
            cost += penalty

        cycle_costs.append(cost)

    if len(cycle_costs) > 0:
        # normalize cost to monthly
        g_tau_data = np.mean(cycle_costs) / tau * 30
        monthly_costs_data.append(g_tau_data)
    else:
        monthly_costs_data.append(np.nan)

## Find τ* that minimizes the expected monthly cost (data-driven)
monthly_costs_array = np.array(monthly_costs_data)
min_index_data = np.nanargmin(monthly_costs_array)
tau_star_data = tau_values[min_index_data]
min_cost_data = monthly_costs_array[min_index_data]

print("\n--- Task D (data-driven) ---")
print(f"Optimal inspection interval tau* (data-driven) = {tau_star_data:.1f} days")
print(f"Expected monthly cost at tau* = {min_cost_data:.2f} thousand euros")

# task d (theory-based)
monthly_costs_theory = []
for tau in tau_values:
    variable_mu = lambda_hat * tau

    # Repair cost
    c_rep_tau = variable_mu * E_repair_per_damage

    # Penalty probability
    p_penalty_tau = penalty_probability(variable_mu, penalty_threshold)
    c_fine_tau = penalty * p_penalty_tau

    # Expected cycle cost
    ecc_tau = c_i + c_rep_tau + c_fine_tau

    # Monthly cost
    g_tau = ecc_tau / tau * 30
    monthly_costs_theory.append(g_tau)

## Convert list to array for easier handling
monthly_costs_theory_array = np.array(monthly_costs_theory)

## Find τ* that minimizes the expected monthly cost (theory-based)
min_index_theory = np.nanargmin(monthly_costs_theory_array)
tau_star_theory = tau_values[min_index_theory]
min_cost_theory = monthly_costs_theory_array[min_index_theory]

print("\n--- Task D (theory-based) ---")
print(f"Optimal inspection interval tau* (theory-based) = {tau_star_theory:.1f} days")
print(f"Expected monthly cost at tau* = {min_cost_theory:.2f} thousand euros")

# task d with Monte Carlo penalty probability
## Monte Carlo settings
n_mc = 2000                              # number of simulated cycles per τ (increase for smoother results)
rng = np.random.default_rng(42)           # reproducible random generator

## Severity categories (values) and their probabilities (in arrays for ease)
severity_values = np.array([0.1, 1.0, 10.0])
severity_probs  = np.array([p_minor, p_moderate, p_major])

## Estimate the probability of a penalty using Monte Carlo simulation
def estimate_penalty_prob_tau(tau, n_mc=20000):
    """
    Estimate P(sum of severities in a τ-day cycle > penalty_threshold)
    for the compound Poisson model:
      N(τ) ~ Poisson(lambda_hat * τ), severities i.i.d. in {0.1, 1, 10}.
    """
    lam_tau = lambda_hat * tau
    ### Draw number of damages per cycle for all simulations
    Ns = rng.poisson(lam_tau, size=n_mc)

    exceed = 0
    for N in Ns:
        if N == 0:
            continue  # no damages → sum = 0 → cannot exceed
        # Draw N severities for this simulated cycle
        sev = rng.choice(severity_values, size=N, p=severity_probs)
        if sev.sum() > penalty_threshold:
            exceed += 1
    return exceed / n_mc

## Compute expected monthly cost for each τ
monthly_costs = []
penalty_probs = []  # store Monte Carlo Pτ for diagnostics/plotting
for tau in tau_values:
    # Expected repair cost per cycle (analytical)
    c_rep_tau = (lambda_hat * tau) * E_repair_per_damage

    # Penalty probability per cycle via Monte Carlo
    p_penalty_tau = estimate_penalty_prob_tau(tau, n_mc=n_mc)
    penalty_probs.append(p_penalty_tau)
    c_fine_tau = penalty * p_penalty_tau

    # Expected cycle cost and monthly cost (30-day month for reporting)
    ecc_tau = c_i + c_rep_tau + c_fine_tau
    g_tau = ecc_tau / tau * 30
    monthly_costs.append(g_tau)

## Convert list to array for easier handling
monthly_costs_mc_array = np.array(monthly_costs)

## Find τ* that minimizes the expected monthly cost (Monte Carlo)
min_index_mc = np.nanargmin(monthly_costs_mc_array)
tau_star_mc = tau_values[min_index_mc]
min_cost_mc = monthly_costs_mc_array[min_index_mc]

print("\n--- Task D (Monte Carlo-based) ---")
print(f"Optimal inspection interval tau* (Monte Carlo) = {tau_star_mc:.1f} days")
print(f"Expected monthly cost at tau* = {min_cost_mc:.2f} thousand euros")

## Plot penalty probability vs tau for diagnostics
plt.figure(figsize=(10,5))
plt.plot(tau_values, penalty_probs, marker='o', linewidth=1)
plt.xlabel("Inspection interval τ (days)")
plt.ylabel("Penalty probability per cycle, P(Sum severity > 20)")
plt.title("Penalty probability per inspection cycle vs τ (Monte Carlo)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot all three approaches to task d in monthly costs vs tau
plt.figure(figsize=(12,6))

## Plot theory-based expected monthly cost
plt.plot(tau_values, monthly_costs_theory, marker='o', linestyle='-', color='blue', label='Theory-based')

## Plot data-driven expected monthly cost
plt.plot(tau_values, monthly_costs_data, marker='x', linestyle='--', color='orange', label='Data-driven')

## Plot Monte Carlo-based expected monthly cost
plt.plot(tau_values, monthly_costs, marker='s', linestyle='-.', color='green', label='Monte Carlo-based')

## Mark tau* for each method
plt.axvline(x=tau_star_theory, color='blue', linestyle=':', label=f'Tau* theory ≈ {tau_star_theory:.1f} d')
plt.axvline(x=tau_star_data, color='orange', linestyle=':', label=f'Tau* data ≈ {tau_star_data:.1f} d')
plt.axvline(x=tau_star_mc, color='green', linestyle=':', label=f'Tau* Monte Carlo ≈ {tau_star_mc:.1f} d')

## Labels, title, legend
plt.xlabel("Inspection interval τ (days)")
plt.ylabel("Expected monthly maintenance cost (thousand €)")
plt.title("Expected monthly maintenance cost vs inspection interval (all methods)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot theory-based and monte carlo in monthly costs vs tau
plt.figure(figsize=(12,6))

# Plot theory-based expected monthly cost
plt.plot(tau_values, monthly_costs_theory, marker='o', linestyle='-', color='blue', label='Theory-based')

# Plot Monte Carlo-based expected monthly cost
plt.plot(tau_values, monthly_costs, marker='s', linestyle='-.', color='green', label='Monte Carlo-based')

# Mark tau* for each method
plt.axvline(x=tau_star_theory, color='blue', linestyle=':', label=f'Tau* theory ≈ {tau_star_theory:.1f} d')
plt.axvline(x=tau_star_mc, color='green', linestyle=':', label=f'Tau* Monte Carlo ≈ {tau_star_mc:.1f} d')

# Labels, title, legend
plt.xlabel("Inspection interval τ (days)")
plt.ylabel("Expected monthly maintenance cost (thousand €)")
plt.title("Expected monthly maintenance cost vs inspection interval")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



