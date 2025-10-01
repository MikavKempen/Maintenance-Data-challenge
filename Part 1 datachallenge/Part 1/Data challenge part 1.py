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

## Count each severity type
minor_count = (all_severities == 0.1).sum()
moderate_count = (all_severities == 1).sum()
major_count = (all_severities == 10).sum()
total_count = len(all_severities)

## Probabilities. The probability of arrival is the same as the empirical probability since the arrivals and severities are i.i.d.
p_minor = minor_count / total_count
p_moderate = moderate_count / total_count
p_major = major_count / total_count

print(f"Probability of minor damage: {p_minor:.3f}") # 51.5%
print(f"Probability of moderate damage: {p_moderate:.3f}") # 33.8%
print(f"Probability of major damage: {p_major:.3f}") # 14.6%

# Task c
ecl = 1 # month

## Expected repair cost per cycle
c_rep = (lambda_hat * days_in_month) * (p_moderate * c_moderate + p_major * c_major)

## Checking if it's even possible to cross the threshold
### Compute total severity per month (sum across all damage columns)
monthly_total_severity = df[damage_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1)

plt.figure(figsize=(12,6))
plt.plot(df["YearMonth"], monthly_total_severity, marker="o", label="Total Severity per Month")
plt.axhline(y=penalty_threshold, color="r", linestyle="--", label=f"Threshold = {penalty_threshold}")
plt.xticks(rotation=45)
plt.ylabel("Total Severity Score")
plt.title("Monthly Total Severity vs Safety Threshold")
plt.legend()
plt.tight_layout()
plt.show()

## Count months exceeding the threshold. Use data-driven approach to determine the probability of crossing the threshold.
## This is acceptable because we have quite a big dataset and the event is not super likely as was seen in the graph.
months_over_threshold = (monthly_total_severity > penalty_threshold).sum()
total_months = len(monthly_total_severity)
percentage_over_threshold = 100 * months_over_threshold / total_months

print(f"Months over threshold: {months_over_threshold} / {total_months}")
print(f"Percentage of months over threshold: {percentage_over_threshold:.2f}%")

# Expected penalty cost per cycle (thousands of euros)
p_penalty = percentage_over_threshold/100 # Conclusion from the data
c_fine = penalty * p_penalty

# Expected cycle cost (thousands of euros)
ecc = c_i + c_rep + c_fine

# Expected monthly cost (since cycle length = 1 month)
g = ecc / ecl

print(f"Expected monthly maintenance cost: {g:.2f} thousand euros") # 30.57 thousand euros

# task d
## Define a range of tau values (days)
tau_values = np.arange(5, 365/2+1, 1)  # from 5 days to 6 months (182.5 days). Steps of 1 day

## Expected repair cost per damage
E_repair_per_damage = p_moderate * c_moderate + p_major * c_major

monthly_costs = []
for tau in tau_values:
    ### Expected repair cost per cycle
    c_rep_tau = (lambda_hat * tau) * E_repair_per_damage

    ### Approximate penalty probability: scale linearly with tau
    p_penalty_tau = min(1.0, p_penalty * tau / days_in_month)
    c_fine_tau = penalty * p_penalty_tau

    ### Expected cycle cost
    ecc_tau = c_i + c_rep_tau + c_fine_tau

    ### Expected monthly cost (per 30 days)
    g_tau = ecc_tau / tau * 30
    monthly_costs.append(g_tau)

## Find tau that minimizes expected monthly cost
min_index = np.argmin(monthly_costs)
tau_star = tau_values[min_index]
min_cost = monthly_costs[min_index]

print(f"Optimal inspection interval tau* = {tau_star:.1f} days")
print(f"Expected monthly cost at tau* = {min_cost:.2f} thousand euros")

# Plot expected monthly cost over tau values
plt.figure(figsize=(10,5))
plt.plot(tau_values, monthly_costs, marker='o')
plt.axvline(x=tau_star, color='r', linestyle='--', label=f"Optimal tau* = {tau_star:.1f} days")
plt.xlabel("Inspection interval τ (days)")
plt.ylabel("Expected monthly maintenance cost (thousand €)")
plt.title("Expected monthly maintenance cost vs inspection interval")
plt.legend()
plt.grid(True)
plt.show()



#d with taking r into account
# ===== Task (d): Optimal τ using Monte Carlo penalty probability (no linear approx) =====

# Search grid for inspection interval τ (respect "at least twice/year" → τ ≤ 182 days)
tau_values = np.arange(5, int(365/2) + 1, 1)

# Expected repair cost per damage (from Task b)
E_repair_per_damage = p_moderate * c_moderate + p_major * c_major

# Monte Carlo settings
n_mc = 2000                              # number of simulated cycles per τ (increase for smoother results)
rng = np.random.default_rng(42)           # reproducible random generator

# Severity categories (values) and their probabilities (from Task b)
severity_values = np.array([0.1, 1.0, 10.0])
severity_probs  = np.array([p_minor, p_moderate, p_major])

def estimate_penalty_prob_tau(tau, n_mc=20000):
    """
    Estimate P(sum of severities in a τ-day cycle > penalty_threshold)
    for the compound Poisson model:
      N(τ) ~ Poisson(lambda_hat * τ), severities i.i.d. in {0.1, 1, 10}.
    """
    lam_tau = lambda_hat * tau
    # Draw number of damages per cycle for all simulations
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

# Compute expected monthly cost for each τ
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

# Find τ* that minimizes the expected monthly cost
min_index = np.argmin(monthly_costs)
tau_star = tau_values[min_index]
min_cost = monthly_costs[min_index]

print(f"Optimal inspection interval tau* = {tau_star:.1f} days")
print(f"Expected monthly cost at tau* = {min_cost:.2f} thousand euros")

# Plot: Expected monthly cost vs τ (Monte Carlo only)
plt.figure(figsize=(10,5))
plt.plot(tau_values, monthly_costs, marker='o', linewidth=1, label="Cost (MC penalty)")
plt.axvline(x=tau_star, color='r', linestyle='--', label=f"tau* = {tau_star:.1f} d")
plt.xlabel("Inspection interval τ (days)")
plt.ylabel("Expected monthly maintenance cost (thousand €)")
plt.title("Expected monthly maintenance cost vs τ (Monte Carlo penalty)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional diagnostic: plot penalty probability vs τ
plt.figure(figsize=(10,5))
plt.plot(tau_values, penalty_probs, marker='o', linewidth=1)
plt.xlabel("Inspection interval τ (days)")
plt.ylabel("Penalty probability per cycle, P(Sum severity > 20)")
plt.title("Penalty probability per inspection cycle vs τ (Monte Carlo)")
plt.grid(True)
plt.tight_layout()
plt.show()


