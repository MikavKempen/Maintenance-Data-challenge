"""
    1CM290 Maintenance Optimziation and Engineering (Lecturer: J. Lee)
    Assignment: Data Cahllenges 2025
    Challenge: Maintenance of solar power plant
    This is a template for the assignment.
    You may fill the parts <YOUR CODE HERE> or add new parts as you need.
    You may change as you need.
    Please add concise and comprehensive comments.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import gurobipy as grb

# Read Data
df_west = pd.read_csv('data_solar_power_plant/spu_efficiency_West.csv')
print(df_west)
df_east = pd.read_csv('data_solar_power_plant/spu_efficiency_East.csv')
print(df_east)

# -------------------------
# Task (a)
# -------------------------
# Data preprocessing
# Convert efficiency -> degradation, enforce bounds and non-increase per SPU,
# then compute daily positive increments and pool across SPUs.
day_col = df_west.columns[0]
eff_west = df_west.drop(columns=[day_col]).copy()

# Clip to [0,1] and enforce non-increasing sequences (handles noise / unlogged cleaning)
eff_west = eff_west.clip(lower=0.0, upper=1.0)
eff_west = eff_west.apply(lambda col: np.minimum.accumulate(col.to_numpy()), axis=0)

# Degradation and daily increments
deg_west = 1.0 - eff_west
ddeg_west = deg_west.diff().iloc[1:, :]  # drop first NaN row

# Keep strictly positive daily increments (Gamma increments) — two-step filter avoids warnings
x = ddeg_west.to_numpy().ravel()
x = x[np.isfinite(x)]
x = x[x > 0]

# (Optional) quick sanity prints
print(f"Number of daily increments (all): {ddeg_west.size}")
print(f"Number of positive increments used for fitting: {x.size}")
print(f"Mean increment: {x.mean():.6f}, Var increment: {x.var(ddof=0):.6f}")

# Parameter estimation (Method of Moments for Gamma)
# For Gamma(k=alpha, theta=beta): mean = alpha*beta, var = alpha*beta^2
m = x.mean()
v = x.var(ddof=0)

# Guard against degenerate variance
if v <= 0 or not np.isfinite(v) or not np.isfinite(m) or m <= 0:
    alp = 1.0
    bet = m if m > 0 and np.isfinite(m) else 1.0
else:
    alp = (m * m) / v
    bet = v / m

print(f"(a) MoM, alpha = {alp:.2f}, beta = {bet:.4f}")

# -------------------------
# Visualization (West / East)
# -------------------------
def clean_efficiency(df):
    day_col = df.columns[0]
    M = df.set_index(day_col).copy()
    M = M.apply(pd.to_numeric, errors='coerce')
    # Mask impossible values (>1 or <0)
    M = M.mask((M < 0) | (M > 1))
    # Clip between 0 and 1
    M = M.clip(lower=0, upper=1)
    return M

# ---- West
eff_west_plot = clean_efficiency(df_west)

plt.figure(figsize=(10, 4))
for col in eff_west_plot.columns:
    plt.plot(eff_west_plot.index, eff_west_plot[col], linewidth=1, alpha=0.8)
plt.axhline(0.8, linestyle='--', color='tab:blue', label='Turn-off threshold')
plt.xlabel('Day')
plt.ylabel('Efficiency')
plt.title('West area: SPU efficiencies over time (15 panels)')
plt.legend()
plt.xlim(df_west['Day'].min(), df_west['Day'].max())  # full range (1–730)
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# ---- East
eff_east_plot = clean_efficiency(df_east)
plt.figure(figsize=(10, 4))
for col in eff_east_plot.columns:
    plt.plot(eff_east_plot.index, eff_east_plot[col], linewidth=1.2, alpha=0.9)
plt.axhline(0.8, linestyle='--', color='tab:blue', label='Turn-off threshold')
plt.xlabel('Day')
plt.ylabel('Efficiency')
plt.title('East area: SPU efficiencies (last month, 5 panels)')
plt.legend()
plt.xlim(df_east['Day'].min(), df_east['Day'].max())
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()


# -------------------------
# Task (d)
# -------------------------
# RUL estimation
RUL = []
# YOUR CODE HERE
print(f"(b) RUL of East area: {RUL}")

# -------------------------
# Task (e)
# -------------------------
# Read Data
df_cost = pd.read_csv('data_solar_power_plant/cost_cleaning.csv')
print(df_cost)
df_rul = pd.read_csv('data_solar_power_plant/RUL_North.csv')
print(df_rul)

# Parameters
C_P = df_cost['C_P, power price'].to_numpy()
C_D = df_cost['C_D, daily charge'].to_numpy()
C_U = df_cost['C_U, unit charge'].to_numpy()
RUL_i = df_rul['RUL (days)'].to_numpy()
max_clean = 3 # Max number of cleaning in a day
I = RUL_i.size
T = df_cost.shape[0]

# Create a new model
m = grb.Model()

# Create variables
# YOUR CODE HERE

# Set objective function
f = 0 # YOUR CODE HERE
m.setObjective(f, grb.GRB.MINIMIZE)

# Add constraints
# YOUR CODE HERE

# Solve it!
m.optimize()
# YOUR CODE HERE

print(f"(f) Optimal objective value: {m.objVal:.2f}")
