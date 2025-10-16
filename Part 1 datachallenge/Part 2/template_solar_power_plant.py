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
# print(df_east)


# -------------------------
# Data quality overview (West)
# -------------------------

# Identify columns
day_col = df_west.columns[0]
eff_cols = df_west.columns.drop(day_col)

# Coerce to numeric (non-numerics -> NaN)
df_west_num = df_west[eff_cols].apply(pd.to_numeric, errors='coerce')

# Counts
total_values = df_west_num.size
after_coerce_nan = df_west_num.isna().sum().sum()
original_nan = df_west[eff_cols].isna().sum().sum()
non_numeric_count = max(0, after_coerce_nan - original_nan)  # entries that became NaN due to coercion

out_of_bounds_mask = (df_west_num < 0) | (df_west_num > 1)
out_of_bounds_count = out_of_bounds_mask.sum().sum()

zero_mask = df_west_num == 0.0
zero_count = zero_mask.sum().sum()

valid_values = total_values - after_coerce_nan

print("---- Data Quality Summary (West Area) ----")
print(f"Total data points (excluding day column): {total_values}")
print(f"Valid numeric entries:                   {valid_values}")
print(f"Missing or NaN values (after coercion):  {after_coerce_nan}")
print(f"Non-numeric original entries:            {non_numeric_count}")
print(f"Values outside [0, 1]:                   {out_of_bounds_count}")
print(f"Values exactly 0.0:                      {zero_count}")
print("------------------------------------------")
print(f"Share of invalid or missing entries: {(after_coerce_nan + out_of_bounds_count) / total_values:.2%}")

# -------------------------
# Top 5 panels with breakdown: invalid vs zero values
# -------------------------

# Invalid = NaN or out-of-bounds
invalid_mask = df_west_num.isna() | (df_west_num < 0) | (df_west_num > 1)
invalid_counts_per_panel = invalid_mask.sum()

# Exact zeros
zero_counts_per_panel = zero_mask.sum()

# Total problematic = invalid OR zero
invalid_or_zero_counts = invalid_counts_per_panel + zero_counts_per_panel

# Select top 5 panels by total problematic values
top5_panels = invalid_or_zero_counts.sort_values(ascending=False).head(5).index

# Subset for printing/plotting
top5_invalid = invalid_counts_per_panel[top5_panels]
top5_zero = zero_counts_per_panel[top5_panels]

print("\nTop 5 panels with most invalid or zero entries (breakdown):")
print(pd.DataFrame({
    "Invalid entries": top5_invalid,
    "Zero entries": top5_zero,
    "Total": top5_invalid + top5_zero
}).sort_values(by="Total", ascending=False))

# Stacked bar chart
plt.figure(figsize=(7, 4))
plt.bar(top5_panels, top5_invalid, label='Invalid (<0, >1, or NaN)', edgecolor='black')
plt.bar(top5_panels, top5_zero, bottom=top5_invalid, label='Exact 0.0', edgecolor='black')
plt.ylabel('Count of problematic entries')
plt.xlabel('SPU (Panel)')
plt.title('Top 5 panels with most invalid or zero entries (West area)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# -------------------------
# Integrated: ensure cleaned frame exists, then interpolate NaNs and plot
# -------------------------

# 0) Normalize any weird tuple column names (safe no-op if already fine)
df_west.columns = [''.join(c) if isinstance(c, tuple) else c for c in df_west.columns]

# 1) Ensure eff_west_clean + day_col exist (matches your earlier cleaning logic)
if 'day_col' not in locals():
    day_col = df_west.columns[0]

if 'eff_west_clean' not in locals():
    eff_west_clean = df_west.copy()
    # numeric conversion for panel columns
    for col in df_west.columns[1:]:
        eff_west_clean[col] = pd.to_numeric(df_west[col], errors='coerce')
    # set out-of-bounds to NaN
    mask_invalid = (eff_west_clean[df_west.columns[1:]] < 0) | (eff_west_clean[df_west.columns[1:]] > 1)
    eff_west_clean.loc[:, df_west.columns[1:]] = eff_west_clean[df_west.columns[1:]].mask(mask_invalid)

# 2) Interpolate only NaNs (keep true zeros), clip back to [0,1]
vals_cols = [c for c in eff_west_clean.columns if c != day_col]

# make sure day is numeric and sorted
eff_west_clean[day_col] = pd.to_numeric(eff_west_clean[day_col], errors='coerce')
eff_west_interp = eff_west_clean.sort_values(by=day_col).copy()

nan_before = eff_west_interp[vals_cols].isna().sum().sum()

eff_west_interp[vals_cols] = eff_west_interp[vals_cols].apply(
    lambda s: s.interpolate(method='linear', limit_direction='both')
)
eff_west_interp[vals_cols] = eff_west_interp[vals_cols].clip(lower=0.0, upper=1.0)

nan_after = eff_west_interp[vals_cols].isna().sum().sum()
print(f"NaNs before interpolation: {nan_before}")
print(f"NaNs after  interpolation: {nan_after}  (filled {nan_before - nan_after})")

# 3) Visualization
plt.figure(figsize=(10, 4))
for col in vals_cols:
    plt.plot(eff_west_interp[day_col], eff_west_interp[col], linewidth=1, alpha=0.9)
plt.axhline(0.8, linestyle='--', label='Turn-off threshold')
plt.xlabel('Day')
plt.ylabel('Efficiency')
plt.title('West area: SPU efficiencies after interpolation (invalid removed, NaNs filled)')
plt.legend()
plt.xlim(eff_west_interp[day_col].min(), eff_west_interp[day_col].max())
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# -------------------------
# Create cleaned CSV for West (remove <0, >1, and 0.0; interpolate NaNs)
# -------------------------

# 0) Normalize any weird tuple column names (safe no-op if already fine)
df_west.columns = [''.join(c) if isinstance(c, tuple) else c for c in df_west.columns]

# 1) Identify columns
day_col = df_west.columns[0]
panel_cols = df_west.columns[1:]

# 2) Start from a copy and coerce panel columns to numeric
eff_west_clean = df_west.copy()
for col in panel_cols:
    eff_west_clean[col] = pd.to_numeric(eff_west_clean[col], errors='coerce')

# 3) Mark invalids and zeros as NaN
mask_out_of_bounds = (eff_west_clean[panel_cols] < 0) | (eff_west_clean[panel_cols] > 1)
mask_zero          = (eff_west_clean[panel_cols] == 0.0)
eff_west_clean.loc[:, panel_cols] = eff_west_clean[panel_cols].mask(mask_out_of_bounds | mask_zero)

# 4) Make sure day is numeric and sorted by time
eff_west_clean[day_col] = pd.to_numeric(eff_west_clean[day_col], errors='coerce')
eff_west_clean = eff_west_clean.sort_values(by=day_col).reset_index(drop=True)

# 5) Interpolate NaNs per panel along time; fill edges; clip to [0,1]
eff_west_interp = eff_west_clean.copy()
eff_west_interp[panel_cols] = eff_west_interp[panel_cols].apply(
    lambda s: s.interpolate(method='linear', limit_direction='both')
)
eff_west_interp[panel_cols] = eff_west_interp[panel_cols].clip(lower=0.0, upper=1.0)

# 6) Report and save
nan_before = eff_west_clean[panel_cols].isna().sum().sum()
nan_after  = eff_west_interp[panel_cols].isna().sum().sum()
print(f"NaNs before interpolation: {nan_before}")
print(f"NaNs after  interpolation: {nan_after} (filled {nan_before - nan_after})")

out_path = 'data_solar_power_plant/spu_efficiency_West_clean.csv'
eff_west_interp.to_csv(out_path, index=False)
print(f"Cleaned West dataset written to: {out_path}")

# -------------------------
# Task (a): Estimate Gamma (alpha, beta) from West data
# -------------------------
# Assumes you already created `eff_west_interp` (cleaned & interpolated) and `day_col`.

# 1) Take only panel columns and ensure numeric dtype
panel_cols = [c for c in eff_west_interp.columns if c != day_col]
eff_west_fit = eff_west_interp[panel_cols].astype(float)

# 2) Enforce non-increasing efficiency per SPU (removes tiny upticks/noise)
#    This matches the Gamma-process assumption on *degradation* (monotone increase).
eff_west_mono = eff_west_fit.copy()
for c in panel_cols:
    eff_west_mono[c] = np.minimum.accumulate(eff_west_mono[c].to_numpy())

# 3) Convert efficiency -> degradation and take daily increments
deg_west = 1.0 - eff_west_mono
ddeg_west = deg_west.diff().iloc[1:, :]  # drop first NaN row (t=0 has no previous day)

# 4) Pool strictly positive daily increments across all SPUs (Gamma increments)
x = ddeg_west.to_numpy().ravel()
x = x[np.isfinite(x)]
x = x[x > 0]

# 5) Method of Moments for Gamma(k=alpha, theta=beta)
#    mean = alpha*beta, variance = alpha*beta^2
m = x.mean()
v = x.var(ddof=0)

if (v <= 0) or (not np.isfinite(v)) or (not np.isfinite(m)) or (m <= 0):
    alp = 1.0
    bet = m if (m > 0 and np.isfinite(m)) else 1.0
else:
    alp = (m * m) / v
    bet = v / m

print("---- Gamma parameter estimation (West) ----")
print(f"Number of increments considered: {x.size}")
print(f"Mean increment m: {m:.8f}")
print(f"Var  increment v: {v:.8f}")
print(f"(a) MoM estimates: alpha = {alp:.4f}, beta = {bet:.6f}")


#
#
# # -------------------------
# # Task (d)
# # -------------------------
# # RUL estimation
# RUL = []
# # YOUR CODE HERE
# print(f"(b) RUL of East area: {RUL}")
#
# # -------------------------
# # Task (e)
# # -------------------------
# # Read Data
# df_cost = pd.read_csv('data_solar_power_plant/cost_cleaning.csv')
# print(df_cost)
# df_rul = pd.read_csv('data_solar_power_plant/RUL_North.csv')
# print(df_rul)
#
# # Parameters
# C_P = df_cost['C_P, power price'].to_numpy()
# C_D = df_cost['C_D, daily charge'].to_numpy()
# C_U = df_cost['C_U, unit charge'].to_numpy()
# RUL_i = df_rul['RUL (days)'].to_numpy()
# max_clean = 3 # Max number of cleaning in a day
# I = RUL_i.size
# T = df_cost.shape[0]
#
# # Create a new model
# m = grb.Model()
#
# # Create variables
# # YOUR CODE HERE
#
# # Set objective function
# f = 0 # YOUR CODE HERE
# m.setObjective(f, grb.GRB.MINIMIZE)
#
# # Add constraints
# # YOUR CODE HERE
#
# # Solve it!
# m.optimize()
# # YOUR CODE HERE
#
# print(f"(f) Optimal objective value: {m.objVal:.2f}")
