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

# =====================================================================
# Task (d): East area — clean like West, then RUL via Monte Carlo (95% CI)
# =====================================================================

# 0) Normalize any weird tuple column names (safe no-op)
df_east.columns = [''.join(c) if isinstance(c, tuple) else c for c in df_east.columns]

# 1) Identify columns
day_col_e = df_east.columns[0]
panel_cols_e = df_east.columns[1:]

# 2) Coerce to numeric
east_clean = df_east.copy()
for col in panel_cols_e:
    east_clean[col] = pd.to_numeric(east_clean[col], errors='coerce')

# 3) Remove out-of-bounds and exact zeros -> NaN
mask_oob_e = (east_clean[panel_cols_e] < 0) | (east_clean[panel_cols_e] > 1)
mask_zero_e = (east_clean[panel_cols_e] == 0.0)
east_clean.loc[:, panel_cols_e] = east_clean[panel_cols_e].mask(mask_oob_e | mask_zero_e)

# 4) Sort by day and interpolate NaNs; clip to [0,1]
east_clean[day_col_e] = pd.to_numeric(east_clean[day_col_e], errors='coerce')
east_clean = east_clean.sort_values(by=day_col_e).reset_index(drop=True)

east_interp = east_clean.copy()
east_interp[panel_cols_e] = east_interp[panel_cols_e].apply(
    lambda s: s.interpolate(method='linear', limit_direction='both')
)
east_interp[panel_cols_e] = east_interp[panel_cols_e].clip(lower=0.0, upper=1.0)

# (Optional) quick plot
plt.figure(figsize=(9,3.6))
for c in panel_cols_e:
    plt.plot(east_interp[day_col_e], east_interp[c], lw=1)
plt.axhline(0.8, ls='--', label='Turn-off threshold')
plt.title('East area: cleaned/interpolated efficiencies (last month)')
plt.xlabel('Day'); plt.ylabel('Efficiency'); plt.ylim(0,1.05); plt.legend(); plt.tight_layout(); plt.show()

# ------------------------------------------------------------
# Gamma parameters for simulation: use West estimates (alp, bet)
# ------------------------------------------------------------
print(f"Using West-fitted Gamma parameters for simulation: alpha={alp:.4f}, beta={bet:.6f}")

# ------------------------------------------------------------
# Monte Carlo RUL simulation helper
# ------------------------------------------------------------
def simulate_rul_days(y_gap, alpha, beta, nsim=50000, max_days=365):
    """
    Simulate RUL (in whole days) until cumulative Gamma increments exceed y_gap.
    Returns an array of length nsim with integer day counts (1..max_days).
    """
    if y_gap <= 0:
        return np.zeros(nsim, dtype=int)  # already at/below threshold

    # Draw daily increments (nsim x max_days), Gamma(shape=alpha, scale=beta)
    incs = np.random.gamma(shape=alpha, scale=beta, size=(nsim, max_days))
    cums = np.cumsum(incs, axis=1)
    hits = (cums >= y_gap)

    # first hit day: argmax gives first True index; handle rows with no hit
    first_idx = hits.argmax(axis=1)  # 0-based
    no_hit = ~hits.any(axis=1)
    first_idx[no_hit] = max_days - 1

    # Convert to whole days
    return first_idx + 1  # days are 1..max_days

# ------------------------------------------------------------
# Compute per-panel current gap and simulate RUL distribution
# ------------------------------------------------------------
results = []
for c in panel_cols_e:
    e_now = float(east_interp[c].iloc[-1])
    y_gap = max(0.0, e_now - 0.8)  # remaining efficiency drop to threshold
    samples = simulate_rul_days(y_gap, alp, bet, nsim=50000, max_days=365)

    # Summary stats in WHOLE DAYS
    mean_days = float(np.mean(samples))
    q2p5 = int(np.floor(np.quantile(samples, 0.025)))
    q97p5 = int(np.ceil(np.quantile(samples, 0.975)))

    results.append({
        "Panel": c,
        "eff_now": e_now,
        "gap_to_0.8": y_gap,
        "RUL_mean_days": int(round(mean_days)),
        "RUL_95CI_low_day": q2p5,
        "RUL_95CI_high_day": q97p5
    })

# Present table
df_rul_east = pd.DataFrame(results).set_index("Panel").sort_index()
print("\n--- East area RUL (Monte Carlo, 95% CI, days) ---")
print(df_rul_east)

# Optional: bar chart of mean RUL with error bars (± half CI width)
plt.figure(figsize=(8,3.5))
x_idx = np.arange(len(df_rul_east))
means = df_rul_east["RUL_mean_days"].to_numpy()
low = df_rul_east["RUL_95CI_low_day"].to_numpy()
high = df_rul_east["RUL_95CI_high_day"].to_numpy()
err = np.vstack([means - low, high - means])
plt.errorbar(x_idx, means, yerr=err, fmt='o', capsize=4)
plt.xticks(x_idx, df_rul_east.index, rotation=45)
plt.ylabel('RUL (days)')
plt.title('East area: RUL (mean ± 95% CI) from Monte Carlo')
plt.tight_layout(); plt.show()


# -------------------------
# Quick Gamma fit on East (for comparison only)
# -------------------------
# Uses same steps as West: enforce non-increasing per SPU -> degradation -> daily increments -> MoM

# 1) Take only panel columns and ensure float
panel_cols_e = [c for c in east_interp.columns if c != day_col_e]
eff_east_fit = east_interp[panel_cols_e].astype(float)

# 2) Enforce non-increasing efficiency per SPU (noise guard, consistent with West)
eff_east_mono = eff_east_fit.copy()
for c in panel_cols_e:
    eff_east_mono[c] = np.minimum.accumulate(eff_east_mono[c].to_numpy())

# 3) Efficiency -> degradation, then daily increments
deg_east = 1.0 - eff_east_mono
ddeg_east = deg_east.diff().iloc[1:, :]  # drop first row (NaNs)

# 4) Pool strictly positive daily increments
x_e = ddeg_east.to_numpy().ravel()
x_e = x_e[np.isfinite(x_e)]
x_e = x_e[x_e > 0]

# 5) Method of Moments on East
if x_e.size == 0:
    print("East fit: no positive increments found (cannot estimate).")
    alp_e = np.nan; bet_e = np.nan; m_e = np.nan; v_e = np.nan
else:
    m_e = x_e.mean()
    v_e = x_e.var(ddof=0)
    if (v_e <= 0) or (not np.isfinite(v_e)) or (not np.isfinite(m_e)) or (m_e <= 0):
        alp_e = 1.0
        bet_e = m_e if (m_e > 0 and np.isfinite(m_e)) else 1.0
    else:
        alp_e = (m_e * m_e) / v_e
        bet_e = v_e / m_e

    print("---- Quick Gamma parameter estimation (East) ----")
    print(f"Number of increments considered: {x_e.size}")
    print(f"Mean increment m_e: {m_e:.8f}")
    print(f"Var  increment v_e: {v_e:.8f}")
    print(f"(East) MoM estimates: alpha_e = {alp_e:.4f}, beta_e = {bet_e:.6f}")

    # 6) Simple comparison to West
    m_w = alp * bet
    print("---- East vs West (increment-level) ----")
    if np.isfinite(m_w) and m_w > 0:
        print(f"Mean increment ratio (East/West): {m_e / m_w:.3f}")
    else:
        print("Cannot compute mean ratio: invalid West mean.")

    # Percent differences (guarding NaNs)
    def pct_diff(a, b):
        return np.nan if not (np.isfinite(a) and np.isfinite(b) and b != 0) else 100.0 * (a - b) / b

    print(f"alpha: East={alp_e:.4f}, West={alp:.4f}, Δ%={pct_diff(alp_e, alp):.1f}%")
    print(f"beta:  East={bet_e:.6f}, West={bet:.6f}, Δ%={pct_diff(bet_e, bet):.1f}%")

    # Quick caution if sample is small
    if x_e.size < 100:
        print("Note: East sample of daily increments is small; MoM estimates may be unstable.")
