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

# Task (a)
# Data preprocessing
# YOUR CODE HERE

# Parameter estimation
alp = 0
bet = 0
# YOUR CODE HERE

print(f"(a) MoM, alpha = {alp:.2f}, beta = {bet:.4f}")

# Task (d)
# RUL estimation
RUL = []
# YOUR CODE HERE
print(f"(b) RUL of East area: {RUL}")

# Task (e)
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

