"""
    1CM290 Maintenance Optimziation and Engineering (Lecturer: J. Lee)
    Assignment: Data Cahllenges 2025
    Challenge: Maintenance of highway bridge
    This is a template for the assignment.
    You may fill the parts <YOUR CODE HERE> or add new parts as you need.
    You may change as you need.
    Please add concise and comprehensive comments.
"""

import numpy as np
import pandas as pd
from math import factorial
from matplotlib import pyplot as plt

# Read Data
df = pd.read_csv('data_highway/inspection_log.csv')
print(df)

# Parameters
c_i = 10 # Inspection cost (thousands euros)
c_minor = 0 # Repair cost for minor damage  (thousands euros)
c_moderate = 1 # Repair cost for moderate damage (thousands euros)
c_major = 10 # Repair cost for major damage (thousands euros)
penalty = 500 # Penalty cost (thousands euros)
penalty_threshold = 20 # Threshold of total severity score for penalty

# Pre-process data
# YOUR CODE HERE

# Task (a)
def likelihood():
    # YOUR CODE HERE
    return


def mle_lam():
    # YOUR CODE HERE
    return


print(f'(a) MLE of lambda: {mle_lam():.4f} damages/day')

# Task (b)
prob_severity = 0 # YOUR CODE HERE
print(f'(b) P_minor = {prob_severity[0]:.4f}, P_moderate = {prob_severity[1]:.4f}, P_major = {prob_severity[2]:.4f}')

# Task (c)
# YOUR CODE HERE
print(f"(c) Monthly maintenance cost = ")

# Task (d)
# YOUR CODE HERE
print(f"(d) Optimal inspection interval = ")

