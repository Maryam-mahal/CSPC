import numpy as np
import matplotlib.pyplot as plt

# Parameter given in the lab sheet
LAMBDA = 0.3

# ==========================================
# TODO 1: Read decay_observed.csv into arrays t and observed
# ==========================================
t, observed = np.loadtxt('decay_observed.csv', delimiter=',', skiprows=1, unpack=True)

# ==========================================
# TODO 2: Set N0 to first observed value and build analytical model
# ==========================================
N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

# ==========================================
# TODO 3: Create 1x2 subplot with shared axes
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 4))

# Left plot: Observed data points
ax1.scatter(t, observed, color='blue', label='Observed')
ax1.set_title('Observed Data')
ax1.set_xlabel('Time')
ax1.set_ylabel('Count')

# Right plot: Analytical model curve
ax2.plot(t, analytical, color='red', label='Analytical')
ax2.set_title('Analytical Law')
ax2.set_xlabel('Time')

plt.tight_layout()

# ==========================================
# TODO 4: Save the figure
# ==========================================
plt.savefig('figure.png')
print("Saved figure.png successfully!")