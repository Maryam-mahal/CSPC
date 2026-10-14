import numpy as np

# TODO 1: Create an array called time_s containing the times 0, 2, 4, 6, 8 and 10 seconds using np.arange
# np.arange(start, stop, step) excludes the stop value, so we use 12 as the upper bound.
time_s = np.arange(0, 12, 2)
print("time_s:", time_s)

# TODO 2: Create an array called wavelength_nm containing 8 evenly spaced wavelengths from 400 nm to 700 nm using np.linspace
wavelength_nm = np.linspace(400, 700, 8)
print("wavelength_nm:", wavelength_nm)

# TODO 3: Create a 2x4 array called concentrations containing two rows [0.10, 0.20, 0.30, 0.40] and [0.15, 0.25, 0.35, 0.45]. Print its shape, dtype and size.
concentrations = np.array([
    [0.10, 0.20, 0.30, 0.40],
    [0.15, 0.25, 0.35, 0.45]
])

print("concentrations:\n", concentrations)
print("Shape:", concentrations.shape)
print("Data Type:", concentrations.dtype)
print("Size:", concentrations.size)