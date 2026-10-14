"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
 - differentiate once  -> velocity
 - differentiate twice -> acceleration (should be ~ constant -g, but noisy)
 - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# TODO 1: read freefall.csv into arrays t and y
# (hint: np.loadtxt with a comma delimiter, skipping the header)
data = np.loadtxt('freefall.csv', delimiter=',', skiprows=1)
t = data[:, 0]
y = data[:, 1]


# TODO 2: compute velocity v = derivative of y w.r.t. t  (np.gradient)
#         and acceleration a = derivative of v w.r.t. t   (np.gradient)
#         Print the mean acceleration. Is it close to -9.81? Is it noisy?
v = np.gradient(y, t)
a = np.gradient(v, t)

print(f"Mean acceleration: {np.mean(a):.2f} m/s^2")


# TODO 3: integrate a back up to recover velocity and position
# (hint: cumulative_trapezoid(a, t, initial=0) + v[0], then integrate that for y)
v_rec = cumulative_trapezoid(a, t, initial=0) + v[0]
y_rec = cumulative_trapezoid(v_rec, t, initial=0) + y[0]
diff = np.abs(y - y_rec)
max_diff = np.max(diff)
print(f"Largest difference in position: {max_diff:.2f} m")

print(f"Mean acceleration: {a.mean():.2f} m/s^2")
print(f"Standard deviation of acceleration: {a.std():.2f} m/s^2")

# TODO 4: make a figure with 3 stacked panels: position, velocity, acceleration
#         vs time. Mark the true -9.81 line on the acceleration panel.
#         Save it as motion.png
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

# Верхняя панель: Положение (Position)
axes[0].plot(t, y, 'b.', label='Measured y(t)')
axes[0].plot(t, y_rec, 'r--', label='Recovered y(t)')
axes[0].set_ylabel('Position (m)')
axes[0].legend()
axes[0].grid(True)

# Средняя панель: Скорость (Velocity)
axes[1].plot(t, v, 'b.', label='Differentiated v(t)')
axes[1].plot(t, v_rec, 'r--', label='Recovered v(t)')
axes[1].set_ylabel('Velocity (m/s)')
axes[1].legend()
axes[1].grid(True)

# Нижняя панель: Ускорение (Acceleration)
axes[2].plot(t, a, 'b-', alpha=0.6, label='Differentiated a(t)')
axes[2].axhline(-9.81, color='g', linestyle='--', label='True g (-9.81 m/s²)')
axes[2].set_xlabel('Time (s)')
axes[2].set_ylabel('Acceleration (m/s²)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('motion.png')
plt.show()