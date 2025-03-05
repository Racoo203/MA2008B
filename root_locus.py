import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sympy as sp
from sympy import symbols, diff, solve, re, im, I
import control

def angle_of_departure(pole, poles, zeros):
    """Calculate the angle of departure for a given complex pole."""
    angle_sum_poles = np.sum([np.angle(pole - p, deg=True) for p in poles if p != pole])
    angle_sum_zeros = np.sum([np.angle(pole - z, deg=True) for z in zeros])
    return (180 + angle_sum_poles - angle_sum_zeros) % 360

def angle_of_arrival(zero, poles, zeros):
    """Calculate the angle of arrival for a given complex zero."""
    angle_sum_poles = np.sum([np.angle(zero - p, deg=True) for p in poles])
    angle_sum_zeros = np.sum([np.angle(zero - z, deg=True) for z in zeros if z != zero])
    return (180 - angle_sum_poles + angle_sum_zeros) % 360

# Define the transfer function
plt.close('all')


num_factors = [np.array([1, 0.2])]
den_factors = [np.array([1, 0, 0]), np.array([1, 3.6])]

num = np.array([1])
den = np.array([1])
for factor in num_factors:
    num = np.convolve(num, factor)
for factor in den_factors:
    den = np.convolve(den, factor)

# Display Transfer Function
print("-------------------------")
print("1. Transfer Function G(s):")
print(f"Numerator: {num}")
print(f"Denominator: {den}")

print("-------------------------")
print("2. System Poles and Zeros:")

G_control = control.TransferFunction(num, den)
poles = control.poles(G_control)
zeros = control.zeros(G_control)

# Display Poles and Zeros
print("\nSystem Poles:")
for p in poles:
    print(f"{p.real:.4f} + {p.imag:.4f}j")

print("\nSystem Zeros:")
print("None" if len(zeros) == 0 else "\n".join(f"{z.real:.4f} + {z.imag:.4f}j" for z in zeros))

# Compute Centroid and Asymptote Angles
num_poles, num_zeros = len(poles), len(zeros)
centroid = (np.sum(poles) - np.sum(zeros)) / (num_poles - num_zeros)
asymptote_angles = (180 * (2 * np.arange(num_poles - num_zeros) + 1)) / (num_poles - num_zeros)


#########################
# ADD Locus on the real axis
# The root locus exists on the real axis to the left of an odd number of poles and zeros of the loop gain, G(s)H(s), that are on the real axis. The real pole and zero locations (i.e., those that are on the real axis) are highlighted on the diagram by pink diamonds, along with the portion of the locus that exists on the real axis that is shown by a pink line.

# Find real axis segments of the root locus
real_poles_zeros = sorted([p.real for p in poles if np.isreal(p)] + [z.real for z in zeros if np.isreal(z)])
real_axis_locus = []

for i in range(len(real_poles_zeros) - 1):
    segment_start = real_poles_zeros[i]
    segment_end = real_poles_zeros[i + 1]
    test_point = (segment_start + segment_end) / 2
    num_poles_zeros_left = sum(1 for pz in real_poles_zeros if pz < test_point)
    if num_poles_zeros_left % 2 == 1:
        real_axis_locus.append((segment_start, segment_end))

# Print valid locus on the real axis intervals
print("\nReal Axis Locus Intervals:")
for segment in real_axis_locus:
    print(f"Interval: ({segment[0]:.4f}, {segment[1]:.4f})")

# Plot real axis segments
# for segment in real_axis_locus:
#     plt.plot([segment[0], segment[1]], [0, 0], 'm-', linewidth=2, label="Real Axis Locus" if segment == real_axis_locus[0] else "")

# Highlight real poles and zeros
# real_poles = [p.real for p in poles if np.isreal(p)]
# real_zeros = [z.real for z in zeros if np.isreal(z)]
# plt.plot(real_poles, [0] * len(real_poles), 'md', markersize=8, label="Real Poles")
# plt.plot(real_zeros, [0] * len(real_zeros), 'mo', markersize=8, label="Real Zeros")

print("-------------------------")
print("3. Centroid and Asymptotes:")
print(f"\nCentroid: {centroid.real:.4f} + {centroid.imag:.4f}j")
print("\nAsymptote Angles (degrees):")
print(asymptote_angles)

# Solve for Breakaway Points
s = symbols('s')
num_sym = sum(coef * s**(len(num)-i-1) for i, coef in enumerate(num))
den_sym = sum(coef * s**(len(den)-i-1) for i, coef in enumerate(den))
G_s = num_sym / den_sym
dK_ds = diff(-1 / G_s, s)
break_points = solve(dK_ds, s)

print("-------------------------")
print("4. Breakaway Points:")
for point in break_points:
    try:
        complex_value = complex(point.evalf())
        print(f"{complex_value.real:.4f} + {complex_value.imag:.4f}j")
    except:
        print(f"{point}")

# Solve for Imaginary Axis Crossings
print("-------------------------")
print("5. Axis Crossings:")
w = symbols('w', real=True)

print("5.1 Real Axis")
# Real axis crossings occur when the imaginary part of the characteristic equation is zero
real_axis_crossings = solve(G_s, s)

# Filter out only real solutions
real_axis_crossings = [sol.evalf() for sol in real_axis_crossings if sol.is_real]

for sol in real_axis_crossings:
    print(f"{sol:.4f}")

print("5.2 Imaginary Axis")

s_jw = I * w  # I is the imaginary unit in SymPy

# Evaluate G(jω)
G_jw = G_s.subs(s, s_jw)

# Characteristic K(jω) = -1 / G(jω)
K_jw = -1 / G_jw

# Extract real and imaginary parts
K_real = re(K_jw)
K_imag = im(K_jw)

# Set up equations K_real = 0 and K_imag = 0
eq_real = K_real
eq_imag = K_imag

# Solve for ω
omega_real_solution = solve(eq_real, w)
omega_imag_solution = solve(eq_imag, w)

print("\nSolutions for ω when the real part of K(jω) = 0:")
for sol in omega_real_solution:
    # Safely handle solutions
    try:
        if sol.is_real and float(sol) > 0:  # Only consider positive real solutions
            print(f"{float(sol):.4f}")
    except:
        print(f"  {sol} (complex or symbolic solution)")

print("\nSolutions for ω when the imaginary part of K(jω) = 0:")
for sol in omega_imag_solution:
    # Safely handle solutions
    try:
        if sol.is_real and float(sol) > 0:  # Only consider positive real solutions
            print(f"{float(sol):.4f}")
    except:
        print(f"  {sol} (complex or symbolic solution)")

# Combine omega_real_solution and omega_imag_solution
omega_solutions = list(set(omega_real_solution).union(set(omega_imag_solution)))

# Evaluate K for each ω and find the one that maximizes K
max_k = -np.inf
max_w = None

for omega in omega_solutions:
    try:
        if omega.is_real and float(omega) > 0:  # Only consider positive real solutions
            K_value = K_jw.subs(w, omega).evalf()
            if K_value > max_k:
                max_k = K_value
                max_w = omega
    except:
        continue

if max_w is not None:
    print(f"\nMaximum K is {max_k:.4f} at ω = {float(max_w):.4f}")
else:
    print("\nNo valid ω found that maximizes K")

# Compute Angles of Departure and Arrival
print("-------------------------")
print("6. Angles of Departure and Arrival:")
for pole in [p for p in poles if np.iscomplex(p)]:
    print(f"Angle of Departure at pole {pole.real:.4f} + {pole.imag:.4f}j: {angle_of_departure(pole, poles, zeros):.2f}°")

for zero in [z for z in zeros if np.iscomplex(z)]:
    print(f"Angle of Arrival at zero {zero.real:.4f} + {zero.imag:.4f}j: {angle_of_arrival(zero, poles, zeros):.2f}°")

if len([p for p in poles if np.iscomplex(p)]) == 0:
    print("No complex poles found to calculate angle of departure")

if len([z for z in zeros if np.iscomplex(z)]) == 0:
    print("No complex zeros found to calculate angle of arrival")

# Plot Root Locus
rlist, klist = control.root_locus_plot(G_control, plot=True)
asymptote_length = 10
for angle in asymptote_angles:
    angle_rad = np.radians(angle)
    x_end, y_end = centroid.real + asymptote_length * np.cos(angle_rad), centroid.imag + asymptote_length * np.sin(angle_rad)
    plt.plot([centroid.real, x_end], [centroid.imag, y_end], 'r--', label=f"Asymptote {angle:.1f}°")

plt.plot(centroid.real, centroid.imag, 'go', markersize=8, label="Centroid")
plt.legend()
plt.title("Root Locus with Asymptotes")
plt.show()