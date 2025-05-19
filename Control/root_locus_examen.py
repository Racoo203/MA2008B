import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, solve, re, im, I
import control

import warnings
warnings.filterwarnings("ignore")

def root_locus_analysis(num_factors, den_factors):
    # Define the transfer function
    plt.close('all')

    num = np.array([1])
    den = np.array([1])

    for factor in num_factors:
        num = np.convolve(num, factor)
    for factor in den_factors:
        den = np.convolve(den, factor)

    print("-------------------------")
    print("Transfer Function G(s):")
    print(f"Numerator: {num}")
    print(f"Denominator: {den}")

    G_control = control.tf(num, den)

    poles = control.poles(G_control)
    zeros = control.zeros(G_control)

    num_poles, num_zeros = len(poles), len(zeros)
    centroid = (np.sum(poles) - np.sum(zeros)) / (num_poles - num_zeros)
    # print(centroid)
    asymptote_angles = (180 * (2 * np.arange(num_poles - num_zeros) + 1)) / (num_poles - num_zeros)

    print("Axis Crossings:")
    w = symbols('w', real=True)
    s = symbols('s')

    num_sym = sum(coef * s**(len(num)-i-1) for i, coef in enumerate(num))
    den_sym = sum(coef * s**(len(den)-i-1) for i, coef in enumerate(den))
    G_s = num_sym / den_sym

    s_jw = I * w

    G_jw = G_s.subs(s, s_jw)

    K_jw = -1 / G_jw

    K_real = re(K_jw)
    K_imag = im(K_jw)

    eq_real = K_real
    eq_imag = K_imag

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

    # Plot Root Locus
    rlist, klist = control.root_locus_plot(G_control, plot=True)
    asymptote_length = 10
    for angle in asymptote_angles:
        angle_rad = np.radians(angle)
        x_end, y_end = centroid.real + asymptote_length * np.cos(angle_rad), centroid.imag + asymptote_length * np.sin(angle_rad)
        plt.plot([centroid.real, x_end], [centroid.imag, y_end], 'r--', label=f"Asymptote {angle:.1f}°")

    plt.plot(centroid.real, centroid.imag, 'go', markersize=12, label="Centroid")  # Increased marker size for centroid

    # plt.plot(poles.real, poles.imag, 'x', markersize=12, label="Poles")  # Increased marker size for poles
    # plt.plot(zeros.real, zeros.imag, 'o', markersize=12, label="Zeros")  # Increased marker size for zeros

    plt.legend()
    plt.title("Root Locus with Asymptotes")
    plt.show()


num_factors = [np.array([1, 9])]
den_factors = [np.array([1, 0]), np.array([1, 4, 11])]

root_locus_analysis(num_factors, den_factors)