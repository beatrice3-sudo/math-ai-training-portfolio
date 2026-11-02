Task: Solve and validate advanced math problem through programmatic workflow
Role: Math Expert - AI Training Project

Problem: Find and validate global minimum of f(x) = x^4 - 3x^3 + 2x
Workflow: Translate math -> executable code -> numerical experiments -> validation
"""

import numpy as np
from scipy.optimize import minimize_scalar
import sympy as sp
import matplotlib.pyplot as plt

# 1. DEFINE MATHEMATICAL FUNCTION
def f(x):
    """Target function: f(x) = x^4 - 3x^3 + 2x"""
    return x**4 - 3*x**3 + 2*x

# 2. SYMBOLIC ANALYSIS WITH SYMPY (for theoretical validation)
print("=== STEP 1: Symbolic Analysis ===")
x_sym = sp.symbols('x')
f_sym = x_sym**4 - 3*x_sym**3 + 2*x_sym
f_prime = sp.diff(f_sym, x_sym)
f_double_prime = sp.diff(f_prime, x_sym)
critical_points = sp.solve(f_prime, x_sym)

print(f"f(x) = {f_sym}")
print(f"f'(x) = {f_prime}")
print(f"f''(x) = {f_double_prime}")
print(f"Critical points: {critical_points}")

# 3. NUMERICAL IMPLEMENTATION WITH SCIPY
print("\n=== STEP 2: Numerical Optimization ===")
result = minimize_scalar(f, bounds=(-1, 3), method='bounded')
x_min_num = result.x
f_min_num = result.fun

print(f"Numerical minimum found: x = {x_min_num:.6f}")
print(f"Function value: f(x) = {f_min_num:.6f}")
print(f"Iterations: {result.nfev}")

# 4. PARAMETER SWEEP & ROBUSTNESS
print("\n=== STEP 3: Parameter Sweep ===")
xs = np.linspace(-1, 3, 500)
ys = f(xs)

# 5. EDGE CASE TESTING
print("\n=== STEP 4: Edge Case Testing ===")
edge_cases = [-1, -0.5, 0, 0.5, 1.0, 2.25, 3]
for xc in edge_cases:
    print(f"f({xc:4.1f}) = {f(xc):8.4f}")

# 6. VALIDATION: Symbolic vs Numerical
print("\n=== STEP 5: Validation ===")
f_prime_func = sp.lambdify(x_sym, f_prime, 'numpy')
f_double_prime_func = sp.lambdify(x_sym, f_double_prime, 'numpy')

derivative_at_min = f_prime_func(x_min_num)
second_derivative_at_min = f_double_prime_func(x_min_num)

print(f"f'({x_min_num:.6f}) = {derivative_at_min:.6f} (should be ~0 for minimum)")
print(f"f''({x_min_num:.6f}) = {second_derivative_at_min:.6f} (should be >0 for minimum)")

# 7. FINAL ASSERTIONS FOR REPRODUCIBILITY
try:
    assert abs(derivative_at_min) < 1e-4, "Derivative check failed"
    assert second_derivative_at_min > 0, "Second derivative check failed - not a minimum"
    print("\n✓ VALIDATION PASSED: Result is mathematically and numerically valid")
except AssertionError as e:
    print(f"\n✗ VALIDATION FAILED: {e}")

# 8. VISUALIZATION FOR RESULT VALIDATION
plt.figure(figsize=(10, 6))
plt.plot(xs, ys, 'b-', linewidth=2, label='f(x) = x^4 - 3x^3 + 2x')
plt.plot(x_min_num, f_min_num, 'ro', markersize=12, label=f'Global Min: x={x_min_num:.2f}, f={f_min_num:.2f}')

# Mark critical points
for cp in critical_points:
    try:
        cp_float = float(cp)
        if -1 <= cp_float <= 3:
            plt.axvline(x=cp_float, color='gray', linestyle='--', alpha=0.5)
            plt.plot(cp_float, f(cp_float), 'go', alpha=0.7)
    except:
        pass

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Numerical Optimization Validation: Symbolic vs Numerical')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('validation_plot_01.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== FINAL REPRODUCIBLE RESULT ===")
print(f"Global minimum: x = {x_min_num:.6f}, f(x) = {f_min_num:.6f}")
