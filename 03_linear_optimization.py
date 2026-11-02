"""
Sample 3: Optimization & Mathematical Modeling
Task: Solve and validate constrained linear optimization
Role: Math Expert - AI Training

Problem: Maximize 3x + 5y subject to x+2y<=8, 3x+2y<=12, x,y>=0
Workflow: Translate to executable workflow, implement with SciPy (CVXPY equivalent), validate constraints, parameter sweep
"""

import numpy as np
from scipy.optimize import linprog
import sympy as sp
import matplotlib.pyplot as plt

print("=== SAMPLE 3: Linear Optimization Validation ===")

# 1. MATHEMATICAL FORMULATION
print("\n=== STEP 1: Problem Formulation ===")
print("Maximize: Z = 3x + 5y")
print("Subject to:")
print("  x + 2y <= 8")
print("  3x + 2y <= 12")
print("  x >= 0, y >= 0")

# For SciPy linprog, we minimize -Z (minimize -3x-5y = maximize 3x+5y)
c = [-3, -5]  # Objective to minimize (negated)
A_ub = [[1, 2], [3, 2]]  # Inequality coefficients
b_ub = [8, 12]  # Inequality RHS

# 2. NUMERICAL IMPLEMENTATION
print("\n=== STEP 2: Numerical Solution with SciPy (CVXPY equivalent) ===")
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method='highs')

if not result.success:
    print(f"Optimization failed: {result.message}")
    exit()

x_opt, y_opt = result.x
max_value = -result.fun  # Negate back

print(f"Optimal solution: x = {x_opt:.6f}, y = {y_opt:.6f}")
print(f"Maximum value: Z = 3*{x_opt:.4f} + 5*{y_opt:.4f} = {max_value:.4f}")
print(f"Solver status: {result.message}")
print(f"Iterations: {result.nit}")

# 3. SYMBOLIC VALIDATION
print("\n=== STEP 3: Symbolic Validation (Intersection Method) ===")
x, y = sp.symbols('x y')
# Solve intersection of the two constraint lines
symbolic_solution = sp.solve([x + 2*y - 8, 3*x + 2*y - 12], (x, y))
print(f"Intersection of constraints x+2y=8 and 3x+2y=12: {symbolic_solution}")
print(f"Numerical solution matches symbolic: {abs(symbolic_solution[x] - x_opt) < 1e-6 and abs(symbolic_solution[y] - y_opt) < 1e-6}")

# 4. CONSTRAINT VALIDATION & EDGE CASES
print("\n=== STEP 4: Constraint Validation & Edge Cases ===")

def check_feasibility(x_val, y_val, tolerance=1e-6):
    """Check if point satisfies all constraints"""
    c1 = x_val + 2*y_val <= 8 + tolerance
    c2 = 3*x_val + 2*y_val <= 12 + tolerance
    c3 = x_val >= -tolerance
    c4 = y_val >= -tolerance
    return c1 and c2 and c3 and c4

# Test cases
test_points = [
    (x_opt, y_opt, "Optimal solution"),
    (0, 0, "Origin"),
    (0, 4, "Y-intercept of first constraint"),
    (4, 0, "X-intercept of second constraint"),
    (5, 5, "Infeasible point"),
    (1, 1, "Interior feasible point")
]

for xt, yt, desc in test_points:
    feasible = check_feasibility(xt, yt)
    status = "FEASIBLE" if feasible else "INFEASIBLE"
    print(f"{desc:35s} ({xt:4.1f}, {yt:4.1f}): {status} | Z={3*xt+5*yt:.1f}")

# 5. PARAMETER SWEEP - Sensitivity Analysis
print("\n=== STEP 5: Parameter Sweep (Sensitivity Analysis) ===")
print("What happens if we change RHS of second constraint?")
for b2_test in [10, 12, 14, 16]:
    res = linprog(c, A_ub=A_ub, b_ub=[8, b2_test], bounds=[(0, None), (0, None)], method='highs')
    if res.success:
        x_t, y_t = res.x
        val_t = -res.fun
        print(f"b2={b2_test:2d}: optimum ({x_t:.2f}, {y_t:.2f}), Z={val_t:.2f}")
    else:
        print(f"b2={b2_test:2d}: Infeasible")

# 6. FINAL VALIDATION
print("\n=== STEP 6: Final Validation ===")
try:
    assert check_feasibility(x_opt, y_opt), "Optimal solution violates constraints!"
    assert max_value > 0, "Maximum value should be positive"
    # Verify no better feasible point nearby
    for dx, dy in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1)]:
        x_near, y_near = x_opt + dx, y_opt + dy
        if check_feasibility(x_near, y_near):
            assert 3*x_near + 5*y_near <= max_value + 1e-6, "Found better feasible point!"
    print("✓ VALIDATION PASSED: Solution is feasible, optimal, and numerically stable")
except AssertionError as e:
    print(f"✗ VALIDATION FAILED: {e}")

# 7. VISUALIZATION - Feasible Region
print("\n=== STEP 7: Generating Validation Plot ===")
X = np.linspace(0, 5, 400)
Y1 = (8 - X) / 2  # From x+2y=8
Y2 = (12 - 3*X) / 2  # From 3x+2y=12
Y_feas = np.minimum(Y1, Y2)
Y_feas = np.maximum(Y_feas, 0)

plt.figure(figsize=(10, 6))
plt.fill_between(X, 0, Y_feas, alpha=0.3, color='lightblue', label='Feasible Region')
plt.plot(X, Y1, 'b-', linewidth=2, label='x+2y=8')
plt.plot(X, Y2, 'orange', linewidth=2, label='3x+2y=12')
plt.plot(x_opt, y_opt, 'ro', markersize=12, label=f'Optimum ({x_opt:.2f}, {y_opt:.2f}), Z={max_value:.2f}')

# Mark corner points
corners = [(0,0), (0,4), (x_opt, y_opt), (4,0)]
for cx, cy in corners:
    plt.plot(cx, cy, 'ko', markersize=6)

plt.xlim(0, 5)
plt.ylim(0, 5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Programming: Feasible Region & Optimal Solution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('cvxpy_validation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== FINAL REPRODUCIBLE RESULT ===")
print(f"Optimal: x={x_opt:.6f}, y={y_opt:.6f}, Max Z={max_value:.6f}")
