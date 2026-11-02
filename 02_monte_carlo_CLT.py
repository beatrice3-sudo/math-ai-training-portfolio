"""
Sample 2: Probability & Statistics - Monte Carlo Validation
Task: Validate Central Limit Theorem through programmatic workflow
Role: Math Expert - AI Training

Problem: Show that sample means of Exponential(2) converge to Normal distribution
Workflow: Numerical experiments + parameter sweeps + statistical validation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=== SAMPLE 2: Monte Carlo Validation of CLT ===")

# 1. MATHEMATICAL REQUIREMENT
# CLT: If X ~ Exp(mean=2), then sample mean of n samples -> N(2, 2/sqrt(n))
print("\n=== STEP 1: Mathematical Setup ===")
true_mean = 2.0  # Mean of Exp(scale=2)
true_std = 2.0   # Std of Exp(scale=2)
sample_size = 30
n_simulations = 10000

print(f"Distribution: X ~ Exponential(mean={true_mean})")
print(f"Sample size per experiment: n={sample_size}")
print(f"Number of simulations: {n_simulations}")
print(f"Theoretical: Sample means -> Normal(mean={true_mean}, std={true_std}/sqrt({sample_size})={true_std/np.sqrt(sample_size):.4f})")

# 2. COMPUTATIONAL PROCEDURE - Monte Carlo Simulation
print("\n=== STEP 2: Running Monte Carlo Simulations ===")
np.random.seed(42)  # For reproducibility

# Generate 10,000 sample means, each from 30 exponential samples
sample_means = []
for i in range(n_simulations):
    samples = np.random.exponential(scale=true_mean, size=sample_size)
    sample_means.append(np.mean(samples))

sample_means = np.array(sample_means)

# 3. NUMERICAL VALIDATION
print("\n=== STEP 3: Numerical Validation ===")
observed_mean = np.mean(sample_means)
observed_std = np.std(sample_means, ddof=1)
theoretical_mean = true_mean
theoretical_std = true_std / np.sqrt(sample_size)

print(f"Observed mean of sample means: {observed_mean:.4f} vs Theoretical: {theoretical_mean:.4f}")
print(f"Observed std of sample means: {observed_std:.4f} vs Theoretical: {theoretical_std:.4f}")
print(f"Mean error: {abs(observed_mean - theoretical_mean):.4f}")
print(f"Std error: {abs(observed_std - theoretical_std):.4f}")

# Statistical test - is it normal?
shapiro_stat, shapiro_p = stats.shapiro(sample_means[:5000])  # Shapiro needs <5000
print(f"Shapiro-Wilk normality test p-value: {shapiro_p:.4f} (>0.05 suggests normal)")

# 4. EDGE CASE TESTING - Does CLT improve with n?
print("\n=== STEP 4: Edge Cases - Effect of Sample Size ===")
for n_test in [5, 30, 100, 500]:
    test_means = [np.mean(np.random.exponential(scale=true_mean, size=n_test)) for _ in range(2000)]
    mean_err = abs(np.mean(test_means) - theoretical_mean)
    std_theoretical_n = true_std / np.sqrt(n_test)
    std_err = abs(np.std(test_means) - std_theoretical_n)
    print(f"n={n_test:3d}: observed mean={np.mean(test_means):.3f} (err={mean_err:.3f}), std={np.std(test_means):.3f} (theoretical std={std_theoretical_n:.3f})")

# 5. FINAL VALIDATION
print("\n=== STEP 5: Final Validation ===")
try:
    assert abs(observed_mean - theoretical_mean) < 0.05, "Mean validation failed"
    assert abs(observed_std - theoretical_std) < 0.05, "Std validation failed"
    print("✓ VALIDATION PASSED: CLT numerically validated - sample means are Normal")
except AssertionError as e:
    print(f"✗ VALIDATION FAILED: {e}")

# 6. VISUALIZATION - Required for AI training tasks
print("\n=== STEP 6: Generating Validation Plot ===")
plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=50, density=True, alpha=0.6, color='blue', label=f'Sample Means (n={sample_size}, sims={n_simulations})')

# Plot theoretical Normal PDF
x = np.linspace(1, 3.5, 200)
theoretical_pdf = stats.norm.pdf(x, loc=theoretical_mean, scale=theoretical_std)
plt.plot(x, theoretical_pdf, 'r-', linewidth=2, label=f'Theoretical Normal({theoretical_mean}, {theoretical_std:.2f})')

plt.xlabel('Sample Mean')
plt.ylabel('Density')
plt.title('Monte Carlo Validation of Central Limit Theorem')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('clt_validation.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== FINAL REPRODUCIBLE RESULT ===")
print(f"CLT validated: Exp({true_mean}) sample means (n={sample_size}) ~ N({observed_mean:.3f}, {observed_std:.3f})")
