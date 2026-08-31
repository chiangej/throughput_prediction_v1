import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Load predictions
# ============================================================

data_A = np.load(
    "test_results_A.npz"
)

data_AB = np.load(
    "test_results_AB.npz"
)


target = data_A["target"]

pred_A = data_A["prediction"]

pred_AB = data_AB["prediction"]


# Safety check
assert len(target) == len(pred_A)
assert len(target) == len(pred_AB)


# ============================================================
# Time axis
# ============================================================

dt = 0.1

time_axis = (
    np.arange(len(target))
    * dt
)


# ============================================================
# Plot
# ============================================================

plt.figure(
    figsize=(12, 6)
)


# measured
plt.plot(
    time_axis,
    target,
    linewidth=2.3,
    label="Measured values"
)


# Phi_AB
plt.plot(
    time_axis,
    pred_AB,
    linestyle="--",
    linewidth=2.3,
    label=r"$\Phi_{AB}$"
)


# Phi_A
plt.plot(
    time_axis,
    pred_A,
    linestyle="--",
    linewidth=2.3,
    label=r"$\Phi_A$"
)


plt.xlabel(
    "Time [s]",
    fontsize=14
)

plt.ylabel(
    "Throughput [Mbps]",
    fontsize=14
)


plt.xlim(
    time_axis[0],
    time_axis[-1]
)


plt.grid(
    True,
    alpha=0.25
)


plt.legend(
    fontsize=13,
    loc="lower right"
)


plt.tight_layout()


plt.savefig(
    "prediction_comparison_A_AB.png",
    dpi=300
)


plt.show()
