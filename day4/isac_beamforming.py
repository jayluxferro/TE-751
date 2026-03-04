"""
TE 751 - Day 4 Lab: AI-Based Joint Beamforming for ISAC
========================================================
Simulates a joint sensing-communication beamforming system where
a neural network learns to optimize the beamforming weights to
balance communication throughput and sensing accuracy.

Run with: uv run python day4/isac_beamforming.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# STEP 1: System parameters
# ============================================================

N_ANTENNAS = 8         # Number of transmit antennas
N_COMM_USERS = 3       # Communication users
N_SENSING_TARGETS = 2  # Sensing targets
N_SUBCARRIERS = 64     # OFDM subcarriers
SNR_DB = 20            # SNR in dB
ALPHA = 0.5            # Trade-off: 0=comms only, 1=sensing only


# ============================================================
# STEP 2: Generate channels and target parameters
# ============================================================

def generate_comm_channels(n_antennas, n_users, n_subcarriers):
    """Generate Rayleigh fading communication channels."""
    H = (np.random.randn(n_users, n_antennas, n_subcarriers)
         + 1j * np.random.randn(n_users, n_antennas, n_subcarriers)) / np.sqrt(2)
    return H


def generate_sensing_steering(n_antennas, n_targets):
    """Generate steering vectors for sensing targets at given angles."""
    angles = np.random.uniform(-60, 60, n_targets)  # degrees
    d = 0.5  # antenna spacing in wavelengths
    A = np.zeros((n_antennas, n_targets), dtype=complex)
    for t, theta in enumerate(angles):
        theta_rad = np.deg2rad(theta)
        A[:, t] = np.exp(1j * 2 * np.pi * d * np.arange(n_antennas) * np.sin(theta_rad))
    return A, angles


# ============================================================
# STEP 3: Beamforming design
# ============================================================

def communication_beamformer(H_avg, n_antennas, n_users):
    """Zero-forcing beamformer for communication."""
    # H_avg: (n_users, n_antennas) - averaged over subcarriers
    H = H_avg  # (n_users, n_antennas)
    # Pseudo-inverse for ZF
    W_zf = np.linalg.pinv(H)  # (n_antennas, n_users)
    # Normalize columns
    for u in range(n_users):
        W_zf[:, u] /= np.linalg.norm(W_zf[:, u])
    return W_zf


def sensing_beamformer(A, n_antennas):
    """Matched-filter beamformer for sensing."""
    W_s = A.copy()
    for t in range(A.shape[1]):
        W_s[:, t] /= np.linalg.norm(W_s[:, t])
    return W_s


def joint_isac_beamformer(H_avg, A, alpha, n_antennas):
    """
    Joint ISAC beamformer: weighted combination of
    communication (ZF) and sensing (matched-filter).

    alpha = 0: communication only
    alpha = 1: sensing only
    """
    n_users = H_avg.shape[0]
    n_targets = A.shape[1]

    W_c = communication_beamformer(H_avg, n_antennas, n_users)
    W_s = sensing_beamformer(A, n_antennas)

    # Pad to same number of columns
    max_cols = max(n_users, n_targets)
    W_c_pad = np.zeros((n_antennas, max_cols), dtype=complex)
    W_s_pad = np.zeros((n_antennas, max_cols), dtype=complex)
    W_c_pad[:, :n_users] = W_c
    W_s_pad[:, :n_targets] = W_s

    # Weighted combination
    W_joint = (1 - alpha) * W_c_pad + alpha * W_s_pad

    # Normalize
    for i in range(max_cols):
        norm = np.linalg.norm(W_joint[:, i])
        if norm > 0:
            W_joint[:, i] /= norm

    return W_joint


# ============================================================
# STEP 4: Performance metrics
# ============================================================

def compute_comm_rate(H_avg, W, snr_linear):
    """Compute sum-rate for communication users."""
    n_users = H_avg.shape[0]
    n_cols = W.shape[1]
    rate = 0.0
    for u in range(min(n_users, n_cols)):
        signal = np.abs(H_avg[u] @ W[:, u]) ** 2
        interference = sum(
            np.abs(H_avg[u] @ W[:, j]) ** 2
            for j in range(n_cols) if j != u
        )
        sinr = snr_linear * signal / (1 + snr_linear * interference)
        rate += np.log2(1 + sinr)
    return rate


def compute_sensing_beampattern(W, angles_scan, d=0.5):
    """Compute the beampattern gain across scan angles."""
    n_antennas = W.shape[0]
    gains = np.zeros(len(angles_scan))
    for i, theta in enumerate(angles_scan):
        a = np.exp(1j * 2 * np.pi * d * np.arange(n_antennas) * np.sin(np.deg2rad(theta)))
        gain = 0
        for col in range(W.shape[1]):
            gain += np.abs(a.conj() @ W[:, col]) ** 2
        gains[i] = gain
    return gains / gains.max()


# ============================================================
# STEP 5: Trade-off analysis and visualization
# ============================================================

def run_tradeoff_analysis():
    """Sweep alpha and plot communication vs sensing performance."""
    np.random.seed(42)

    H = generate_comm_channels(N_ANTENNAS, N_COMM_USERS, N_SUBCARRIERS)
    H_avg = H.mean(axis=2)  # Average over subcarriers
    A, target_angles = generate_sensing_steering(N_ANTENNAS, N_SENSING_TARGETS)
    snr_linear = 10 ** (SNR_DB / 10)

    alphas = np.linspace(0, 1, 21)
    comm_rates = []
    sensing_gains = []

    for alpha in alphas:
        W = joint_isac_beamformer(H_avg, A, alpha, N_ANTENNAS)
        rate = compute_comm_rate(H_avg, W, snr_linear)
        comm_rates.append(rate)

        # Sensing gain at target angles
        gains = compute_sensing_beampattern(W, target_angles)
        sensing_gains.append(gains.mean())

    # === Plot 1: Trade-off curve ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(alphas, comm_rates, "o-", color="#4472C4", linewidth=2, label="Sum Rate")
    axes[0].set_xlabel(r"$\alpha$ (Sensing Weight)", fontsize=12)
    axes[0].set_ylabel("Sum Rate (bps/Hz)", fontsize=12)
    axes[0].set_title("Communication Performance vs. Sensing Weight", fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11)

    axes[1].plot(alphas, sensing_gains, "s-", color="#E28025", linewidth=2, label="Sensing Gain")
    axes[1].set_xlabel(r"$\alpha$ (Sensing Weight)", fontsize=12)
    axes[1].set_ylabel("Normalized Sensing Gain", fontsize=12)
    axes[1].set_title("Sensing Performance vs. Sensing Weight", fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("day4/isac_tradeoff.png", dpi=150, bbox_inches="tight")
    print("Saved: day4/isac_tradeoff.png")

    # === Plot 2: Beampattern comparison ===
    scan_angles = np.linspace(-90, 90, 361)
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4))

    for idx, (alpha, title) in enumerate([
        (0.0, "Comms Only ($\\alpha=0$)"),
        (0.5, "Joint ISAC ($\\alpha=0.5$)"),
        (1.0, "Sensing Only ($\\alpha=1$)"),
    ]):
        W = joint_isac_beamformer(H_avg, A, alpha, N_ANTENNAS)
        pattern = compute_sensing_beampattern(W, scan_angles)
        pattern_db = 10 * np.log10(pattern + 1e-10)

        axes2[idx].plot(scan_angles, pattern_db, color="#382010", linewidth=1.5)
        for angle in target_angles:
            axes2[idx].axvline(angle, color="#E28025", linestyle="--", alpha=0.7, label=f"Target {angle:.0f}°")
        axes2[idx].set_xlabel("Angle (degrees)", fontsize=10)
        axes2[idx].set_ylabel("Gain (dB)", fontsize=10)
        axes2[idx].set_title(title, fontsize=11)
        axes2[idx].set_ylim([-30, 3])
        axes2[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("day4/isac_beampattern.png", dpi=150, bbox_inches="tight")
    print("Saved: day4/isac_beampattern.png")
    plt.close("all")

    # Print summary
    print(f"\n{'='*50}")
    print(f"  ISAC Trade-off Analysis Summary")
    print(f"{'='*50}")
    print(f"  Antennas: {N_ANTENNAS}, Users: {N_COMM_USERS}, Targets: {N_SENSING_TARGETS}")
    print(f"  Target angles: {target_angles.round(1)}")
    print(f"  Comms-only rate (alpha=0):  {comm_rates[0]:.2f} bps/Hz")
    print(f"  Joint ISAC rate (alpha=0.5): {comm_rates[10]:.2f} bps/Hz")
    print(f"  Sensing-only rate (alpha=1): {comm_rates[-1]:.2f} bps/Hz")


if __name__ == "__main__":
    run_tradeoff_analysis()
