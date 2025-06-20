import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# === CONFIGURATION ===
# Paths, constants, and visualization setup for comparison between homogeneous and heterogeneous systems
homo_dir = "homo_circle_sweep"
hetero_dir = "hetero_circle_mean_sweep"
curvity_range = range(-50, 51)
highlight = [-50, -25, 0, 25, 50]
colors = ['blue', 'green', 'black', 'orange', 'red']
boxSize = 5
cx, cy = boxSize / 2, boxSize / 2
r_max = boxSize / 2
n_bins = 100
r0active = 0.05
particle_area = np.pi * r0active**2

print("\n[INFO] Starting full circular homo vs hetero comparison...\n")

# === FUNCTIONS ===

def load_all_radii(folder):
    """
    Load all particle radial distances from pickled trajectory files in a folder.
    Computes R = sqrt((x - cx)^2 + (y - cy)^2) for all particles.
    """
    radii_all = []
    for f in sorted(os.listdir(folder)):
        if f.startswith("file_") and f.endswith(".pkl"):
            with open(os.path.join(folder, f), "rb") as pf:
                data = pickle.load(pf)
            x = data[:, 0, :].flatten()
            y = data[:, 1, :].flatten()
            R = np.sqrt((x - cx)**2 + (y - cy)**2)
            radii_all.append(R)
    return np.concatenate(radii_all) if radii_all else np.array([])

def compute_density(r):
    """
    Compute the radial density profile, corrected by Jacobian (1/r) and normalized.
    """
    hist, edges = np.histogram(r, bins=n_bins, range=(0, r_max))
    centers = (edges[:-1] + edges[1:]) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        density = hist / centers
        density[np.isnan(density)] = 0
        density[np.isinf(density)] = 0
    density /= density.sum() if density.sum() > 0 else 1
    return centers, density

def compute_radial_stats(r, d):
    """
    Compute mean, standard deviation, and entropy of a given radial density distribution.
    These serve as order parameters for system behavior.
    """
    mean_r = np.sum(r * d)
    std_r = np.sqrt(np.sum((r - mean_r)**2 * d))
    ent = entropy(d[d > 0])
    return mean_r, std_r, ent

def process_system(base_dir, label):
    """
    Main loop to process either homo- or hetero- systems:
    - Reads position data
    - Computes radial density
    - Collects mean/std/entropy metrics
    """
    stats = []
    radial_profiles = {}
    for c in curvity_range:
        print(f"[{label.upper()}] Curvity {c}")
        if label == "homo":
            run_dir = os.path.join(base_dir, f"run_{c}")
            if not os.path.exists(run_dir): continue
            radii = load_all_radii(run_dir)
        else:
            base_path = os.path.join(base_dir, f"mean_{c}")
            if not os.path.exists(base_path): continue
            all_r = []
            for trial in os.listdir(base_path):
                trial_path = os.path.join(base_path, trial)
                if os.path.isdir(trial_path):
                    all_r.append(load_all_radii(trial_path))
            radii = np.concatenate(all_r) if all_r else np.array([])

        if len(radii) == 0:
            continue

        r, D = compute_density(radii)
        mean_r, std_r, ent = compute_radial_stats(r, D)
        stats.append((c, mean_r, std_r, ent))
        radial_profiles[c] = (r, D)
    return stats, radial_profiles

def compute_ff_from_files(base_dir, label):
    """
    Computes filling fraction (fraction of space filled by particles) as function of radius
    across all files per curvity. Includes mean and maximum.
    """
    curvs, mean_ffs, max_ffs = [], [], []
    for c in curvity_range:
        print(f"[FF TRUE] {label.upper()} curvity {c}")
        if label == "homo":
            run_dir = os.path.join(base_dir, f"run_{c}")
            if not os.path.exists(run_dir): continue
            file_dirs = [run_dir]
        else:
            base_path = os.path.join(base_dir, f"mean_{c}")
            if not os.path.exists(base_path): continue
            file_dirs = [os.path.join(base_path, trial) for trial in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, trial))]

        all_r = []
        for trial_dir in file_dirs:
            for f in sorted(os.listdir(trial_dir)):
                if f.startswith("file_") and f.endswith(".pkl"):
                    with open(os.path.join(trial_dir, f), "rb") as pf:
                        data = pickle.load(pf)
                    x = data[:, 0, :].flatten()
                    y = data[:, 1, :].flatten()
                    R = np.sqrt((x - cx)**2 + (y - cy)**2)
                    all_r.extend(R)

        if len(all_r) == 0:
            continue

        hist, edges = np.histogram(all_r, bins=n_bins, range=(0, r_max))
        bin_centers = (edges[:-1] + edges[1:]) / 2
        dr = edges[1] - edges[0]
        shell_areas = 2 * np.pi * bin_centers * dr

        with np.errstate(divide='ignore', invalid='ignore'):
            ff = (hist * particle_area) / shell_areas
            ff[np.isnan(ff)] = 0
            ff[np.isinf(ff)] = 0

        curvs.append(c)
        mean_ffs.append(np.mean(ff))
        max_ffs.append(np.max(ff))

    return curvs, mean_ffs, max_ffs

# === LOAD DATA ===
stats_homo, profiles_homo = process_system(homo_dir, "homo")
stats_hetero, profiles_hetero = process_system(hetero_dir, "hetero")
curv_h, mean_h, std_h, ent_h = zip(*sorted(stats_homo))
curv_het, mean_het, std_het, ent_het = zip(*sorted(stats_hetero))

# === FIGURE 0: Coverage Comparison from summary.pkl ===
print("\n=== [FIGURE 0] Coverage vs Curvity from summary.pkl ===")

homo_coverage, hetero_coverage = [], []

for c in curvity_range:
    # Homogeneous
    path_h = os.path.join(homo_dir, f"run_{c}", f"summary_{c}.pkl")
    if os.path.exists(path_h):
        with open(path_h, "rb") as f:
            s = pickle.load(f)
        if "mean_coverage" in s and "std_coverage" in s:
            homo_coverage.append((c, s["mean_coverage"], s["std_coverage"]))

    # Heterogeneous
    path_het = os.path.join(hetero_dir, f"mean_{c}", "summary.pkl")
    if os.path.exists(path_het):
        with open(path_het, "rb") as f:
            s = pickle.load(f)
        if "mean_coverage" in s and "std_coverage" in s:
            hetero_coverage.append((c, s["mean_coverage"], s["std_coverage"]))

# === Plotting ===
if homo_coverage and hetero_coverage:
    curv_h, cov_h, std_h = zip(*sorted(homo_coverage))
    curv_het, cov_het, std_het = zip(*sorted(hetero_coverage))

    plt.figure(figsize=(12, 6))
    plt.errorbar(curv_h, cov_h, yerr=std_h, fmt='o', capsize=3,
                 color='steelblue', ecolor='lightgray', label='homogeneous')
    plt.plot(curv_h, cov_h, color='steelblue')
    plt.errorbar(curv_het, cov_het, yerr=std_het, fmt='x', capsize=3,
                 color='orangered', ecolor='gray', label='heterogeneous')
    plt.plot(curv_het, cov_het, color='orangered')
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Figure 1: Coverage vs Curvity — Homo vs Hetero")
    plt.xlabel("Curvity")
    plt.ylabel("Total Coverage")
    plt.ylim(0.6, 1.01)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Coverage data missing for some curvities.")

# === FIGURE 1: Mean Radial Distance ===
plt.figure()
plt.plot(curv_h, mean_h, 'o-', label="homogeneous", color='steelblue')
plt.plot(curv_het, mean_het, 'x-', label="heterogeneous", color='orangered')
plt.title("Figure 1: Mean Radial Distance vs Curvity")
plt.xlabel("Curvity")
plt.ylabel("Mean R")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === FIGURE 2: Std Radius ===
plt.figure()
plt.plot(curv_h, std_h, 'o--', label="homogeneous", color='steelblue')
plt.plot(curv_het, std_het, 'x--', label="heterogeneous", color='orangered')
plt.title("Figure 2: Std of Radial Distance vs Curvity")
plt.xlabel("Curvity")
plt.ylabel("Std(R)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === FIGURE 3: Entropy ===
plt.figure()
plt.plot(curv_h, ent_h, 'o-', label="homogeneous", color='steelblue')
plt.plot(curv_het, ent_het, 'x-', label="heterogeneous", color='orangered')
plt.title("Figure 3: Radial Density Entropy vs Curvity")
plt.xlabel("Curvity")
plt.ylabel("Entropy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === FIGURE 4: Overlay of All P(R | curvity) with Zoom ===
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
for c, (r, D) in profiles_homo.items():
    axs[0].plot(r, D, color='gray', alpha=0.2)
for c, (r, D) in profiles_hetero.items():
    axs[0].plot(r, D, linestyle='--', color='gray', alpha=0.2)
for c, color in zip(highlight, colors):
    if c in profiles_homo:
        r, D = profiles_homo[c]
        axs[0].plot(r, D, color=color, linewidth=2.5, label=f"Homo {c}")
    if c in profiles_hetero:
        r, D = profiles_hetero[c]
        axs[0].plot(r, D, linestyle='--', color=color, linewidth=2.5, label=f"Hetero {c}")
axs[0].set_title("Figure 4a: Full View — P(R | curvity)")
axs[0].set_xlabel("Radial Distance R")
axs[0].set_ylabel("Normalized Density")
axs[0].grid(True)
axs[0].legend()

for c, (r, D) in profiles_homo.items():
    axs[1].plot(r, D, color='gray', alpha=0.2)
for c, (r, D) in profiles_hetero.items():
    axs[1].plot(r, D, linestyle='--', color='gray', alpha=0.2)
for c, color in zip(highlight, colors):
    if c in profiles_homo:
        r, D = profiles_homo[c]
        axs[1].plot(r, D, color=color, linewidth=2.5)
    if c in profiles_hetero:
        r, D = profiles_hetero[c]
        axs[1].plot(r, D, linestyle='--', color=color, linewidth=2.5)
axs[1].set_xlim(2.0, 2.5)
axs[1].set_title("Figure 4b: Zoom on R = 2.0–2.5")
axs[1].set_xlabel("Radial Distance R")
axs[1].grid(True)
plt.tight_layout()
plt.show()

# === FIGURE 5: Corrected Filling Fraction Comparison ===
curv_h_ff, mean_ff_h, max_ff_h = compute_ff_from_files(homo_dir, "homo")
curv_het_ff, mean_ff_het, max_ff_het = compute_ff_from_files(hetero_dir, "hetero")

fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
axs[0].plot(curv_h_ff, mean_ff_h, 'o-', color='steelblue', label="homogeneous")
axs[0].plot(curv_het_ff, mean_ff_het, 'x-', color='orangered', label="heterogeneous")
axs[0].set_title("Figure 5a: Mean Filling Fraction vs Curvity")
axs[0].set_xlabel("Curvity")
axs[0].set_ylabel("Mean Filling Fraction")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(curv_h_ff, max_ff_h, 'o--', color='navy', label="homogeneous")
axs[1].plot(curv_het_ff, max_ff_het, 'x--', color='darkred', label="heterogeneous")
axs[1].set_title("Figure 5b: Max Filling Fraction vs Curvity")
axs[1].set_xlabel("Curvity")
axs[1].set_ylabel("Max Filling Fraction")
axs[1].grid(True)
axs[1].legend()

plt.suptitle("Figure 5: Filling Fraction via Frame-Based Histogram — Homo vs Hetero")
plt.tight_layout()
plt.show()
