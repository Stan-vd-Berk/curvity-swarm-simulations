import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

# Setup for rectangular phase space comparison between homogeneous and heterogeneous swarms
# === CONFIGURATION ===
homo_dir = "homocurvityTest_rectangular"
hetero_dir = "hetero_rectangular_mean_sweep"
curvity_range = range(-50, 51)
highlight = [-50, -25, 0, 25, 50]
colors = ['blue', 'green', 'black', 'orange', 'red']
boxSize = 5
cx, cy = boxSize / 2, boxSize / 2
r_max = boxSize * np.sqrt(2) / 2 # Maximum Euclidian distance, in this case corner radial distance.
n_bins = 100
r0active = 0.05
particle_area = np.pi * r0active**2

print("\n[INFO] Starting full rectangular radial analysis...\n")

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


# === FIGURE 1: Coverage vs Curvity ===
print("[INFO] Plotting Figure 1: Coverage vs Curvity")

homo_data, hetero_data = [], []
for c in curvity_range:
    homo_path = os.path.join(homo_dir, f"run_{c}", f"summary_{c}.pkl")
    if os.path.exists(homo_path):
        with open(homo_path, "rb") as f:
            s = pickle.load(f)
        if "mean_coverage" in s and "std_coverage" in s:
            homo_data.append((c, s["mean_coverage"], s["std_coverage"]))

    hetero_path = os.path.join(hetero_dir, f"mean_{c}", "summary.pkl")
    if os.path.exists(hetero_path):
        with open(hetero_path, "rb") as f:
            s = pickle.load(f)
        if "mean_coverage" in s and "std_coverage" in s:
            hetero_data.append((c, s["mean_coverage"], s["std_coverage"]))

if homo_data and hetero_data:
    hc, hm, hs = zip(*sorted(homo_data))
    hec, hem, hes = zip(*sorted(hetero_data))
    plt.figure(figsize=(12, 6))
    plt.errorbar(hc, hm, yerr=hs, fmt='o', capsize=2, color='steelblue', label='homogeneous', ecolor='lightgray')
    plt.plot(hc, hm, color='steelblue')
    plt.errorbar(hec, hem, yerr=hes, fmt='x', capsize=3, color='orangered', label='heterogeneous', ecolor='gray')
    plt.plot(hec, hem, color='orangered')
    plt.axvline(0, color='red', linestyle='--', linewidth=1)
    plt.title("Figure 1: Coverage vs Mean Curvity — Homo vs Hetero")
    plt.xlabel("Curvity")
    plt.ylabel("Total Coverage")
    plt.ylim(0.6, 1.01)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === LOAD DATA ===
stats_homo, profiles_homo = process_system(homo_dir, "homo")
stats_hetero, profiles_hetero = process_system(hetero_dir, "hetero")

curv_h, mean_h, std_h, ent_h = zip(*sorted(stats_homo))
curv_het, mean_het, std_het, ent_het = zip(*sorted(stats_hetero))

# === FIGURE 2: Mean Radial Distance ===
plt.figure()
plt.plot(curv_h, mean_h, 'o-', label="homogeneous", color='steelblue')
plt.plot(curv_het, mean_het, 'x-', label="heterogeneous", color='orangered')
plt.title("Figure 2: Mean Radial Distance vs Curvity")
plt.xlabel("Curvity")
plt.ylabel("Mean R")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === FIGURE 3: Std Radius ===
plt.figure()
plt.plot(curv_h, std_h, 'o--', label="homogeneous", color='steelblue')
plt.plot(curv_het, std_het, 'x--', label="heterogeneous", color='orangered')
plt.title("Figure 3: Std of Radial Distance vs Curvity")
plt.xlabel("Curvity")
plt.ylabel("Std(R)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === FIGURE 4: Entropy ===
plt.figure()
plt.plot(curv_h, ent_h, 'o-', label="homogeneous", color='steelblue')
plt.plot(curv_het, ent_het, 'x-', label="heterogeneous", color='orangered')
plt.title("Figure 4: Radial Density Entropy vs Curvity")
plt.xlabel("Curvity")
plt.ylabel("Entropy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === FIGURE 4: Radial Distribution Overlays ===
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

# Full view: subplot (a)
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

axs[0].set_title("Figure 2a(a): Full View — P(R | κ)")
axs[0].set_xlabel("Radial Distance R")
axs[0].set_ylabel("Normalized Density")
axs[0].grid(True)
axs[0].legend()

# Zoom view: subplot (b)
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

axs[1].set_xlim(2.4, 3.5)
axs[1].set_title("Figure 2a(b): Zoom on R = 2.4–3.5")
axs[1].set_xlabel("Radial Distance R")
axs[1].grid(True)

plt.tight_layout()
plt.show()

# === FIGURE 5: Filling Fraction Comparison ===
curv_h_ff, mean_ff_h, max_ff_h = compute_ff_from_files(homo_dir, "homo")
curv_het_ff, mean_ff_het, max_ff_het = compute_ff_from_files(hetero_dir, "hetero")

fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
axs[0].plot(curv_h_ff, mean_ff_h, 'o-', color='steelblue', label="homogeneous")
axs[0].plot(curv_het_ff, mean_ff_het, 'x-', color='orangered', label="heterogeneous")
axs[0].set_title("Figure 2b(a): Mean Filling Fraction vs Curvity")
axs[0].set_xlabel("Curvity")
axs[0].set_ylabel("Mean Filling Fraction")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(curv_h_ff, max_ff_h, 'o--', color='navy', label="homogeneous")
axs[1].plot(curv_het_ff, max_ff_het, 'x--', color='darkred', label="heterogeneous")
axs[1].set_title("Figure 2b(b): Max Filling Fraction vs Curvity")
axs[1].set_xlabel("Curvity")
axs[1].set_ylabel("Max Filling Fraction")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()
