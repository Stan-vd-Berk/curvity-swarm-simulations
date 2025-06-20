import numpy as np
import os
import pickle
import pandas as pd
from scipy.stats import truncnorm
import ecoSystem as es
import greens as gr
import timePropagation as tp

# === PARAMETERS ===
# Simulation setup: heterogeneous swarm in circular box with variable curvity means
params = 8
dt = 1E-5
T = 1E5
tSave = 10000
tArray = 10
r0active = 0.05
boxSize = 5
wS0 = 1E2
greenFunc = gr.greens.grInteractingHeteroPopCircularWall
repeats = 10
N = 100
kT = 1
v0 = 20

def generate_gaussian_with_exact_mean(N, target_mean, lower=-50, upper=50, std=None):
    """
    Generate an array of N curvities sampled from a truncated Gaussian with:
    - a precise mean (after linear adjustment)
    - bounds to stay within curvity range
    Ensures heterogeneity around a desired mean for each swarm configuration.
    """
    if std is None:
        std = max(0.01, min(abs(lower - target_mean), abs(upper - target_mean)))
    a, b = (lower - target_mean) / std, (upper - target_mean) / std
    values = truncnorm.rvs(a, b, loc=target_mean, scale=std, size=N)
    adjustment = target_mean - np.mean(values)
    values += adjustment
    values = np.clip(values, lower, upper)
    np.random.shuffle(values)
    return values

def compute_circular_coverage(df, boxSize, cell_size=50):
    """
    Compute area coverage by checking which grid cells inside a circular boundary
    have been visited by at least one particle.
    This is the core metric used to evaluate CCP performance.
    """
    step = boxSize / cell_size
    x_bins = np.arange(0, boxSize, step)
    y_bins = np.arange(0, boxSize, step)
    nx, ny = len(x_bins), len(y_bins)
    xv, yv = np.meshgrid(x_bins + step/2, y_bins + step/2, indexing='ij')
    valid_mask = (xv - boxSize/2)**2 + (yv - boxSize/2)**2 <= (boxSize/2)**2
    total_valid = np.sum(valid_mask)
    visited = np.zeros((nx, ny), dtype=bool)
    for x, y in zip(df['x'], df['y']):
        ix = int(np.clip(x // step, 0, nx - 1))
        iy = int(np.clip(y // step, 0, ny - 1))
        visited[ix, iy] = True
    return np.sum(visited & valid_mask) / total_valid

def load_generation_data(folder_path):
    """
    Load particle trajectories from pickled simulation files and
    convert to flat DataFrame for spatial analysis.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".pkl") and f.startswith("file_")]
    all_frames = []
    for f in sorted(files):
        with open(os.path.join(folder_path, f), 'rb') as pf:
            data = pickle.load(pf)
        df = {
            'frame': np.tile(np.arange(data.shape[0]), N),
            'particle': np.repeat(np.arange(N), data.shape[0]),
            'x': data[:, 0, :].flatten(),
            'y': data[:, 1, :].flatten()
        }
        all_frames.append(pd.DataFrame(df))
    return pd.concat(all_frames, ignore_index=True)

# For each target mean curvity from -50 to 50, run "repeats" stochastic trials
output_dir = "hetero_circle_mean_sweep"
os.makedirs(output_dir, exist_ok=True)

for target_mean in range(-50, 51): # curvity range
    mean_dir = os.path.join(output_dir, f"mean_{target_mean}")
    os.makedirs(mean_dir, exist_ok=True)

    summary_path = os.path.join(mean_dir, "summary.pkl")
    if os.path.exists(summary_path):
         # Skip if results already exist in order to allow the program to halt intermediately.
        print(f"[Mean {target_mean} already completed.")
        continue


    coverages = []
    realized_means = []

    for run_id in range(repeats):
        # Generate heterogeneous curvities centered around target mean
        curvities = generate_gaussian_with_exact_mean(N, target_mean)
        realized_means.append(float(np.mean(curvities)))

        run_dir = os.path.join(mean_dir, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)

        # Initialize random positions and orientations
        x, y = np.random.uniform(0, boxSize, N), np.random.uniform(0, boxSize, N)
        orientations = np.random.uniform(0, 2*np.pi, N)
        nx = np.cos(orientations)
        ny = np.sin(orientations)

        # Setup and run ecosystem simulation
        eco = es.ecoSystem(N=N, params=params)
        eco.initializeEcosystem(boxSize, r0active * np.ones(N), wS0 * np.ones(N), v0 * np.ones(N), 0)
        eco.sp[eco.wAC, :] = curvities
        eco.sp[eco.xC, :] = x
        eco.sp[eco.yC, :] = y
        eco.sp[eco.nxC, :] = nx
        eco.sp[eco.nyC, :] = ny

        # Propagate the system and integrate time series
        file_base = os.path.join(run_dir, f"file_{run_id:03d}.pkl")
        prop = tp.timePropagation(dt, eco, file_base)
        prop.timeProp5RungePeriodic(T, tSave, tArray, greenFunc, kT, boxSize, particlesToSave=range(N))

        df = load_generation_data(run_dir)
        cov = compute_circular_coverage(df, boxSize)
        coverages.append(cov)

    # Analyze coverage from simulation output and save to pickle file
    avg_cov = np.mean(coverages)
    std_cov = np.std(coverages)
    mean_realized = float(np.mean(realized_means))

    with open(os.path.join(mean_dir, "summary.pkl"), "wb") as f:
        pickle.dump({
            "target_mean": target_mean,
            "realized_mean": mean_realized,
            "all_realized_means": realized_means,
            "mean_coverage": float(avg_cov),
            "std_coverage": float(std_cov),
            "all_coverages": coverages
        }, f)

    print(f"Mean {target_mean},  Coverage: {avg_cov:.4f} ± {std_cov:.4f}, Realized mean: {mean_realized:.4f}")
