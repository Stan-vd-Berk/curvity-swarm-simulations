import os
import pickle
import numpy as np
import pandas as pd
import ecoSystem as es
import greens as gr
import timePropagation as tp

# === CONFIGURATION ===
# Homogeneous swarm setup in circular confinement; each run has a single curvity value
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
v0 = 20
kT = 1
root_dir = "homo_circle_sweep"
os.makedirs(root_dir, exist_ok=True)

# === UTILS ===
def compute_circular_coverage(df, boxSize, cell_size=50):
    """
    Compute how much of the circular domain has been visited by particles.
    Grid-based area coverage metric inside circular boundary.
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

def load_run_data(file_path, N):
    """
    Load a single simulation chunk into a flat DataFrame.
    Extracts x/y positions from pickled numpy arrays.
    """
    with open(file_path, "rb") as pf:
        data = pickle.load(pf)
    df = {
        'frame': np.tile(np.arange(data.shape[0]), N),
        'particle': np.repeat(np.arange(N), data.shape[0]),
        'x': data[:, 0, :].flatten(),
        'y': data[:, 1, :].flatten(),
    }
    return pd.DataFrame(df)

def load_all_chunks(run_folder, r):
    """
    Concatenate all 10 simulation chunks (000–009) into one DataFrame per repeat.
    Returns None if any chunk is missing.
    """
    frames = []
    for t in range(10):  # 000 to 009
        file_path = os.path.join(run_folder, f"file_{r}_{str(t).zfill(3)}_of_010.pkl")
        if not os.path.exists(file_path):
            print(f"Missing {file_path}")
            return None
        df = load_run_data(file_path, N)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def generate_initial_conditions():
    """
    Generate random particle positions uniformly inside a circle and
    assign random orientations.
    """
    cx, cy = boxSize / 2, boxSize / 2
    radius = boxSize / 2
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 1, N)) * radius
    x = cx + radii * np.cos(angles)
    y = cy + radii * np.sin(angles)
    orientations = np.random.uniform(0, 2 * np.pi, N)
    nx = np.cos(orientations)
    ny = np.sin(orientations)
    return x, y, nx, ny

# === FULL SIMULATION: HOMOGENEOUS CIRCULAR PHASE SPACE ===
# Sweep through curvity values from -50 to 50 and simulate each one

print("\n Running full coverage simulations for curvity -50 to 50")
for curvity in range(-50, 51):
    run_folder = os.path.join(root_dir, f"run_{curvity}")
    os.makedirs(run_folder, exist_ok=True)

    if os.path.exists(os.path.join(run_folder, f"summary_{curvity}.pkl")):
        print(f" Curvity {curvity} already completed.")
        continue

    print(f" Curvity = {curvity}")
    coverages = []

    for r in range(repeats):
        # Initialize particles
        x, y, nx, ny = generate_initial_conditions()

        # Setup and run ecoSystem simulation
        eco = es.ecoSystem(N=N, params=params)
        eco.initializeEcosystem(boxSize, r0active*np.ones(N), wS0*np.ones(N), v0*np.ones(N), 0)
        eco.sp[eco.wAC, :] = curvity # Homogeneous curvity for all particles
        eco.sp[eco.xC, :] = x
        eco.sp[eco.yC, :] = y
        eco.sp[eco.nxC, :] = nx
        eco.sp[eco.nyC, :] = ny
        # Propagate the system and integrate time series

        file_base = os.path.join(run_folder, f"file_{r}")
        prop = tp.timePropagation(dt, eco, file_base)
        prop.timeProp5RungePeriodic(T, tSave, tArray, greenFunc, kT, boxSize, particlesToSave=range(N))

        df_all = load_all_chunks(run_folder, r)
        if df_all is not None:
            cov = compute_circular_coverage(df_all, boxSize)
            coverages.append(cov)

    if len(coverages) == repeats:

        # Analyze coverage from simulation output and save to pickle file
        avg_cov = np.mean(coverages)
        std_cov = np.std(coverages)
        summary = {
            "curvity": curvity,
            "mean_coverage": float(avg_cov),
            "std_coverage": float(std_cov),
            "all_coverages": coverages
        }
        with open(os.path.join(run_folder, f"summary_{curvity}.pkl"), "wb") as f:
            pickle.dump(summary, f)
        print(f"Curvity {curvity}, Mean: {avg_cov:.4f}, Std: {std_cov:.4f}")

print("\n All curvities processed successfully.")
