import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# --- CONFIGURATION ---
SEARCH_PATHS = [
    "cmake-build-debug/output/data",
    "cmake-build-release/output/data",
    "output/data",
    "./data"
]


def find_data_folder():
    for path in SEARCH_PATHS:
        if os.path.exists(path):
            if glob.glob(os.path.join(path, "*.txt")):
                return path
    return None


def read_and_plot(filepath, output_folder):
    filename = os.path.basename(filepath)
    print(f"Processing: {filename}...")

    try:
        with open(filepath, 'r') as f:
            header = f.readline().split()
            if not header: return
            Nx = int(header[0])
            Ny = int(header[1])

            data = []
            for line in f:
                data.extend([float(x) for x in line.split()])
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    data = np.array(data)

    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        print("⚠️  WARNING: Data contains NaN or Inf.")
        return

    try:
        T_flat = data[0::2]
        psi_flat = data[1::2]
        T = T_flat.reshape((Ny, Nx))
        psi = psi_flat.reshape((Ny, Nx))
    except ValueError:
        return

    # --- PLOTTING FIX START ---
    fig, ax = plt.subplots(figsize=(10, 4))

    # Physical Dimensions (Matches C++ Lx=4.0, Ly=1.0)
    Lx, Ly = 4.0, 1.0

    # 1. Plot Temperature (Heat map)
    # We define the extent explicitly so it sits between 0 and 4.0
    im = ax.imshow(T, cmap='inferno', origin='lower', extent=[0, Lx, 0, Ly], aspect='auto', vmin=0.0, vmax=1.0)

    # 2. Overlay Streamlines (Fluid motion)
    # FIX: Create a grid that matches physical coordinates (0 to 4.0), NOT indices (0 to 100)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    # Calculate velocity
    u, v = np.gradient(psi)
    v = -v

    # Plot arrows using the correct X, Y grid
    ax.streamplot(X, Y, v, u, color='cyan', linewidth=0.6, density=1.0, arrowsize=0.7)

    # --- PLOTTING FIX END ---

    plt.colorbar(im, label='Temperature (Normalized)')

    step_num = filename.replace('step_', '').replace('.txt', '')
    plt.title(f"Solar Convection - Step {step_num}")
    plt.xlabel("x (Mm)")
    plt.ylabel("z (Height Mm)")

    save_path = os.path.join(output_folder, filename.replace('.txt', '.png'))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# --- MAIN ---
data_folder = find_data_folder()

if not data_folder:
    print("❌ No data files found.")
else:
    print(f"✅ Found data in: {data_folder}")
    images_folder = os.path.join(os.path.dirname(data_folder), "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    files = sorted(glob.glob(os.path.join(data_folder, "*.txt")), key=os.path.getmtime)
    for f in files:
        read_and_plot(f, images_folder)
    print("Done.")