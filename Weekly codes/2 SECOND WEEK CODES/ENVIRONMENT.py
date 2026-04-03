import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import random
from matplotlib.patches import Patch

#parameters
# Total physical size = 500μm x 500μm.
# 1 px = 5μm such that a cell takes up 1 px (100*100 grid)
GRID_SIZE = 100
TIME_STEP_DT = 1.0     # 1 hour per time step
SIMULATION_STEPS = 50

INITIAL_CELLS = 50
INIT_PCD_PLUS_FRACTION = 0.50

INIT_NUTRIENT_LEVEL = 100.0
INIT_ANTIFUNGAL_LEVEL = 50.0
DIFFUSION_NUTRIENT = 0.2       # D_n (Must be <= 0.25 for stability)
DIFFUSION_ANTIFUNGAL = 0.1    # D_f

# agents (the two genotpes are subclasses)
class CellAgent:
    def __init__(self, x, y, biomass=1.0):
        self.x = int(x)
        self.y = int(y)
        self.alive = True
        self.biomass = float(biomass)

class PCDPlus(CellAgent):
    pass

class PCDMinus(CellAgent):
    pass

# model
class PetriDishModel:
    def __init__(self):
        s = GRID_SIZE
        self.size = s
        self.nutrient_layer = np.full((s, s), INIT_NUTRIENT_LEVEL, dtype=float)
        self.antifungal_layer = np.full((s, s), INIT_ANTIFUNGAL_LEVEL, dtype=float)
        self.occupied_layer = np.zeros((s, s), dtype=bool)
        self.agents = []
        self.laplacian_kernel = np.array([[0,  1, 0],
                                          [1, -4, 1],
                                          [0,  1, 0]])

        num_plus = int(round(INITIAL_CELLS * INIT_PCD_PLUS_FRACTION))

        # placing cells randomly while ensuring no overlap (1 cell per pixel)
        for i in range(INITIAL_CELLS):
            placed = False
            attempts = 0
            while not placed and attempts < 10000:
                rx = np.random.randint(0, s)
                ry = np.random.randint(0, s)
                if self.is_space_clear(rx, ry):
                    self.occupied_layer[ry, rx] = True
                    if i < num_plus:
                        self.agents.append(PCDPlus(rx, ry))
                    else:
                        self.agents.append(PCDMinus(rx, ry))
                    placed = True
                attempts += 1

    def is_space_clear(self, x, y):
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return not self.occupied_layer[y, x]
    
    def step_environment(self):
        laplacian_n = convolve(self.nutrient_layer, self.laplacian_kernel, mode='reflect')
        self.nutrient_layer += (DIFFUSION_NUTRIENT * laplacian_n) * TIME_STEP_DT
        self.nutrient_layer = np.clip(self.nutrient_layer, 0, None)

        laplacian_f = convolve(self.antifungal_layer, self.laplacian_kernel, mode='reflect')
        self.antifungal_layer += (DIFFUSION_ANTIFUNGAL * laplacian_f) * TIME_STEP_DT
        self.antifungal_layer = np.clip(self.antifungal_layer, 0, None)

# plotting 

def plot_initial_state():
    print("Initializing 100x100 Environment (1px = 5μm)...")
    model = PetriDishModel()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Subplot 1: Nutrient Heatmap
    im1 = axes[0].imshow(model.nutrient_layer, cmap='viridis', origin='lower', vmin=0, vmax=INIT_NUTRIENT_LEVEL, aspect='equal')
    axes[0].set_title('Nutrients)')
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Subplot 2: ANTIFUNGAL Heatmap
    im2 = axes[1].imshow(model.antifungal_layer, cmap='Blues', origin='lower', vmin=0, vmax=INIT_ANTIFUNGAL_LEVEL, aspect='equal')
    axes[1].set_title('Antifungal agent)')
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Subplot 3: Cells (Pixel-based visualization)
    axes[2].set_title('Petri dish')
    axes[2].set_xlim(-0.5, model.size - 0.5)
    axes[2].set_ylim(-0.5, model.size - 0.5)
    axes[2].set_aspect('equal')

    plus_x = [a.x for a in model.agents if isinstance(a, PCDPlus)]
    plus_y = [a.y for a in model.agents if isinstance(a, PCDPlus)]
    minus_x = [a.x for a in model.agents if isinstance(a, PCDMinus)]
    minus_y = [a.y for a in model.agents if isinstance(a, PCDMinus)]

    if plus_x:
        axes[2].scatter(plus_x, plus_y, label='PCD+', color='blue', s=16, linewidths=0.5)
    if minus_x:
        axes[2].scatter(minus_x, minus_y, label='PCD-', color='red', s=16, linewidths=0.5)

    axes[2].legend(loc='upper right', fontsize='small')

    plt.suptitle(f"Simulation grid of {GRID_SIZE}x{GRID_SIZE} (Initial)")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_initial_state()
