import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import random
from matplotlib.patches import Patch

# --- AGENT CLASSES ---

class CellAgent:
    """Base class for all cells."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.alive = True

    def step(self, local_ANTIFUNGAL):
        # To be overridden by subclasses
        pass

class PCDPlus(CellAgent):
    """PCD+ Genotype: 70% chance of apoptosis if ANTIFUNGAL >= 30"""
    def step(self, local_ANTIFUNGAL):
        if self.alive and local_ANTIFUNGAL >= 30.0:
            if random.random() < 0.70:
                self.alive = False

class PCDMinus(CellAgent):
    """PCD- Genotype: 10% chance of necrosis if ANTIFUNGAL >= 50"""
    def step(self, local_ANTIFUNGAL):
        if self.alive and local_ANTIFUNGAL >= 50.0:
            if random.random() < 0.10:
                self.alive = False

# --- MODEL CLASS ---

class PetriDishModel:
    def __init__(self, size_um=500, initial_cells=500):
        self.size = size_um
        
        self.nutrient_layer = np.full((self.size, self.size), 100.0)
        
        # Note: Starting at 50.0 means cells will experience death triggers immediately!
        self.ANTIFUNGAL_layer = np.full((self.size, self.size), 50.0)
        
        # PDE Parameters
        self.D_n = 0.2       # Nutrient diffusion coeff
        self.lambda_n = 0.1  # Nutrient consumption rate by LIVING cells
        
        self.D_f = 0.1       # ANTIFUNGAL diffusion coeff
        self.lambda_f = 0.05 # ANTIFUNGAL absorption rate by DEAD cells
        
        self.laplacian_kernel = np.array([[0,  1, 0],
                                          [1, -4, 1],
                                          [0,  1, 0]])
        
        # Initialize a 50/50 mix of the two genotypes
        self.agents = []
        for i in range(initial_cells):
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            if i % 2 == 0:
                self.agents.append(PCDPlus(x, y))
            else:
                self.agents.append(PCDMinus(x, y))

    def step_environment(self, dt=1.0):
        # 1. Ask cells to check their local environment and update status
        for agent in self.agents:
            if agent.alive:
                local_f = self.ANTIFUNGAL_layer[agent.y, agent.x]
                agent.step(local_f)
                
        # 2. Map where the living and dead cells are currently located
        living_density = np.zeros((self.size, self.size))
        dead_density = np.zeros((self.size, self.size))
        
        for agent in self.agents:
            if agent.alive:
                living_density[agent.y, agent.x] += 1
            else:
                dead_density[agent.y, agent.x] += 1
        
        # 3. Nutrients (Only LIVING cells act as sinks)
        laplacian_n = convolve(self.nutrient_layer, self.laplacian_kernel, mode='reflect')
        diffusion_n = self.D_n * laplacian_n
        consumption_n = self.lambda_n * living_density * self.nutrient_layer
        
        self.nutrient_layer += (diffusion_n - consumption_n) * dt
        self.nutrient_layer = np.clip(self.nutrient_layer, 0, None)

        # 4. ANTIFUNGAL (Only DEAD cells act as sinks)
        laplacian_f = convolve(self.ANTIFUNGAL_layer, self.laplacian_kernel, mode='reflect')
        diffusion_f = self.D_f * laplacian_f
        consumption_f = self.lambda_f * dead_density * self.ANTIFUNGAL_layer
        
        self.ANTIFUNGAL_layer += (diffusion_f - consumption_f) * dt
        self.ANTIFUNGAL_layer = np.clip(self.ANTIFUNGAL_layer, 0, None)

    def get_rgb_cell_grid(self):
        """Generates an RGB image to differentiate cell types."""
        # Create a white background
        img = np.ones((self.size, self.size, 3))
        
        for agent in self.agents:
            if agent.alive:
                if isinstance(agent, PCDPlus):
                    img[agent.y, agent.x] = [0, 0.8, 0] # Green for PCD+
                elif isinstance(agent, PCDMinus):
                    img[agent.y, agent.x] = [0, 0, 0.8] # Blue for PCD-
            else:
                img[agent.y, agent.x] = [0.8, 0, 0] # Red for Dead cells
                
        return img

    def visualize(self, step_num):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        extent = [0, self.size, 0, self.size]

        im1 = axes[0].imshow(self.nutrient_layer, cmap='Greens', extent=extent, origin='lower', vmin=0, vmax=100)
        axes[0].set_title('Nutrient Layer')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(self.ANTIFUNGAL_layer, cmap='Blues', extent=extent, origin='lower', vmin=0, vmax=50)
        axes[1].set_title('ANTIFUNGAL Layer')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Cell Visualization
        rgb_cells = self.get_rgb_cell_grid()
        axes[2].imshow(rgb_cells, extent=extent, origin='lower')
        axes[2].set_title(f'Cells (Step {step_num})')
        
        # Custom Legend for the Cells
        legend_elements = [Patch(facecolor=[0, 0.8, 0], label='PCD+ (Alive)'),
                           Patch(facecolor=[0, 0, 0.8], label='PCD- (Alive)'),
                           Patch(facecolor=[0.8, 0, 0], label='Dead')]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.suptitle(f"Simulation at Time Step {step_num}")
        plt.tight_layout()
        plt.show()

# --- Run the Model ---
if __name__ == "__main__":
    model = PetriDishModel(size_um=500, initial_cells=500)
    
    # Show Initial State
    model.visualize(step_num=0)
    
    # Run the simulation forward in time
    for i in range(1, 51):
        model.step_environment(dt=1.0)
        
    # Show State after 50 steps
    model.visualize(step_num=50)