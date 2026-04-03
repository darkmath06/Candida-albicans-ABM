import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import random
from matplotlib.patches import Patch

# ==========================================
# --- GLOBAL EXPERIMENTAL PARAMETERS ---
# ==========================================

# Grid and Time setup
GRID_SIZE_UM = 500
INITIAL_CELLS = 500
SIMULATION_STEPS = 50
TIME_STEP_DT = 1.0

# Initial Population Setup
INIT_PCD_PLUS_FRACTION = 0.9   # 0.5 = 50% PCD+, 50% PCD-. (e.g., 0.9 = 90% PCD+)

# Initial Environment Concentrations
INIT_NUTRIENT_LEVEL = 50.0
INIT_ANTIFUNGAL_LEVEL = 50.0

# Physics & Diffusion
DIFFUSION_NUTRIENT = 0.2       # D_n (Must be <= 0.25)
DIFFUSION_ANTIFUNGAL = 0.1    # D_f

# Biological Growth Parameters (Maltose model)
CONSUMPTION_RATE = 0.5         # How fast a cell can intake available nutrients
MAINTENANCE_COEFF = 0.015      # m: Nutrient needed per step just to survive
YIELD_TRUE = 0.39              # Y_true: Efficiency of converting nutrient to biomass
DIVISION_BIOMASS = 2.0         # Biomass threshold to trigger reproduction
STARVATION_BIOMASS = 0.1       # Biomass level where cell dies of starvation

# Dead Cell Sponge Limits
MAX_ANTIFUNGAL_BINDING = 15.0 # Absolute max amount a dead cell can absorb
BINDING_RATE_ANTIFUNGAL = 2.0 # How fast a dead cell absorbs the drug

# Genotype Rules (PCD+)
PCD_PLUS_DEATH_THRESHOLD = 30.0 
PCD_PLUS_DEATH_PROB = 0.70      

# Genotype Rules (PCD-)
PCD_MINUS_DEATH_THRESHOLD = 50.0 
PCD_MINUS_DEATH_PROB = 0.10      

# ==========================================
# --- AGENT CLASSES ---
# ==========================================

class CellAgent:
    """Base class for all cells."""
    def __init__(self, x, y, biomass=1.0):
        self.x = x
        self.y = y
        self.alive = True
        self.bound_ANTIFUNGAL = 0.0 
        self.biomass = biomass

    def step_ANTIFUNGAL(self, local_ANTIFUNGAL):
        pass

class PCDPlus(CellAgent):
    def step_ANTIFUNGAL(self, local_ANTIFUNGAL):
        if self.alive and local_ANTIFUNGAL >= PCD_PLUS_DEATH_THRESHOLD:
            if random.random() < PCD_PLUS_DEATH_PROB:
                self.alive = False

class PCDMinus(CellAgent):
    def step_ANTIFUNGAL(self, local_ANTIFUNGAL):
        if self.alive and local_ANTIFUNGAL >= PCD_MINUS_DEATH_THRESHOLD:
            if random.random() < PCD_MINUS_DEATH_PROB:
                self.alive = False

# ==========================================
# --- MODEL CLASS ---
# ==========================================

class PetriDishModel:
    def __init__(self):
        self.size = GRID_SIZE_UM
        
        self.nutrient_layer = np.full((self.size, self.size), INIT_NUTRIENT_LEVEL)
        self.ANTIFUNGAL_layer = np.full((self.size, self.size), INIT_ANTIFUNGAL_LEVEL)
        
        self.laplacian_kernel = np.array([[0,  1, 0],
                                          [1, -4, 1],
                                          [0,  1, 0]])
        
        self.agents = []
        
        # Calculate exact counts based on the global fraction
        num_pcd_plus = int(INITIAL_CELLS * INIT_PCD_PLUS_FRACTION)
        
        for i in range(INITIAL_CELLS):
            x = np.random.randint(0, self.size)
            y = np.random.randint(0, self.size)
            
            # First batch becomes PCD+, the rest become PCD-
            if i < num_pcd_plus:
                self.agents.append(PCDPlus(x, y))
            else:
                self.agents.append(PCDMinus(x, y))

    def step_environment(self):
        new_offspring = [] 
        
        # 1. Ask cells to execute biological behaviors
        for agent in self.agents:
            # --- LIVING CELL LOGIC ---
            if agent.alive:
                # A. Check ANTIFUNGAL Triggers
                local_f = self.ANTIFUNGAL_layer[agent.y, agent.x]
                agent.step_ANTIFUNGAL(local_f)
                
                # B. Eat, Grow, and Reproduce (if it survived the drug check)
                if agent.alive:
                    local_n = self.nutrient_layer[agent.y, agent.x]
                    
                    # Intake what it can
                    intake = min(CONSUMPTION_RATE * agent.biomass * TIME_STEP_DT, local_n)
                    self.nutrient_layer[agent.y, agent.x] -= intake
                    
                    # Calculate growth economics
                    maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                    net_nutrient = intake - maintenance_cost
                    
                    if net_nutrient > 0:
                        agent.biomass += net_nutrient * YIELD_TRUE
                    else:
                        agent.biomass += net_nutrient 
                        
                    # Check vital signs and division
                    if agent.biomass <= STARVATION_BIOMASS:
                        agent.alive = False # Died of starvation
                        
                    elif agent.biomass >= DIVISION_BIOMASS:
                        agent.biomass /= 2.0 # Split mass in half
                        
                        # Spatial offset
                        new_x = np.clip(agent.x + random.choice([-1, 0, 1]), 0, self.size - 1)
                        new_y = np.clip(agent.y + random.choice([-1, 0, 1]), 0, self.size - 1)
                        
                        if isinstance(agent, PCDPlus):
                            new_offspring.append(PCDPlus(new_x, new_y, biomass=agent.biomass))
                        else:
                            new_offspring.append(PCDMinus(new_x, new_y, biomass=agent.biomass))

            # --- DEAD CELL LOGIC ---
            if not agent.alive and agent.bound_ANTIFUNGAL < MAX_ANTIFUNGAL_BINDING:
                capacity_remaining = MAX_ANTIFUNGAL_BINDING - agent.bound_ANTIFUNGAL
                want_to_absorb = min(BINDING_RATE_ANTIFUNGAL * TIME_STEP_DT, capacity_remaining)
                actual_absorption = min(want_to_absorb, self.ANTIFUNGAL_layer[agent.y, agent.x])
                
                agent.bound_ANTIFUNGAL += actual_absorption
                self.ANTIFUNGAL_layer[agent.y, agent.x] -= actual_absorption
                
        # Append all new baby cells to the main population
        self.agents.extend(new_offspring)
        
        # 2. Update Environment (Pure Diffusion)
        laplacian_n = convolve(self.nutrient_layer, self.laplacian_kernel, mode='reflect')
        self.nutrient_layer += (DIFFUSION_NUTRIENT * laplacian_n) * TIME_STEP_DT
        self.nutrient_layer = np.clip(self.nutrient_layer, 0, None)

        laplacian_f = convolve(self.ANTIFUNGAL_layer, self.laplacian_kernel, mode='reflect')
        self.ANTIFUNGAL_layer += (DIFFUSION_ANTIFUNGAL * laplacian_f) * TIME_STEP_DT
        self.ANTIFUNGAL_layer = np.clip(self.ANTIFUNGAL_layer, 0, None)

    def get_rgb_cell_grid(self):
        img = np.ones((self.size, self.size, 3))
        for agent in self.agents:
            if agent.alive:
                if isinstance(agent, PCDPlus):
                    img[agent.y, agent.x] = [0, 0.8, 0] # Green
                elif isinstance(agent, PCDMinus):
                    img[agent.y, agent.x] = [0, 0, 0.8] # Blue
            else:
                img[agent.y, agent.x] = [0.8, 0, 0] # Red (Dead)
        return img

    def visualize(self, step_num):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        extent = [0, self.size, 0, self.size]

        im1 = axes[0].imshow(self.nutrient_layer, cmap='Greens', extent=extent, origin='lower', vmin=0, vmax=INIT_NUTRIENT_LEVEL)
        axes[0].set_title('Nutrient Layer')
        fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

        im2 = axes[1].imshow(self.ANTIFUNGAL_layer, cmap='Blues', extent=extent, origin='lower', vmin=0, vmax=INIT_ANTIFUNGAL_LEVEL)
        axes[1].set_title('ANTIFUNGAL Layer')
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        rgb_cells = self.get_rgb_cell_grid()
        axes[2].imshow(rgb_cells, extent=extent, origin='lower')
        axes[2].set_title(f'Cells (Total: {len(self.agents)})')
        
        legend_elements = [Patch(facecolor=[0, 0.8, 0], label='PCD+ (Alive)'),
                           Patch(facecolor=[0, 0, 0.8], label='PCD- (Alive)'),
                           Patch(facecolor=[0.8, 0, 0], label='Dead')]
        axes[2].legend(handles=legend_elements, loc='upper right', fontsize=8)

        plt.suptitle(f"Simulation at Time Step {step_num}")
        plt.tight_layout()
        plt.show()

# ==========================================
# --- RUN SIMULATION ---
# ==========================================
if __name__ == "__main__":
    model = PetriDishModel()
    model.visualize(step_num=0)
    
    for i in range(1, 101):
        model.step_environment()
        
    model.visualize(step_num=100)