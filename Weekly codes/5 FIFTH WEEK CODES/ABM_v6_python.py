import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import convolve
from PIL import Image
import random
import math
from collections import defaultdict
import io

# ==========================================
# --- GLOBAL EXPERIMENTAL PARAMETERS ---
# ==========================================

GRID_SIZE_PX = 150 # 1 px = 5 µm, making the physical grid 500 µm across
INITIAL_CELLS = 1
SIMULATION_STEPS = 400
ANTIFUNGAL_INJECTION_STEP = 73 # 24h
TIME_STEP_DT = 1.0 / 3.0 # Assuming 1 step = 20 minutes of biological time

# --- Exposure Modes Definition ---
SINGLE_SHOCK = 0
CONTINUOUS = 1
PULSATED = 2

# --- Antifungal Exposure Settings ---
ANTIFUNGAL_EXPOSURE_MODE = SINGLE_SHOCK   
ANTIFUNGAL_PULSE_INTERVAL = 100        

# --- Nutrient Exposure Settings ---
NUTRIENT_EXPOSURE_MODE = SINGLE_SHOCK
NUTRIENT_INJECTION_STEP = 0          # Step 0 means it just uses the initial agar nutrients without mid-sim replenishments
NUTRIENT_PULSE_INTERVAL = 36         # If set to PULSATED, how often nutrients are replenished

# --- Biomass Capacity Limits ---
MAX_BIOMASS_PER_PX = 3.0 

# --- Environment Levels ---
INIT_NUTRIENT_LEVEL = 12.0 
INIT_ANTIFUNGAL_LEVEL = 5.0 
    
# --- Diffusion Settings ---
DIFFUSION_NUTRIENT = 0.57
DIFFUSION_ANTIFUNGAL = 0.3
DIFFUSION_ITERATIONS = 15 

# --- Growth & Metabolism ---
MU_MAX = 0.7            
MONOD_KS = 5.0           
MAINTENANCE_COEFF = 0.015 
YIELD_TRUE = 0.39        
NEWBORN_BIOMASS = 1.0    
DIVISION_BIOMASS = 2.0   
STARVATION_BIOMASS = 0.75 
RESERVOIR_FRACTION = 0.1 

# --- Mechanics & Space ---
PUSH_PROBABILITY = 0.3 
MAX_PUSH_RADIUS = 10    

# --- Antifungal Binding Kinetics ---
MAX_ANTIFUNGAL_BINDING_LIVE = 0.42       # Capacity for intact, living cells
MAX_ANTIFUNGAL_BINDING_DEAD_APOP = 2.5   # Capacity for dead apoptotic cells (higher sponge effect)
MAX_ANTIFUNGAL_BINDING_DEAD_NECRO = 0.0  # Capacity for dead necrotic/starved cells (lower sponge effect)
K_ON_ANTIFUNGAL = 0.01              # Adsorption rate constant
K_OFF_ANTIFUNGAL = 0.005            # Desorption rate constant

# --- Stress, Apoptosis & Necrosis ---
ANTIFUNGAL_DAMAGE_THRESHOLD = 3.0   # Threshold to start accumulating damage (x-intercept)
STRESS_START_TIME = 0.0             # Hours of exposure before death risks begin
APOPTOSIS_DURATION = 2.0            # Hours the apoptosis process takes
APOPTOSIS_LEAK_RATE = 0.5           # Fraction of current internal nutrients leaked per hour during apoptosis
ASSAY_DURATION_HOURS = 200.0/60.0   # Calibration time for dose-response percentages

# --- NEW: Continuous Mechanistic Death Parameters ---
TOTAL_DEATH_MAX = 0.999        # Caps at 99.9%
TOTAL_DEATH_STEEPNESS = 0.8    # How fast the colony dies
TOTAL_DEATH_C50 = 4.0          # Dose where 50% of cells die

NECRO_SLOPE = 0.056571         # Linear increase of necrosis per µg/ml
NECRO_INTERCEPT = -0.036       # Y-intercept of the necrosis line


# ==========================================
# --- AGENT TYPES ---
# ==========================================

class CellAgent:
    def __init__(self, agent_id, pos, biomass, internal_nutrients):
        self.id = agent_id
        self.pos = pos
        self.alive = True
        self.dead_necrosis = False
        self.dead_starvation = False
        self.bound_ANTIFUNGAL = 0.0
        self.biomass = biomass
        self.internal_nutrients = internal_nutrients
        self.ANTIFUNGAL_exposure_time = 0.0
        
    @property
    def is_apoptotic(self): return False
    @property
    def dead_apoptosis(self): return False
    @property
    def can_divide(self): return True


class PCDPlusCell(CellAgent):
    def __init__(self, agent_id, pos, biomass, internal_nutrients, can_divide_flag=True):
        super().__init__(agent_id, pos, biomass, internal_nutrients)
        self._is_apoptotic = False
        self._dead_apoptosis = False
        self.apoptosis_timer = 0.0
        self._can_divide = can_divide_flag
        
    @property
    def is_apoptotic(self): return self._is_apoptotic
    @is_apoptotic.setter
    def is_apoptotic(self, val): self._is_apoptotic = val
    
    @property
    def dead_apoptosis(self): return self._dead_apoptosis
    
    @property
    def can_divide(self): return self._can_divide
    
    def set_dead_apoptosis(self):
        self.alive = False
        self.is_apoptotic = False
        self._dead_apoptosis = True


class PCDMinusCell(CellAgent):
    pass


# ==========================================
# --- MODEL PROPERTIES & INITIALIZATION ---
# ==========================================

class PetriDishModel:
    def __init__(self, agent_class, starting_positions):
        self.grid_size = GRID_SIZE_PX
        self.nutrient_layer = np.full((GRID_SIZE_PX, GRID_SIZE_PX), INIT_NUTRIENT_LEVEL, dtype=float)
        self.ANTIFUNGAL_layer = np.zeros((GRID_SIZE_PX, GRID_SIZE_PX), dtype=float)
        
        self.laplacian_kernel = np.array([[1/6, 2/3, 1/6], [2/3, -10/3, 2/3], [1/6, 2/3, 1/6]])
        self.neighbor_kernel = np.array([[1/6, 2/3, 1/6], [2/3, 0.0, 2/3], [1/6, 2/3, 1/6]])
        
        self.is_pcd_plus = (agent_class == PCDPlusCell)
        self.total_lost_nutrients = 0.0
        
        self.agents = {}
        self.next_id = 1
        
        init_internal = NEWBORN_BIOMASS * RESERVOIR_FRACTION
        
        for pos in starting_positions:
            if self.is_pcd_plus:
                agent = PCDPlusCell(self.next_id, pos, NEWBORN_BIOMASS, init_internal, True)
            else:
                agent = PCDMinusCell(self.next_id, pos, NEWBORN_BIOMASS, init_internal)
            self.agents[agent.id] = agent
            self.next_id += 1

    def get_agents_at(self, pos):
        # In a very large grid, spatial hashing is better. Given the size, a full filter is ok, 
        # but a spatial dict is much faster.
        return [a for a in self.agents.values() if a.pos == pos]

    def get_nearby_positions(self, pos, r):
        x, y = pos
        positions = []
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if dx == 0 and dy == 0: continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    positions.append((nx, ny))
        return positions


def get_death_rates(c):
    if c < ANTIFUNGAL_DAMAGE_THRESHOLD:
        return 0.0, 0.0
    
    total_death = TOTAL_DEATH_MAX / (1.0 + math.exp(-TOTAL_DEATH_STEEPNESS * (c - TOTAL_DEATH_C50)))
    necro_raw = (NECRO_SLOPE * c) + NECRO_INTERCEPT
    
    target_necro_frac = max(0.0, min(necro_raw, total_death))
    target_apop_frac = total_death - target_necro_frac
    target_total_frac = min(0.999, total_death)
    
    if target_total_frac <= 0.01: 
        return 0.0, 0.0
        
    hourly_total_rate = -math.log(1.0 - target_total_frac) / ASSAY_DURATION_HOURS
    
    ratio_apop = target_apop_frac / total_death
    ratio_necro = target_necro_frac / total_death
    
    return hourly_total_rate * ratio_apop, hourly_total_rate * ratio_necro


def apply_stress_plus(agent, local_ANTIFUNGAL, model):
    if not agent.alive: return

    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD:
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT

    if agent.is_apoptotic:
        agent.apoptosis_timer += TIME_STEP_DT
        x, y = agent.pos
        leak_fraction = min(1.0, APOPTOSIS_LEAK_RATE * TIME_STEP_DT)
        leak_amount = agent.internal_nutrients * leak_fraction
        agent.internal_nutrients -= leak_amount
        model.nutrient_layer[y, x] += leak_amount

        if agent.apoptosis_timer >= APOPTOSIS_DURATION:
            agent.set_dead_apoptosis()
            model.nutrient_layer[y, x] += agent.internal_nutrients
            agent.internal_nutrients = 0.0
    else:
        if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME:
            rate_apop, rate_necro = get_death_rates(local_ANTIFUNGAL)
            total_rate = rate_apop + rate_necro
            
            if total_rate > 0:
                prob_death = 1.0 - math.exp(-total_rate * TIME_STEP_DT)
                if random.random() < prob_death:
                    prob_apop_given_death = rate_apop / total_rate
                    if random.random() < prob_apop_given_death:
                        agent.is_apoptotic = True 
                    else:
                        agent.alive = False
                        agent.dead_necrosis = True
                        model.nutrient_layer[agent.pos[1], agent.pos[0]] += agent.internal_nutrients
                        agent.internal_nutrients = 0.0
                        
                        model.ANTIFUNGAL_layer[agent.pos[1], agent.pos[0]] += agent.bound_ANTIFUNGAL
                        agent.bound_ANTIFUNGAL = 0.0


def apply_stress_minus(agent, local_ANTIFUNGAL, model):
    if not agent.alive: return

    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD:
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT

    if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME:
        rate_apop, rate_necro = get_death_rates(local_ANTIFUNGAL)
        total_rate = rate_apop + rate_necro
        
        if total_rate > 0:
            prob_death = 1.0 - math.exp(-total_rate * TIME_STEP_DT)
            if random.random() < prob_death:
                agent.alive = False
                agent.dead_necrosis = True
                
                model.nutrient_layer[agent.pos[1], agent.pos[0]] += agent.internal_nutrients
                agent.internal_nutrients = 0.0
                
                model.ANTIFUNGAL_layer[agent.pos[1], agent.pos[0]] += agent.bound_ANTIFUNGAL
                agent.bound_ANTIFUNGAL = 0.0


# ==========================================
# --- CORE MODEL STEP (Execution Logic) ---
# ==========================================

def complex_model_step(model):
    n_demands = {}
    f_demands = {}
    f_releases = {}

    # Faster spatial lookups
    spatial_grid = defaultdict(list)
    for agent in model.agents.values():
        spatial_grid[agent.pos].append(agent)

    # --- PASS 1: Calculate Demands ---
    for agent in model.agents.values():
        x, y = agent.pos
        
        if agent.alive:
            local_n = model.nutrient_layer[y, x]
            max_reservoir = agent.biomass * RESERVOIR_FRACTION
            reservoir_deficit = max(0.0, max_reservoir - agent.internal_nutrients)

            if agent.is_apoptotic:
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                n_demands[agent.id] = maintenance_cost + reservoir_deficit
            else:
                if agent.biomass < DIVISION_BIOMASS:
                    mu = MU_MAX * (local_n / (MONOD_KS + local_n))
                    max_possible_growth_biomass = min(mu * agent.biomass * TIME_STEP_DT, DIVISION_BIOMASS - agent.biomass)
                    growth_demand_n = max_possible_growth_biomass / YIELD_TRUE
                else:
                    growth_demand_n = 0.0
                
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                n_demands[agent.id] = growth_demand_n + maintenance_cost + reservoir_deficit
                
        local_f = model.ANTIFUNGAL_layer[y, x]
        bound_f = agent.bound_ANTIFUNGAL
        
        if agent.alive:
            current_max = MAX_ANTIFUNGAL_BINDING_LIVE
        elif agent.dead_apoptosis:
            current_max = MAX_ANTIFUNGAL_BINDING_DEAD_APOP
        else:
            current_max = MAX_ANTIFUNGAL_BINDING_DEAD_NECRO

        cap_remaining = max(0.0, current_max - bound_f)
        net_change = (K_ON_ANTIFUNGAL * local_f * cap_remaining - K_OFF_ANTIFUNGAL * bound_f) * TIME_STEP_DT

        if net_change > 0:
            f_demands[agent.id] = net_change
        else:
            f_releases[agent.id] = min(-net_change, bound_f)

    newborn_spots = []
    planned_biomass = defaultdict(float)

    # --- PASS 2: Allocate & Update ---
    agent_list = list(model.agents.values())
    random.shuffle(agent_list)

    for agent in agent_list:
        x, y = agent.pos
        
        if agent.alive:
            if isinstance(agent, PCDPlusCell):
                apply_stress_plus(agent, model.ANTIFUNGAL_layer[y, x], model)
            else:
                apply_stress_minus(agent, model.ANTIFUNGAL_layer[y, x], model)

        if agent.alive and n_demands.get(agent.id, 0.0) > 0:
            demand = n_demands[agent.id]
            actual_intake = min(demand, model.nutrient_layer[y, x])
            model.nutrient_layer[y, x] -= actual_intake
            agent.internal_nutrients += actual_intake

            maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

            if agent.is_apoptotic:
                burn = min(agent.internal_nutrients, maintenance_cost)
                agent.internal_nutrients -= burn
                model.total_lost_nutrients += burn
            else:
                if agent.internal_nutrients >= maintenance_cost:
                    agent.internal_nutrients -= maintenance_cost
                    model.total_lost_nutrients += maintenance_cost

                    if agent.biomass < DIVISION_BIOMASS:
                        max_growth_biomass = min(MU_MAX * agent.biomass * TIME_STEP_DT, DIVISION_BIOMASS - agent.biomass)
                        max_growth_n = max_growth_biomass / YIELD_TRUE
                        
                        actual_growth_n = min(agent.internal_nutrients, max_growth_n)
                        actual_growth_biomass = actual_growth_n * YIELD_TRUE
                        
                        agent.biomass += actual_growth_biomass
                        agent.internal_nutrients -= actual_growth_n
                else:
                    remaining_deficit = maintenance_cost - agent.internal_nutrients
                    model.total_lost_nutrients += agent.internal_nutrients
                    agent.internal_nutrients = 0.0
                    
                    biomass_burned = remaining_deficit * YIELD_TRUE
                    agent.biomass -= biomass_burned
                    model.total_lost_nutrients += remaining_deficit
                    
                    if agent.biomass <= STARVATION_BIOMASS:
                        agent.alive = False
                        agent.dead_starvation = True
                        
                        model.ANTIFUNGAL_layer[y, x] += agent.bound_ANTIFUNGAL
                        agent.bound_ANTIFUNGAL = 0.0

                if agent.alive:
                    max_reservoir = agent.biomass * RESERVOIR_FRACTION
                    if agent.internal_nutrients > max_reservoir:
                        excess = agent.internal_nutrients - max_reservoir
                        agent.internal_nutrients = max_reservoir
                        model.nutrient_layer[y, x] += excess

                    if agent.biomass >= DIVISION_BIOMASS:
                        if not agent.can_divide:
                            excess = agent.biomass - DIVISION_BIOMASS
                            agent.biomass = DIVISION_BIOMASS
                            model.total_lost_nutrients += (excess / YIELD_TRUE) 
                        else:
                            chosen_spot = None
                            immediate_hood = [agent.pos] + model.get_nearby_positions(agent.pos, 1)
                            
                            spot_status = []
                            for p in immediate_hood:
                                cur_b = sum(a.biomass for a in spatial_grid[p])
                                total_b = cur_b + planned_biomass[p]
                                spot_status.append({'pos': p, 'bio': total_b})
                                
                            available_spots = [s for s in spot_status if s['bio'] + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX]
                            
                            if available_spots:
                                truly_empty = [s for s in available_spots if s['bio'] == 0.0]
                                candidate_pool = truly_empty if truly_empty else available_spots
                                
                                weights = []
                                for s in candidate_pool:
                                    occupied_neighbors = 0
                                    for np_pos in model.get_nearby_positions(s['pos'], 1):
                                        cur_b = sum(a.biomass for a in spatial_grid[np_pos])
                                        if (cur_b + planned_biomass[np_pos]) > 0.0:
                                            occupied_neighbors += 1
                                    weights.append(float(occupied_neighbors)**3 + 1.0)
                                    
                                chosen_spot = random.choices([s['pos'] for s in candidate_pool], weights=weights, k=1)[0]
                                
                            elif random.random() < PUSH_PROBABILITY:
                                block = model.get_nearby_positions(agent.pos, MAX_PUSH_RADIUS)
                                empty_spots = []
                                for p in block:
                                    cur_b = sum(a.biomass for a in spatial_grid[p])
                                    if cur_b + planned_biomass[p] + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX:
                                        empty_spots.append(p)
                                        
                                if empty_spots:
                                    min_dist = min((x - p[0])**2 + (y - p[1])**2 for p in empty_spots)
                                    best_spots = [p for p in empty_spots if (x - p[0])**2 + (y - p[1])**2 <= min_dist + 2]
                                    
                                    weights = []
                                    for p in best_spots:
                                        cur_b = sum(a.biomass for a in spatial_grid[p])
                                        weights.append(MAX_BIOMASS_PER_PX - (cur_b + planned_biomass[p]))
                                        
                                    chosen_spot = random.choices(best_spots, weights=weights, k=1)[0]

                            if chosen_spot is not None:
                                agent.biomass -= NEWBORN_BIOMASS 
                                daughter_fraction = NEWBORN_BIOMASS / (agent.biomass + NEWBORN_BIOMASS)
                                daughter_n = agent.internal_nutrients * daughter_fraction
                                agent.internal_nutrients -= daughter_n

                                planned_biomass[chosen_spot] += NEWBORN_BIOMASS
                                newborn_spots.append((chosen_spot, NEWBORN_BIOMASS, daughter_n))
                            else:
                                excess = agent.biomass - DIVISION_BIOMASS
                                agent.biomass = DIVISION_BIOMASS
                                model.total_lost_nutrients += (excess / YIELD_TRUE)

        if f_demands.get(agent.id, 0.0) > 0:
            demand = f_demands[agent.id]
            actual_binding = min(demand, model.ANTIFUNGAL_layer[y, x])
            agent.bound_ANTIFUNGAL += actual_binding
            model.ANTIFUNGAL_layer[y, x] -= actual_binding
        elif f_releases.get(agent.id, 0.0) > 0:
            release = f_releases[agent.id]
            agent.bound_ANTIFUNGAL -= release
            model.ANTIFUNGAL_layer[y, x] += release

    for pos, b, n in newborn_spots:
        cur_b = sum(a.biomass for a in spatial_grid[pos])
        if cur_b + b <= MAX_BIOMASS_PER_PX + 0.01:
            if model.is_pcd_plus:
                new_agent = PCDPlusCell(model.next_id, pos, b, n, True)
            else:
                new_agent = PCDMinusCell(model.next_id, pos, b, n)
            model.agents[new_agent.id] = new_agent
            spatial_grid[pos].append(new_agent)
            model.next_id += 1
        else:
            model.total_lost_nutrients += (b / YIELD_TRUE) + n

    # --- PDE Diffusion ---
    alpha_n = DIFFUSION_NUTRIENT * TIME_STEP_DT
    alpha_f = DIFFUSION_ANTIFUNGAL * TIME_STEP_DT

    rhs_n = model.nutrient_layer + (alpha_n / 2.0) * convolve(model.nutrient_layer, model.laplacian_kernel, mode='nearest')
    rhs_f = model.ANTIFUNGAL_layer + (alpha_f / 2.0) * convolve(model.ANTIFUNGAL_layer, model.laplacian_kernel, mode='nearest')

    u_n = np.copy(model.nutrient_layer)
    u_f = np.copy(model.ANTIFUNGAL_layer)

    denom_n = 1.0 + (5.0 / 3.0) * alpha_n
    denom_f = 1.0 + (5.0 / 3.0) * alpha_f

    for _ in range(DIFFUSION_ITERATIONS):
        u_n = (rhs_n + (alpha_n / 2.0) * convolve(u_n, model.neighbor_kernel, mode='nearest')) / denom_n
        u_f = (rhs_f + (alpha_f / 2.0) * convolve(u_f, model.neighbor_kernel, mode='nearest')) / denom_f

    np.maximum(u_n, 0.0, out=model.nutrient_layer)
    np.maximum(u_f, 0.0, out=model.ANTIFUNGAL_layer)


# ==========================================
# --- UTILITY & MASS BALANCES ---
# ==========================================

def get_antifungal_breakdown(model):
    env_mass = np.sum(model.ANTIFUNGAL_layer)
    bound_live = sum(a.bound_ANTIFUNGAL for a in model.agents.values() if a.alive)
    bound_dead = sum(a.bound_ANTIFUNGAL for a in model.agents.values() if not a.alive)
    total = env_mass + bound_live + bound_dead
    return {'env': env_mass, 'live': bound_live, 'dead': bound_dead, 'total': total}

def get_nutrient_breakdown(model):
    env_mass = np.sum(model.nutrient_layer)
    internal = sum(a.internal_nutrients for a in model.agents.values())
    biomass_eq = sum(a.biomass / YIELD_TRUE for a in model.agents.values())
    burned = model.total_lost_nutrients
    total = env_mass + internal + biomass_eq + burned
    return {'env': env_mass, 'internal': internal, 'biomass': biomass_eq, 'burned': burned, 'total': total}

def generate_subplots(model, title_prefix, step, ax1, ax2, ax3, ax4):
    alive_agents = [a for a in model.agents.values() if a.alive]
    
    def jitter(val): return val + 0.4 * (random.random() - 0.5)

    healthy_x = [jitter(a.pos[0]) for a in alive_agents if not a.is_apoptotic]
    healthy_y = [jitter(a.pos[1]) for a in alive_agents if not a.is_apoptotic]

    apoptotic_x = [jitter(a.pos[0]) for a in alive_agents if a.is_apoptotic]
    apoptotic_y = [jitter(a.pos[1]) for a in alive_agents if a.is_apoptotic]

    dead_apoptosis_x = [jitter(a.pos[0]) for a in model.agents.values() if a.dead_apoptosis]
    dead_apoptosis_y = [jitter(a.pos[1]) for a in model.agents.values() if a.dead_apoptosis]

    dead_necrosis_x = [jitter(a.pos[0]) for a in model.agents.values() if a.dead_necrosis]
    dead_necrosis_y = [jitter(a.pos[1]) for a in model.agents.values() if a.dead_necrosis]

    dead_starved_x = [jitter(a.pos[0]) for a in model.agents.values() if a.dead_starvation]
    dead_starved_y = [jitter(a.pos[1]) for a in model.agents.values() if a.dead_starvation]

    num_live = len(alive_agents)
    
    n_plot = np.copy(model.nutrient_layer)
    f_plot = np.copy(model.ANTIFUNGAL_layer)

    ax1.clear()
    ax1.imshow(n_plot, cmap='viridis', vmin=0, vmax=INIT_NUTRIENT_LEVEL, origin='lower')
    ax1.set_title(f"{title_prefix} Nutrients (Step {step})")

    ax2.clear()
    ax2.imshow(f_plot, cmap='Blues', vmin=0, vmax=INIT_ANTIFUNGAL_LEVEL, origin='lower')
    ax2.set_title(f"{title_prefix} Antifungal")

    ax3.clear()
    ax3.set_xlim(0, GRID_SIZE_PX)
    ax3.set_ylim(0, GRID_SIZE_PX)
    ax3.set_title(f"{title_prefix} Live Cells: {num_live}")
    ax3.set_aspect('equal')

    healthy_color = 'blue' if model.is_pcd_plus else 'red'
    healthy_label = 'PCD+' if model.is_pcd_plus else 'PCD-'

    if dead_starved_x:
        ax3.scatter(dead_starved_x, dead_starved_y, label="Dead (Starve)", color='magenta', s=5, edgecolors='none')
    if dead_necrosis_x:
        ax3.scatter(dead_necrosis_x, dead_necrosis_y, label="Dead (Necro)", color='black', s=5, edgecolors='none')
    if dead_apoptosis_x:
        ax3.scatter(dead_apoptosis_x, dead_apoptosis_y, label="Dead (Apop)", color='darkgray', s=5, edgecolors='none')
    if apoptotic_x:
        ax3.scatter(apoptotic_x, apoptotic_y, label="Apoptotic", color='orange', s=4, edgecolors='none')
    if healthy_x:
        ax3.scatter(healthy_x, healthy_y, label=healthy_label, color=healthy_color, s=3, edgecolors='none')
        
    ax3.legend(loc='upper right', fontsize='x-small')

    density_grid = np.zeros((GRID_SIZE_PX, GRID_SIZE_PX))
    for a in model.agents.values():
        density_grid[a.pos[1], a.pos[0]] += a.biomass

    cmap = mcolors.LinearSegmentedColormap.from_list("", ["white", "green", "yellow", "orange"])
    ax4.clear()
    im = ax4.imshow(density_grid, cmap=cmap, vmin=0, vmax=MAX_BIOMASS_PER_PX, origin='lower')
    ax4.set_title(f"{title_prefix} Biomass Density")


# ==========================================
# --- RUN SIMULATION ---
# ==========================================

def main():
    print("Initializing Python Grid models (Capacity Version)...")
    
    cx, cy = GRID_SIZE_PX / 2.0, GRID_SIZE_PX / 2.0
    all_pos = [(x, y) for x in range(GRID_SIZE_PX) for y in range(GRID_SIZE_PX)]
    all_pos.sort(key=lambda p: (p[0] - cx)**2 + (p[1] - cy)**2)
    starting_positions = all_pos[:min(INITIAL_CELLS, len(all_pos))]
    
    model_plus = PetriDishModel(PCDPlusCell, starting_positions)
    model_minus = PetriDishModel(PCDMinusCell, starting_positions)
    
    every_n_steps = 6 
    fps = 10 

    history_plus = {'alive': [], 'dead_apop': [], 'dead_necro': [], 'dead_starve': [], 'total': []}
    history_minus = {'alive': [], 'dead_apop': [], 'dead_necro': [], 'dead_starve': [], 'total': []}

    def record_metrics(hist, mod):
        hist['alive'].append(sum(1 for a in mod.agents.values() if a.alive))
        hist['dead_apop'].append(sum(1 for a in mod.agents.values() if a.dead_apoptosis))
        hist['dead_necro'].append(sum(1 for a in mod.agents.values() if a.dead_necrosis))
        hist['dead_starve'].append(sum(1 for a in mod.agents.values() if a.dead_starvation))
        hist['total'].append(len(mod.agents))

    record_metrics(history_plus, model_plus)
    record_metrics(history_minus, model_minus)

    injected_n_plus = (GRID_SIZE_PX * GRID_SIZE_PX * INIT_NUTRIENT_LEVEL) + (NEWBORN_BIOMASS / YIELD_TRUE) + (NEWBORN_BIOMASS * RESERVOIR_FRACTION)
    injected_n_minus = injected_n_plus
    injected_f_plus = 0.0
    injected_f_minus = 0.0

    print("Starting dual simulation and recording frames...")
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    plt.tight_layout()
    frames = []

    for step in range(1, SIMULATION_STEPS + 1):
        
        inject_nutrient_now = False
        if NUTRIENT_EXPOSURE_MODE == SINGLE_SHOCK:
            inject_nutrient_now = (step == NUTRIENT_INJECTION_STEP)
        elif NUTRIENT_EXPOSURE_MODE == CONTINUOUS:
            inject_nutrient_now = (step >= NUTRIENT_INJECTION_STEP)
        elif NUTRIENT_EXPOSURE_MODE == PULSATED:
            inject_nutrient_now = (step >= NUTRIENT_INJECTION_STEP) and ((step - NUTRIENT_INJECTION_STEP) % NUTRIENT_PULSE_INTERVAL == 0)

        inject_antifungal_now = False
        if ANTIFUNGAL_EXPOSURE_MODE == SINGLE_SHOCK:
            inject_antifungal_now = (step == ANTIFUNGAL_INJECTION_STEP)
        elif ANTIFUNGAL_EXPOSURE_MODE == CONTINUOUS:
            inject_antifungal_now = (step >= ANTIFUNGAL_INJECTION_STEP)
        elif ANTIFUNGAL_EXPOSURE_MODE == PULSATED:
            inject_antifungal_now = (step >= ANTIFUNGAL_INJECTION_STEP) and ((step - ANTIFUNGAL_INJECTION_STEP) % ANTIFUNGAL_PULSE_INTERVAL == 0)

        if inject_nutrient_now:
            if step == NUTRIENT_INJECTION_STEP or NUTRIENT_EXPOSURE_MODE == PULSATED:
                print(f"--- REPLENISHING NUTRIENTS AT STEP {step} ---")
            injected_n_plus += np.sum(INIT_NUTRIENT_LEVEL - model_plus.nutrient_layer)
            injected_n_minus += np.sum(INIT_NUTRIENT_LEVEL - model_minus.nutrient_layer)
            model_plus.nutrient_layer.fill(INIT_NUTRIENT_LEVEL)
            model_minus.nutrient_layer.fill(INIT_NUTRIENT_LEVEL)

        if inject_antifungal_now:
            if step == ANTIFUNGAL_INJECTION_STEP or ANTIFUNGAL_EXPOSURE_MODE == PULSATED:
                print(f"--- INJECTING ANTIFUNGAL AT STEP {step} ---")
            injected_f_plus += np.sum(INIT_ANTIFUNGAL_LEVEL - model_plus.ANTIFUNGAL_layer)
            injected_f_minus += np.sum(INIT_ANTIFUNGAL_LEVEL - model_minus.ANTIFUNGAL_layer)
            model_plus.ANTIFUNGAL_layer.fill(INIT_ANTIFUNGAL_LEVEL)
            model_minus.ANTIFUNGAL_layer.fill(INIT_ANTIFUNGAL_LEVEL)

        complex_model_step(model_plus)
        complex_model_step(model_minus)

        record_metrics(history_plus, model_plus)
        record_metrics(history_minus, model_minus)

        if step % every_n_steps == 0:
            generate_subplots(model_plus, "PCD+", step, axes[0,0], axes[0,1], axes[0,2], axes[0,3])
            generate_subplots(model_minus, "PCD-", step, axes[1,0], axes[1,1], axes[1,2], axes[1,3])
            
            fig.canvas.draw()
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight')
            img_buf.seek(0)
            frames.append(Image.open(img_buf))
            
        if step % 10 == 0:
            print(f"Progress: Step {step} / {SIMULATION_STEPS}")

    gif_name = "dual_petri_dish_simulation.gif"
    if frames:
        frames[0].save(gif_name, save_all=True, append_images=frames[1:], optimize=False, duration=int(1000/fps), loop=0)
    print(f"Success! GIF saved as: {gif_name}")
    plt.close(fig)

    # --- CALCULATE EXPERIMENTAL DOUBLING TIME ---
    t_phase_hours = ANTIFUNGAL_INJECTION_STEP * TIME_STEP_DT
    
    n0_plus = history_plus['alive'][0]
    nt_plus = history_plus['alive'][ANTIFUNGAL_INJECTION_STEP]
    if nt_plus > n0_plus:
        td_plus = t_phase_hours * math.log(2) / math.log(nt_plus / n0_plus)
        print(f"=> PCD+ Estimated Doubling Time (pre-injection): {td_plus:.2f} hours")
    else:
        print("=> PCD+ Estimated Doubling Time: N/A (no net growth)")

    n0_minus = history_minus['alive'][0]
    nt_minus = history_minus['alive'][ANTIFUNGAL_INJECTION_STEP]
    if nt_minus > n0_minus:
        td_minus = t_phase_hours * math.log(2) / math.log(nt_minus / n0_minus)
        print(f"=> PCD- Estimated Doubling Time (pre-injection): {td_minus:.2f} hours")
    else:
        print("=> PCD- Estimated Doubling Time: N/A (no net growth)")

    # --- MASS BALANCE REPORT ---
    print("\n=================================================")
    print(f"FINAL MASS BALANCE REPORT (Step {SIMULATION_STEPS})")
    print("=================================================")
    
    n_plus = get_nutrient_breakdown(model_plus)
    f_plus = get_antifungal_breakdown(model_plus)
    
    n_minus = get_nutrient_breakdown(model_minus)
    f_minus = get_antifungal_breakdown(model_minus)

    print("PCD+ NUTRIENTS:")
    print(f"  Initial/Injected:  {injected_n_plus:.2f}")
    print(f"  Final Total:       {n_plus['total']:.2f}")
    print(f"    -> Environment:  {n_plus['env']:.2f}")
    print(f"    -> Biomass (Eq): {n_plus['biomass']:.2f}")
    print(f"    -> Reservoir:    {n_plus['internal']:.2f}")
    print(f"    -> Burned/Lost:  {n_plus['burned']:.2f}\n")

    print("PCD+ ANTIFUNGAL:")
    print(f"  Initial/Injected:  {injected_f_plus:.2f}")
    print(f"  Final Total:       {f_plus['total']:.2f}")
    print(f"    -> Environment:  {f_plus['env']:.2f}")
    print(f"    -> Bound(Alive): {f_plus['live']:.2f}")
    print(f"    -> Bound(Dead):  {f_plus['dead']:.2f}\n")

    print("PCD- NUTRIENTS:")
    print(f"  Initial/Injected:  {injected_n_minus:.2f}")
    print(f"  Final Total:       {n_minus['total']:.2f}")
    print(f"    -> Environment:  {n_minus['env']:.2f}")
    print(f"    -> Biomass (Eq): {n_minus['biomass']:.2f}")
    print(f"    -> Reservoir:    {n_minus['internal']:.2f}")
    print(f"    -> Burned/Lost:  {n_minus['burned']:.2f}\n")

    print("PCD- ANTIFUNGAL:")
    print(f"  Initial/Injected:  {injected_f_minus:.2f}")
    print(f"  Final Total:       {f_minus['total']:.2f}")
    print(f"    -> Environment:  {f_minus['env']:.2f}")
    print(f"    -> Bound(Alive): {f_minus['live']:.2f}")
    print(f"    -> Bound(Dead):  {f_minus['dead']:.2f}\n")

    # --- GENERATE REQUESTED DYNAMICS PLOTS ---
    time_axis = [i * TIME_STEP_DT for i in range(SIMULATION_STEPS + 1)]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
    
    ax1.set_title("Population (Alive) Over Time")
    ax1.set_xlabel("Time (hrs)")
    ax1.set_ylabel("Cells")
    ax1.plot(time_axis, history_plus['alive'], label="PCD+ Alive", color='blue', linewidth=2)
    ax1.plot(time_axis, history_minus['alive'], label="PCD- Alive", color='red', linewidth=2)
    ax1.legend()

    ax2.set_title("Dead Cells On Grid Over Time")
    ax2.set_xlabel("Time (hrs)")
    ax2.set_ylabel("Dead Cells")
    ax2.plot(time_axis, history_plus['dead_apop'], label="PCD+ Apop", color='orange', linewidth=2)
    ax2.plot(time_axis, history_plus['dead_necro'], label="PCD+ Necro", color='black', linewidth=2)
    ax2.plot(time_axis, history_minus['dead_necro'], label="PCD- Necro", color='gray', linestyle='dashed', linewidth=2)
    ax2.plot(time_axis, history_plus['dead_starve'], label="PCD+ Starved", color='magenta', linewidth=2)
    ax2.plot(time_axis, history_minus['dead_starve'], label="PCD- Starved", color='purple', linestyle='dashed', linewidth=2)
    ax2.legend()

    surv_plus = [(a / tot * 100.0) if tot > 0 else 0.0 for a, tot in zip(history_plus['alive'], history_plus['total'])]
    surv_minus = [(a / tot * 100.0) if tot > 0 else 0.0 for a, tot in zip(history_minus['alive'], history_minus['total'])]
    
    ax3.set_title("Survivability (% Alive) Over Time")
    ax3.set_xlabel("Time (hrs)")
    ax3.set_ylabel("Survival (%)")
    ax3.set_ylim(0, 105)
    ax3.plot(time_axis, surv_plus, label="PCD+", color='blue', linewidth=2)
    ax3.plot(time_axis, surv_minus, label="PCD-", color='red', linewidth=2)
    ax3.legend()

    plt.tight_layout()
    plt.savefig("population_dynamics.png")
    print("Success! Dynamics plots saved as: population_dynamics.png")


if __name__ == "__main__":
    main()