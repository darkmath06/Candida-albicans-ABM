using Agents
using Random
using StatsBase
using Images
using ImageFiltering
using Plots
using Printf

# ==========================================
# --- GLOBAL EXPERIMENTAL PARAMETERS ---
# ==========================================

const GRID_SIZE_PX = 150 # 1 px = 5 µm, making the physical grid 500 µm across
const INITIAL_CELLS = 1
const SIMULATION_STEPS = 600
const ANTIFUNGAL_INJECTION_STEP = 73 # 24h
const TIME_STEP_DT = 1.0 / 3.0 # Assuming 1 step = 20 minutes of biological time

# --- Exposure Modes Definition ---
@enum ExposureMode SINGLE_SHOCK CONTINUOUS PULSATED
@enum SpatialMode UNIFORM POINT_SOURCES

# --- Antifungal Exposure Settings ---
const ANTIFUNGAL_EXPOSURE_MODE = SINGLE_SHOCK   
const ANTIFUNGAL_PULSE_INTERVAL = 100        

# --- Antifungal Spatial Settings ---
const ANTIFUNGAL_SPATIAL_MODE = POINT_SOURCES
const ANTIFUNGAL_SOURCES = [(100, 40), (40, 100), (40, 40), (100, 100)] # (X, Y) coordinates for injections
const ANTIFUNGAL_SOURCE_DOSE = 10000.0 # High concentration placed at each source pixel to diffuse outwards
const ANTIFUNGAL_SOURCE_RADIUS = 2 # A radius of 2 makes a 5x5 injection patch around each source coordinate

# --- Nutrient Exposure Settings ---
const NUTRIENT_EXPOSURE_MODE = SINGLE_SHOCK
const NUTRIENT_INJECTION_STEP = 0          # Step 0 means it just uses the initial agar nutrients without mid-sim replenishments
const NUTRIENT_PULSE_INTERVAL = 36         # If set to PULSATED, how often nutrients are replenished

# --- Biomass Capacity Limits ---
const MAX_BIOMASS_PER_PX = 3 

# --- Environment Levels ---
const INIT_NUTRIENT_LEVEL = 12.0 
const INIT_ANTIFUNGAL_LEVEL = 5.0 # Used if SPATIAL_MODE is UNIFORM
    
# --- Diffusion Settings ---
const DIFFUSION_NUTRIENT = 0.57
const DIFFUSION_ANTIFUNGAL = 0.3
const DIFFUSION_ITERATIONS = 15 

# --- Growth & Metabolism ---
const MU_MAX = 0.7            
const MONOD_KS = 5.0           
const MAINTENANCE_COEFF = 0.015 
const YIELD_TRUE = 0.39        
const NEWBORN_BIOMASS = 1.0    
const DIVISION_BIOMASS = 2.0   
const STARVATION_BIOMASS = 0.75 
const RESERVOIR_FRACTION = 0.1 

# --- Mechanics & Space ---
const PUSH_PROBABILITY = 0.3 
const MAX_PUSH_RADIUS = 10    

# --- Antifungal Binding Kinetics ---
const MAX_ANTIFUNGAL_BINDING_LIVE = 0.42       # Capacity for intact, living cells
const MAX_ANTIFUNGAL_BINDING_DEAD_APOP = 2.5   # Capacity for dead apoptotic cells (higher sponge effect)
const MAX_ANTIFUNGAL_BINDING_DEAD_NECRO = 0.42 # Capacity for dead necrotic/starved cells
const K_ON_ANTIFUNGAL = 0.01              # Adsorption rate constant
const K_OFF_ANTIFUNGAL = 0.005            # Desorption rate constant

# --- Stress, Apoptosis & Necrosis (UPDATED TO MATCH 1999 PAPER) ---
const ANTIFUNGAL_DAMAGE_THRESHOLD = 0.3      # Threshold to start accumulating damage (x-intercept)
const FUNGISTATIC_THRESHOLD = 0.5            # Above this dose, cells halt growth and replication ("Zombie" state)
const RESUSCITATION_THRESHOLD = 1.0          # Below this dose, apoptotic cells can abort cell death and recover
const STRESS_START_TIME = 0.0                # Hours of exposure before death risks begin
const APOPTOSIS_DURATION = 2.0               # Hours the apoptosis process takes
const APOPTOSIS_LEAK_RATE = 0.5              # Fraction of current internal nutrients leaked per hour during apoptosis
const ASSAY_DURATION_HOURS = 200/60          # Calibration time for dose-response percentages

# --- Continuous Mechanistic Death Parameters ---
const TOTAL_DEATH_MAX = 0.999        # Caps at 99.9%
const TOTAL_DEATH_STEEPNESS = 0.8    # How fast the colony dies
const TOTAL_DEATH_C50 = 4          # Dose where 50% of cells die (Shifted down to match Liao 1999 Data)

const NECRO_SLOPE = 0.056571         # Linear increase of necrosis per µg/ml
const NECRO_INTERCEPT = -0.036       # Y-intercept of the necrosis line

# ==========================================
# --- AGENT TYPES ---
# ==========================================

@agent struct PCDPlusCell(GridAgent{2})
    alive::Bool
    is_apoptotic::Bool
    apoptosis_timer::Float64
    ANTIFUNGAL_exposure_time::Float64
    dead_apoptosis::Bool
    dead_necrosis::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
    internal_nutrients::Float64 
    can_divide::Bool       
end

@agent struct PCDMinusCell(GridAgent{2})
    alive::Bool
    ANTIFUNGAL_exposure_time::Float64
    dead_necrosis::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
    internal_nutrients::Float64 
end

# --- Accessors for Type-Stable Generic Operations ---
is_apoptotic(a::PCDPlusCell) = a.is_apoptotic
is_apoptotic(a::PCDMinusCell) = false

can_divide(a::PCDPlusCell) = a.can_divide
can_divide(a::PCDMinusCell) = true

is_dead_apoptosis(a::PCDPlusCell) = a.dead_apoptosis
is_dead_apoptosis(a::PCDMinusCell) = false

set_dead_apoptosis!(a::PCDPlusCell) = begin
    a.alive = false
    a.is_apoptotic = false
    a.dead_apoptosis = true
end
set_dead_apoptosis!(a::PCDMinusCell) = nothing

# ==========================================
# --- MODEL PROPERTIES & INITIALIZATION ---
# ==========================================

mutable struct PetriDishProperties
    nutrient_layer::Matrix{Float64}
    ANTIFUNGAL_layer::Matrix{Float64}
    laplacian_kernel::Matrix{Float64}
    neighbor_kernel::Matrix{Float64}
    is_pcd_plus::Bool
    total_lost_nutrients::Float64 
end

function initialize_model(AgentType::Type, starting_positions::Vector{Tuple{Int, Int}})
    space = GridSpace((GRID_SIZE_PX, GRID_SIZE_PX); periodic=false)

    props = PetriDishProperties(
        fill(INIT_NUTRIENT_LEVEL, GRID_SIZE_PX, GRID_SIZE_PX),
        fill(0.0, GRID_SIZE_PX, GRID_SIZE_PX),
        [1/6 2/3 1/6; 2/3 -10/3 2/3; 1/6 2/3 1/6],
        [1/6 2/3 1/6; 2/3 0.0 2/3; 1/6 2/3 1/6],
        AgentType === PCDPlusCell,
        0.0
    )

    model = StandardABM(AgentType, space; properties=props, model_step! = complex_model_step!)

    init_internal = NEWBORN_BIOMASS * RESERVOIR_FRACTION

    for pos in starting_positions
        if AgentType === PCDPlusCell
            add_agent!(pos, PCDPlusCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, NEWBORN_BIOMASS, init_internal, true)
        else
            add_agent!(pos, PCDMinusCell, model, true, 0.0, false, false, 0.0, NEWBORN_BIOMASS, init_internal)
        end
    end

    return model
end

# ==========================================
# --- ANTIFUNGAL EXPOSURE LOGIC ---
# ==========================================

function get_death_rates(c::Float64)
    # Biological cutoff: no damage below threshold
    if c < ANTIFUNGAL_DAMAGE_THRESHOLD
        return 0.0, 0.0
    end
    
    # 1. Calculate the TOTAL percentage of cells dying (Logistic S-curve)
    total_death = TOTAL_DEATH_MAX / (1.0 + exp(-TOTAL_DEATH_STEEPNESS * (c - TOTAL_DEATH_C50)))
    
    # 2. Calculate Necrosis using the linear trendline
    necro_raw = (NECRO_SLOPE * c) + NECRO_INTERCEPT
    
    # Bound necrosis so it doesn't drop below 0 or exceed the total death fraction
    target_necro_frac = clamp(necro_raw, 0.0, total_death)
    
    # 3. Apoptosis is simply the remaining death
    target_apop_frac = total_death - target_necro_frac

    target_total_frac = min(0.999, total_death)
    
    # Minor safety check to prevent extremely small rates from eating computation
    if target_total_frac <= 0.01; return 0.0, 0.0; end
    
    # Convert the observed assay fraction to a continuous hourly exponential rate
    hourly_total_rate = -log(1.0 - target_total_frac) / ASSAY_DURATION_HOURS
    
    # Distribute the rate according to the probability ratio
    ratio_apop = target_apop_frac / total_death
    ratio_necro = target_necro_frac / total_death
    
    return hourly_total_rate * ratio_apop, hourly_total_rate * ratio_necro
end

function apply_stress!(agent::PCDPlusCell, local_ANTIFUNGAL::Float64, model)
    if !agent.alive; return; end

    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    end

    if agent.is_apoptotic
        # --- NEW: RESUSCITATION MECHANIC ---
        if local_ANTIFUNGAL < RESUSCITATION_THRESHOLD
            # Drug concentration has dropped; cell aborts apoptosis and recovers!
            agent.is_apoptotic = false
            agent.apoptosis_timer = 0.0
        else
            agent.apoptosis_timer += TIME_STEP_DT
            
            # --- Gradual Nutrient Leak (Membrane Permeabilization) ---
            x, y = agent.pos
            leak_fraction = min(1.0, APOPTOSIS_LEAK_RATE * TIME_STEP_DT)
            leak_amount = agent.internal_nutrients * leak_fraction
            agent.internal_nutrients -= leak_amount
            model.nutrient_layer[y, x] += leak_amount

            if agent.apoptosis_timer >= APOPTOSIS_DURATION
                set_dead_apoptosis!(agent)
                
                # --- NUTRIENT RECYCLING ---
                model.nutrient_layer[y, x] += agent.internal_nutrients
                agent.internal_nutrients = 0.0
            end
        end
    else
        if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME
            rate_apop, rate_necro = get_death_rates(local_ANTIFUNGAL)
            total_rate = rate_apop + rate_necro
            
            if total_rate > 0
                prob_death = 1.0 - exp(-total_rate * TIME_STEP_DT)
                if rand() < prob_death
                    prob_apop_given_death = rate_apop / total_rate
                    if rand() < prob_apop_given_death
                        agent.is_apoptotic = true 
                    else
                        agent.alive = false
                        agent.dead_necrosis = true
                        
                        # Lysis: Instantly dump internal nutrients AND bound antifungal
                        model.nutrient_layer[agent.pos[2], agent.pos[1]] += agent.internal_nutrients
                        agent.internal_nutrients = 0.0
                        
                        model.ANTIFUNGAL_layer[agent.pos[2], agent.pos[1]] += agent.bound_ANTIFUNGAL
                        agent.bound_ANTIFUNGAL = 0.0
                    end
                end
            end
        end
    end
end

function apply_stress!(agent::PCDMinusCell, local_ANTIFUNGAL::Float64, model)
    if !agent.alive; return; end

    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    end

    if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME
        rate_apop, rate_necro = get_death_rates(local_ANTIFUNGAL)
        
        # The PCD- cell still takes fatal damage based on total toxicity, 
        # but because it cannot undergo apoptosis, all death is routed via necrosis.
        total_rate = rate_apop + rate_necro
        
        if total_rate > 0
            prob_death = 1.0 - exp(-total_rate * TIME_STEP_DT)
            if rand() < prob_death
                agent.alive = false
                agent.dead_necrosis = true
                
                # Necrotic Lysis: Instantly dump internal nutrients and bound antifungal into environment
                model.nutrient_layer[agent.pos[2], agent.pos[1]] += agent.internal_nutrients
                agent.internal_nutrients = 0.0
                
                model.ANTIFUNGAL_layer[agent.pos[2], agent.pos[1]] += agent.bound_ANTIFUNGAL
                agent.bound_ANTIFUNGAL = 0.0
            end
        end
    end
end

# ==========================================
# --- CORE MODEL STEP (Execution Logic) ---
# ==========================================

function complex_model_step!(model)
    
    n_demands = Dict{Int, Float64}()
    f_demands = Dict{Int, Float64}()
    f_releases = Dict{Int, Float64}()

    # --- PASS 1: Calculate Demands ---
    for agent in allagents(model)
        x, y = agent.pos
        
        # 1. Nutrient Demands 
        if agent.alive
            local_n = model.nutrient_layer[y, x]
            local_f = model.ANTIFUNGAL_layer[y, x]
            max_reservoir = agent.biomass * RESERVOIR_FRACTION
            reservoir_deficit = max(0.0, max_reservoir - agent.internal_nutrients)

            if is_apoptotic(agent)
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                n_demands[agent.id] = maintenance_cost + reservoir_deficit
            else
                # --- NEW: FUNGISTATIC / ZOMBIE STATE ---
                if local_f >= FUNGISTATIC_THRESHOLD
                    growth_demand_n = 0.0 # Growth is completely halted
                    reservoir_deficit = 0.0 # Zombie cells stop active stockpiling; they only scavenge for immediate maintenance
                else
                    if agent.biomass < DIVISION_BIOMASS
                        mu = MU_MAX * (local_n / (MONOD_KS + local_n))
                        max_possible_growth_biomass = min(mu * agent.biomass * TIME_STEP_DT, DIVISION_BIOMASS - agent.biomass)
                        growth_demand_n = max_possible_growth_biomass / YIELD_TRUE
                    else
                        growth_demand_n = 0.0
                    end
                end
                
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                n_demands[agent.id] = growth_demand_n + maintenance_cost + reservoir_deficit
            end
        end
        
        # 2. Antifungal Reversible Binding
        local_f = model.ANTIFUNGAL_layer[y, x]
        bound_f = agent.bound_ANTIFUNGAL
        
        current_max = if agent.alive
            MAX_ANTIFUNGAL_BINDING_LIVE
        elseif is_dead_apoptosis(agent)
            MAX_ANTIFUNGAL_BINDING_DEAD_APOP
        else
            MAX_ANTIFUNGAL_BINDING_DEAD_NECRO
        end

        cap_remaining = max(0.0, current_max - bound_f)

        net_change = (K_ON_ANTIFUNGAL * local_f * cap_remaining - K_OFF_ANTIFUNGAL * bound_f) * TIME_STEP_DT

        if net_change > 0
            f_demands[agent.id] = net_change
        else
            f_releases[agent.id] = min(-net_change, bound_f)
        end
    end

    newborn_spots = Tuple{Tuple{Int,Int}, Float64, Float64}[] 
    planned_biomass = Dict{Tuple{Int,Int}, Float64}()

    # --- PASS 2: Allocate & Update ---
    randomized_agents = shuffle!(collect(allagents(model)))

    for agent in randomized_agents
        x, y = agent.pos
        
        if agent.alive
            apply_stress!(agent, model.ANTIFUNGAL_layer[y, x], model)
        end

        if agent.alive && get(n_demands, agent.id, 0.0) > 0
            demand = n_demands[agent.id]
            
            actual_intake = min(demand, model.nutrient_layer[y, x])
            model.nutrient_layer[y, x] -= actual_intake
            agent.internal_nutrients += actual_intake

            maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

            if is_apoptotic(agent)
                burn = min(agent.internal_nutrients, maintenance_cost)
                agent.internal_nutrients -= burn
                model.total_lost_nutrients += burn
            else
                if agent.internal_nutrients >= maintenance_cost
                    agent.internal_nutrients -= maintenance_cost
                    model.total_lost_nutrients += maintenance_cost

                    # Only grow if there's no fungistatic arrest
                    local_f = model.ANTIFUNGAL_layer[y, x]
                    if local_f < FUNGISTATIC_THRESHOLD && agent.biomass < DIVISION_BIOMASS
                        max_growth_biomass = min(MU_MAX * agent.biomass * TIME_STEP_DT, DIVISION_BIOMASS - agent.biomass)
                        max_growth_n = max_growth_biomass / YIELD_TRUE
                        
                        actual_growth_n = min(agent.internal_nutrients, max_growth_n)
                        actual_growth_biomass = actual_growth_n * YIELD_TRUE
                        
                        agent.biomass += actual_growth_biomass
                        agent.internal_nutrients -= actual_growth_n
                    end
                else
                    remaining_deficit = maintenance_cost - agent.internal_nutrients
                    model.total_lost_nutrients += agent.internal_nutrients
                    agent.internal_nutrients = 0.0
                    
                    biomass_burned = remaining_deficit * YIELD_TRUE
                    agent.biomass -= biomass_burned
                    model.total_lost_nutrients += remaining_deficit
                    
                    if agent.biomass <= STARVATION_BIOMASS
                        agent.alive = false
                        agent.dead_starvation = true
                        
                        # Starvation Lysis: Instantly dump bound antifungal into environment
                        model.ANTIFUNGAL_layer[y, x] += agent.bound_ANTIFUNGAL
                        agent.bound_ANTIFUNGAL = 0.0
                    end
                end

                if agent.alive
                    max_reservoir = agent.biomass * RESERVOIR_FRACTION
                    if agent.internal_nutrients > max_reservoir
                        excess = agent.internal_nutrients - max_reservoir
                        agent.internal_nutrients = max_reservoir
                        model.nutrient_layer[y, x] += excess
                    end

                    if agent.biomass >= DIVISION_BIOMASS
                        if !can_divide(agent)
                            excess = agent.biomass - DIVISION_BIOMASS
                            agent.biomass = DIVISION_BIOMASS
                            model.total_lost_nutrients += (excess / YIELD_TRUE) 
                        else
                            chosen_spot = nothing
                            immediate_hood = [agent.pos; collect(nearby_positions(agent.pos, model, 1))]
                            
                            # Assess biomass in immediate neighborhood
                            spot_status = map(immediate_hood) do p
                                cur_b = sum((a.biomass for a in agents_in_position(p, model)), init=0.0)
                                total_b = cur_b + get(planned_biomass, p, 0.0)
                                (pos = p, bio = total_b)
                            end
                            
                            # Filter to spots that have enough capacity for the newborn
                            available_spots = filter(s -> s.bio + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX, spot_status)
                            
                            if !isempty(available_spots)
                                truly_empty = filter(s -> s.bio == 0.0, available_spots)
                                candidate_pool = !isempty(truly_empty) ? truly_empty : available_spots
                                
                                # --- FIX: Surface Tension Weighting ---
                                weights = map(candidate_pool) do s
                                    occupied_neighbors = count(nearby_positions(s.pos, model, 1)) do np
                                        cur_b = sum((a.biomass for a in agents_in_position(np, model)), init=0.0)
                                        (cur_b + get(planned_biomass, np, 0.0)) > 0.0
                                    end
                                    return Float64(occupied_neighbors)^3 + 1.0 
                                end
                                
                                chosen_idx = sample(1:length(candidate_pool), Weights(weights))
                                chosen_spot = candidate_pool[chosen_idx].pos
                                
                            elseif rand() < PUSH_PROBABILITY
                                block = collect(nearby_positions(agent.pos, model, MAX_PUSH_RADIUS))
                                empty_spots = filter(p -> begin
                                    cur_b = sum((a.biomass for a in agents_in_position(p, model)), init=0.0)
                                    cur_b + get(planned_biomass, p, 0.0) + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX
                                end, block)
                                
                                if !isempty(empty_spots)
                                    min_dist = minimum((x - p[1])^2 + (y - p[2])^2 for p in empty_spots)
                                    best_spots = filter(p -> (x - p[1])^2 + (y - p[2])^2 <= min_dist + 2, empty_spots)
                                    
                                    weights = map(best_spots) do p
                                        cur_b = sum((a.biomass for a in agents_in_position(p, model)), init=0.0)
                                        MAX_BIOMASS_PER_PX - (cur_b + get(planned_biomass, p, 0.0))
                                    end
                                    chosen_spot = sample(best_spots, Weights(weights))
                                end
                            end

                            if chosen_spot !== nothing
                                agent.biomass -= NEWBORN_BIOMASS 
                                
                                daughter_fraction = NEWBORN_BIOMASS / (agent.biomass + NEWBORN_BIOMASS)
                                daughter_n = agent.internal_nutrients * daughter_fraction
                                agent.internal_nutrients -= daughter_n

                                planned_biomass[chosen_spot] = get(planned_biomass, chosen_spot, 0.0) + NEWBORN_BIOMASS
                                push!(newborn_spots, (chosen_spot, NEWBORN_BIOMASS, daughter_n))
                            else
                                excess = agent.biomass - DIVISION_BIOMASS
                                agent.biomass = DIVISION_BIOMASS
                                model.total_lost_nutrients += (excess / YIELD_TRUE)
                            end
                        end
                    end
                end
            end
        end

        # 3. Antifungal Binding
        if get(f_demands, agent.id, 0.0) > 0
            demand = f_demands[agent.id]
            actual_binding = min(demand, model.ANTIFUNGAL_layer[y, x])
            agent.bound_ANTIFUNGAL += actual_binding
            model.ANTIFUNGAL_layer[y, x] -= actual_binding
        elseif get(f_releases, agent.id, 0.0) > 0
            release = f_releases[agent.id]
            agent.bound_ANTIFUNGAL -= release
            model.ANTIFUNGAL_layer[y, x] += release
        end
    end

    # Spawning newborns
    for (pos, b, n) in newborn_spots
        cur_b = sum((a.biomass for a in agents_in_position(pos, model)), init=0.0)
        
        if cur_b + b <= MAX_BIOMASS_PER_PX + 0.01 
            if model.is_pcd_plus
                add_agent!(pos, PCDPlusCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, b, n, true)
            else
                add_agent!(pos, PCDMinusCell, model, true, 0.0, false, false, 0.0, b, n)
            end
        else
            model.total_lost_nutrients += (b / YIELD_TRUE) + n
        end
    end

    # --- PDE Diffusion ---
    alpha_n = DIFFUSION_NUTRIENT * TIME_STEP_DT
    alpha_f = DIFFUSION_ANTIFUNGAL * TIME_STEP_DT

    rhs_n = model.nutrient_layer .+ (alpha_n / 2.0) .* imfilter(model.nutrient_layer, centered(model.laplacian_kernel), "replicate")
    rhs_f = model.ANTIFUNGAL_layer .+ (alpha_f / 2.0) .* imfilter(model.ANTIFUNGAL_layer, centered(model.laplacian_kernel), "replicate")

    u_n = copy(model.nutrient_layer)
    u_f = copy(model.ANTIFUNGAL_layer)

    denom_n = 1.0 + (5.0 / 3.0) * alpha_n
    denom_f = 1.0 + (5.0 / 3.0) * alpha_f

    for _ in 1:DIFFUSION_ITERATIONS
        u_n = (rhs_n .+ (alpha_n / 2.0) .* imfilter(u_n, centered(model.neighbor_kernel), "replicate")) ./ denom_n
        u_f = (rhs_f .+ (alpha_f / 2.0) .* imfilter(u_f, centered(model.neighbor_kernel), "replicate")) ./ denom_f
    end

    model.nutrient_layer .= max.(u_n, 0.0)
    model.ANTIFUNGAL_layer .= max.(u_f, 0.0)
end

# ==========================================
# --- UTILITY & MASS BALANCES ---
# ==========================================

function get_antifungal_breakdown(model)
    env_mass = sum(model.ANTIFUNGAL_layer)
    bound_live = sum((a.bound_ANTIFUNGAL for a in allagents(model) if a.alive), init=0.0)
    bound_dead = sum((a.bound_ANTIFUNGAL for a in allagents(model) if !a.alive), init=0.0)
    total = env_mass + bound_live + bound_dead
    return (env=env_mass, live=bound_live, dead=bound_dead, total=total)
end

function get_nutrient_breakdown(model)
    env_mass = sum(model.nutrient_layer)
    internal = sum((a.internal_nutrients for a in allagents(model)), init=0.0)
    biomass_eq = sum((a.biomass / YIELD_TRUE for a in allagents(model)), init=0.0)
    burned = model.total_lost_nutrients
    total = env_mass + internal + biomass_eq + burned
    return (env=env_mass, internal=internal, biomass=biomass_eq, burned=burned, total=total)
end

function generate_subplots(model, title_prefix::String, step::Int)
    alive_agents = [a for a in allagents(model) if a.alive]
    
    # Tiny random jitter added to coordinates strictly for plotting, so overlapping cells at the same grid px are visibly distinguished
    jitter(val) = val + 0.4 * (rand() - 0.5)

    healthy_x = Float64[jitter(a.pos[1]) for a in alive_agents if !is_apoptotic(a)]
    healthy_y = Float64[jitter(a.pos[2]) for a in alive_agents if !is_apoptotic(a)]

    apoptotic_x = Float64[jitter(a.pos[1]) for a in alive_agents if is_apoptotic(a)]
    apoptotic_y = Float64[jitter(a.pos[2]) for a in alive_agents if is_apoptotic(a)]

    dead_apoptosis_x = Float64[jitter(a.pos[1]) for a in allagents(model) if is_dead_apoptosis(a)]
    dead_apoptosis_y = Float64[jitter(a.pos[2]) for a in allagents(model) if is_dead_apoptosis(a)]

    dead_necrosis_x = Float64[jitter(a.pos[1]) for a in allagents(model) if a.dead_necrosis]
    dead_necrosis_y = Float64[jitter(a.pos[2]) for a in allagents(model) if a.dead_necrosis]

    dead_starved_x = Float64[jitter(a.pos[1]) for a in allagents(model) if a.dead_starvation]
    dead_starved_y = Float64[jitter(a.pos[2]) for a in allagents(model) if a.dead_starvation]

    num_live = length(alive_agents)
    
    n_plot = copy(model.nutrient_layer)
    n_plot[1, 1] = INIT_NUTRIENT_LEVEL; n_plot[1, 2] = 0.0

    f_plot = copy(model.ANTIFUNGAL_layer)
    # Dynamically adjust the clims so point-sources with high doses don't just wash out the color map
    f_max = maximum(model.ANTIFUNGAL_layer)
    f_clim_max = f_max > 0 ? max(INIT_ANTIFUNGAL_LEVEL, f_max) : INIT_ANTIFUNGAL_LEVEL

    p1 = Plots.heatmap(n_plot, title="$title_prefix Nutrients (Step $step)", 
                 color=:viridis, clims=(0, INIT_NUTRIENT_LEVEL), aspect_ratio=:equal)

    p2 = Plots.heatmap(f_plot, title="$title_prefix Antifungal", 
                 color=:ice, clims=(0, f_clim_max), aspect_ratio=:equal)

    p3 = Plots.plot(title="$title_prefix Live Cells: $num_live", 
              xlims=(1, GRID_SIZE_PX), ylims=(1, GRID_SIZE_PX), 
              aspect_ratio=:equal, legend=:topright)

    healthy_color = model.is_pcd_plus ? :blue : :red
    healthy_label = model.is_pcd_plus ? "PCD+" : "PCD-"

    # Plot dead cells first so they are rendered at the bottom
    if !isempty(dead_starved_x)
        Plots.scatter!(p3, dead_starved_x, dead_starved_y, label="Dead (Starve)", color=:magenta, markersize=3, markerstrokewidth=0)
    end
    if !isempty(dead_necrosis_x)
        Plots.scatter!(p3, dead_necrosis_x, dead_necrosis_y, label="Dead (Necro)", color=:black, markersize=3, markerstrokewidth=0)
    end
    if !isempty(dead_apoptosis_x)
        Plots.scatter!(p3, dead_apoptosis_x, dead_apoptosis_y, label="Dead (Apop)", color=:darkgray, markersize=3, markerstrokewidth=0)
    end
    
    # Plot alive/apoptotic cells last so they are rendered on top
    if !isempty(apoptotic_x)
        Plots.scatter!(p3, apoptotic_x, apoptotic_y, label="Apoptotic", color=:orange, markersize=2.5, markerstrokewidth=0)
    end
    if !isempty(healthy_x)
        Plots.scatter!(p3, healthy_x, healthy_y, label=healthy_label, color=healthy_color, markersize=2, markerstrokewidth=0)
    end

    # --- NEW: Biomass Density Plot ---
    # We create a 2D matrix summing the biomass of all agents in each pixel.
    density_grid = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    for a in allagents(model)
        x, y = a.pos
        density_grid[y, x] += a.biomass
    end

    # Map colors: 0=White, 1=Green, 2=Yellow, 3=Orange
    density_colors = cgrad([:white, :green, :yellow, :orange], [0.0, 0.33, 0.66, 1.0])
    
    p4 = Plots.heatmap(density_grid, title="$title_prefix Biomass Density", 
                 color=density_colors, clims=(0, MAX_BIOMASS_PER_PX), aspect_ratio=:equal, 
                 colorbar_ticks=(0:MAX_BIOMASS_PER_PX, string.(0:MAX_BIOMASS_PER_PX)))

    return p1, p2, p3, p4
end

# ==========================================
# --- RUN SIMULATION ---
# ==========================================

function main()
    println("Initializing strictly typed Agents.jl models (Capacity Version)...")
    
    cx = (GRID_SIZE_PX + 1) / 2.0
    cy = (GRID_SIZE_PX + 1) / 2.0
    
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    
    starting_positions = all_pos[1:min(INITIAL_CELLS, length(all_pos))]
    
    model_plus = initialize_model(PCDPlusCell, starting_positions)
    model_minus = initialize_model(PCDMinusCell, starting_positions)
    
    every_n_steps = 6 
    fps = 10 

    history_plus = Dict(:alive => Int[], :dead_apop => Int[], :dead_necro => Int[], :dead_starve => Int[], :total => Int[])
    history_minus = Dict(:alive => Int[], :dead_apop => Int[], :dead_necro => Int[], :dead_starve => Int[], :total => Int[])

    # Push Step 0 data
    push!(history_plus[:alive], count(a -> a.alive, allagents(model_plus)))
    push!(history_plus[:dead_apop], count(is_dead_apoptosis, allagents(model_plus)))
    push!(history_plus[:dead_necro], count(a -> a.dead_necrosis, allagents(model_plus)))
    push!(history_plus[:dead_starve], count(a -> a.dead_starvation, allagents(model_plus)))
    push!(history_plus[:total], nagents(model_plus))

    push!(history_minus[:alive], count(a -> a.alive, allagents(model_minus)))
    push!(history_minus[:dead_apop], count(is_dead_apoptosis, allagents(model_minus)))
    push!(history_minus[:dead_necro], count(a -> a.dead_necrosis, allagents(model_minus)))
    push!(history_minus[:dead_starve], count(a -> a.dead_starvation, allagents(model_minus)))
    push!(history_minus[:total], nagents(model_minus))

    # --- Initialize Mass Tracking ---
    # Accounts for the agar grid + the initial biomass and internal reservoirs of the starting cells
    injected_n_plus = (GRID_SIZE_PX * GRID_SIZE_PX * INIT_NUTRIENT_LEVEL) + (NEWBORN_BIOMASS / YIELD_TRUE) + (NEWBORN_BIOMASS * RESERVOIR_FRACTION)
    injected_n_minus = injected_n_plus
    injected_f_plus = 0.0
    injected_f_minus = 0.0

    println("Starting dual simulation and recording frames...")
    
    anim = Plots.@animate for step in 1:SIMULATION_STEPS
        
        # --- Determine if nutrient injection should happen this step ---
        inject_nutrient_now = false
        if NUTRIENT_EXPOSURE_MODE == SINGLE_SHOCK
            inject_nutrient_now = (step == NUTRIENT_INJECTION_STEP)
        elseif NUTRIENT_EXPOSURE_MODE == CONTINUOUS
            inject_nutrient_now = (step >= NUTRIENT_INJECTION_STEP)
        elseif NUTRIENT_EXPOSURE_MODE == PULSATED
            inject_nutrient_now = (step >= NUTRIENT_INJECTION_STEP) && ((step - NUTRIENT_INJECTION_STEP) % NUTRIENT_PULSE_INTERVAL == 0)
        end

        # --- Determine if antifungal injection should happen this step ---
        inject_antifungal_now = false
        if ANTIFUNGAL_EXPOSURE_MODE == SINGLE_SHOCK
            inject_antifungal_now = (step == ANTIFUNGAL_INJECTION_STEP)
        elseif ANTIFUNGAL_EXPOSURE_MODE == CONTINUOUS
            inject_antifungal_now = (step >= ANTIFUNGAL_INJECTION_STEP)
        elseif ANTIFUNGAL_EXPOSURE_MODE == PULSATED
            inject_antifungal_now = (step >= ANTIFUNGAL_INJECTION_STEP) && ((step - ANTIFUNGAL_INJECTION_STEP) % ANTIFUNGAL_PULSE_INTERVAL == 0)
        end

        # --- Apply the injections ---
        if inject_nutrient_now
            # Only print the message for discrete events to avoid console spam during CONTINUOUS
            if step == NUTRIENT_INJECTION_STEP || NUTRIENT_EXPOSURE_MODE == PULSATED
                println("--- REPLENISHING NUTRIENTS AT STEP $step ---")
            end
            
            # Track net newly added mass (target layer minus current layer)
            injected_n_plus += sum(INIT_NUTRIENT_LEVEL .- model_plus.nutrient_layer)
            injected_n_minus += sum(INIT_NUTRIENT_LEVEL .- model_minus.nutrient_layer)
            
            model_plus.nutrient_layer .= INIT_NUTRIENT_LEVEL
            model_minus.nutrient_layer .= INIT_NUTRIENT_LEVEL
        end

        if inject_antifungal_now
            if step == ANTIFUNGAL_INJECTION_STEP || ANTIFUNGAL_EXPOSURE_MODE == PULSATED
                println("--- INJECTING ANTIFUNGAL AT STEP $step ---")
            end
            
            if ANTIFUNGAL_SPATIAL_MODE == UNIFORM
                # Original logic: flood the entire grid uniformly
                injected_f_plus += sum(INIT_ANTIFUNGAL_LEVEL .- model_plus.ANTIFUNGAL_layer)
                injected_f_minus += sum(INIT_ANTIFUNGAL_LEVEL .- model_minus.ANTIFUNGAL_layer)
                
                model_plus.ANTIFUNGAL_layer .= INIT_ANTIFUNGAL_LEVEL
                model_minus.ANTIFUNGAL_layer .= INIT_ANTIFUNGAL_LEVEL
            elseif ANTIFUNGAL_SPATIAL_MODE == POINT_SOURCES
                # New logic: Inject concentrated dose at specific coordinates with a specified radius
                for (cx, cy) in ANTIFUNGAL_SOURCES
                    for dx in -ANTIFUNGAL_SOURCE_RADIUS:ANTIFUNGAL_SOURCE_RADIUS
                        for dy in -ANTIFUNGAL_SOURCE_RADIUS:ANTIFUNGAL_SOURCE_RADIUS
                            sx, sy = cx + dx, cy + dy
                            
                            if 1 <= sx <= GRID_SIZE_PX && 1 <= sy <= GRID_SIZE_PX
                                model_plus.ANTIFUNGAL_layer[sy, sx] += ANTIFUNGAL_SOURCE_DOSE
                                model_minus.ANTIFUNGAL_layer[sy, sx] += ANTIFUNGAL_SOURCE_DOSE
                                
                                # Accurately track the mass added
                                injected_f_plus += ANTIFUNGAL_SOURCE_DOSE
                                injected_f_minus += ANTIFUNGAL_SOURCE_DOSE
                            end
                        end
                    end
                end
            end
        end

        Agents.step!(model_plus, 1)
        Agents.step!(model_minus, 1)

        # Record metrics for Plus
        push!(history_plus[:alive], count(a -> a.alive, allagents(model_plus)))
        push!(history_plus[:dead_apop], count(is_dead_apoptosis, allagents(model_plus)))
        push!(history_plus[:dead_necro], count(a -> a.dead_necrosis, allagents(model_plus)))
        push!(history_plus[:dead_starve], count(a -> a.dead_starvation, allagents(model_plus)))
        push!(history_plus[:total], nagents(model_plus))

        # Record metrics for Minus
        push!(history_minus[:alive], count(a -> a.alive, allagents(model_minus)))
        push!(history_minus[:dead_apop], count(is_dead_apoptosis, allagents(model_minus)))
        push!(history_minus[:dead_necro], count(a -> a.dead_necrosis, allagents(model_minus)))
        push!(history_minus[:dead_starve], count(a -> a.dead_starvation, allagents(model_minus)))
        push!(history_minus[:total], nagents(model_minus))

        p1_n, p1_f, p1_c, p1_p = generate_subplots(model_plus, "PCD+", step)
        p2_n, p2_f, p2_c, p2_p = generate_subplots(model_minus, "PCD-", step)

        if step % 10 == 0
            println("Progress: Step $step / $SIMULATION_STEPS")
        end

        # Plot must be the last evaluated expression for the @animate macro
        Plots.plot(p1_n, p1_f, p1_c, p1_p, p2_n, p2_f, p2_c, p2_p, layout=(2, 4), size=(1600, 800))
    end every every_n_steps  

    gif_name = "dual_petri_dish_simulation.gif"
    Plots.gif(anim, gif_name, fps = fps)
    println("Success! GIF saved as: ", gif_name)

    # ==========================================
    # --- CALCULATE EXPERIMENTAL DOUBLING TIME ---
    # ==========================================
    # Calculate based on the unhindered growth phase before antifungal injection
    t_phase_hours = ANTIFUNGAL_INJECTION_STEP * TIME_STEP_DT
    
    n0_plus = history_plus[:alive][1]
    nt_plus = history_plus[:alive][ANTIFUNGAL_INJECTION_STEP + 1] # +1 because array includes Step 0
    if nt_plus > n0_plus
        td_plus = t_phase_hours * log(2) / log(nt_plus / n0_plus)
        println("=> PCD+ Estimated Doubling Time (pre-injection): ", round(td_plus, digits=2), " hours")
    else
        println("=> PCD+ Estimated Doubling Time: N/A (no net growth)")
    end

    n0_minus = history_minus[:alive][1]
    nt_minus = history_minus[:alive][ANTIFUNGAL_INJECTION_STEP + 1]
    if nt_minus > n0_minus
        td_minus = t_phase_hours * log(2) / log(nt_minus / n0_minus)
        println("=> PCD- Estimated Doubling Time (pre-injection): ", round(td_minus, digits=2), " hours")
    else
        println("=> PCD- Estimated Doubling Time: N/A (no net growth)")
    end

    # ==========================================
    # --- MASS BALANCE REPORT ---
    # ==========================================
    println("\n=================================================")
    println("FINAL MASS BALANCE REPORT (Step $SIMULATION_STEPS)")
    println("=================================================")
    
    n_plus = get_nutrient_breakdown(model_plus)
    f_plus = get_antifungal_breakdown(model_plus)
    
    n_minus = get_nutrient_breakdown(model_minus)
    f_minus = get_antifungal_breakdown(model_minus)

    @printf("PCD+ NUTRIENTS:\n")
    @printf("  Initial/Injected:  %.2f\n", injected_n_plus)
    @printf("  Final Total:       %.2f\n", n_plus.total)
    @printf("    -> Environment:  %.2f\n", n_plus.env)
    @printf("    -> Biomass (Eq): %.2f\n", n_plus.biomass)
    @printf("    -> Reservoir:    %.2f\n", n_plus.internal)
    @printf("    -> Burned/Lost:  %.2f\n\n", n_plus.burned)

    @printf("PCD+ ANTIFUNGAL:\n")
    @printf("  Initial/Injected:  %.2f\n", injected_f_plus)
    @printf("  Final Total:       %.2f\n", f_plus.total)
    @printf("    -> Environment:  %.2f\n", f_plus.env)
    @printf("    -> Bound(Alive): %.2f\n", f_plus.live)
    @printf("    -> Bound(Dead):  %.2f\n\n", f_plus.dead)

    @printf("PCD- NUTRIENTS:\n")
    @printf("  Initial/Injected:  %.2f\n", injected_n_minus)
    @printf("  Final Total:       %.2f\n", n_minus.total)
    @printf("    -> Environment:  %.2f\n", n_minus.env)
    @printf("    -> Biomass (Eq): %.2f\n", n_minus.biomass)
    @printf("    -> Reservoir:    %.2f\n", n_minus.internal)
    @printf("    -> Burned/Lost:  %.2f\n\n", n_minus.burned)

    @printf("PCD- ANTIFUNGAL:\n")
    @printf("  Initial/Injected:  %.2f\n", injected_f_minus)
    @printf("  Final Total:       %.2f\n", f_minus.total)
    @printf("    -> Environment:  %.2f\n", f_minus.env)
    @printf("    -> Bound(Alive): %.2f\n", f_minus.live)
    @printf("    -> Bound(Dead):  %.2f\n\n", f_minus.dead)

  # ==========================================
    # --- GENERATE REQUESTED DYNAMICS PLOTS ---
    # ==========================================
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT

    # Plot 1: Population (Alive)
    p_pop = Plots.plot(title="Population (Alive) Over Time", xlabel="Time (hrs)", ylabel="Cells", linewidth=2)
    Plots.plot!(p_pop, time_axis, history_plus[:alive], label="PCD+ Alive", color=:blue)
    Plots.plot!(p_pop, time_axis, history_minus[:alive], label="PCD- Alive", color=:red)

    # Plot 2: Dead Cells On Grid (Apop, Necro, and Starved)
    p_death = Plots.plot(title="Dead Cells On Grid Over Time", xlabel="Time (hrs)", ylabel="Dead Cells", linewidth=2)
    Plots.plot!(p_death, time_axis, history_plus[:dead_apop], label="PCD+ Apop", color=:orange)
    Plots.plot!(p_death, time_axis, history_plus[:dead_necro], label="PCD+ Necro", color=:black)
    Plots.plot!(p_death, time_axis, history_minus[:dead_necro], label="PCD- Necro", color=:gray, linestyle=:dash)
    Plots.plot!(p_death, time_axis, history_plus[:dead_starve], label="PCD+ Starved", color=:magenta)
    Plots.plot!(p_death, time_axis, history_minus[:dead_starve], label="PCD- Starved", color=:purple, linestyle=:dash)

    # Plot 3: Survivability (% Alive)
    surv_plus = [tot > 0 ? (a / tot) * 100.0 : 0.0 for (a, tot) in zip(history_plus[:alive], history_plus[:total])]
    surv_minus = [tot > 0 ? (a / tot) * 100.0 : 0.0 for (a, tot) in zip(history_minus[:alive], history_minus[:total])]
    
    p_surv = Plots.plot(title="Survivability (% Alive) Over Time", xlabel="Time (hrs)", ylabel="Survival (%)", linewidth=2, ylims=(0, 105))
    Plots.plot!(p_surv, time_axis, surv_plus, label="PCD+", color=:blue)
    Plots.plot!(p_surv, time_axis, surv_minus, label="PCD-", color=:red)

    # Combine and save
    final_plot = Plots.plot(p_pop, p_death, p_surv, layout=(3, 1), size=(800, 1000))
    Plots.savefig(final_plot, "population_dynamics.png")
    println("Success! Dynamics plots saved as: population_dynamics.png")
end

main()