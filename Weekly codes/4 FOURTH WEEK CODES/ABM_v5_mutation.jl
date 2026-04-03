using Agents
using Random
using StatsBase
using Images
using ImageFiltering
using Plots

# ==========================================
# --- MUTATION & ENVIRONMENT SETTINGS ---
# ==========================================
const START_AS_PCD_PLUS = false       # True: Population starts as PCD+. False: Starts as PCD-.
const MUTATION_STEP = 50              # Step at which mutation is introduced
const MUTATION_LOCATION = :edge     # Options: :center, :edge, :random
const MUTATION_COUNT = 1              # Number of cells that mutate when the step is reached

# ==========================================
# --- GLOBAL EXPERIMENTAL PARAMETERS ---
# ==========================================

const GRID_SIZE_PX = 100 # 1 px = 5 µm, making the physical grid 500 µm across
const INITIAL_CELLS = 100
const SIMULATION_STEPS = 450
const ANTIFUNGAL_INJECTION_STEP = 100 # Inject halfway 
const TIME_STEP_DT = 1.0 / 3.0 # Assuming 1 step = 20 minutes of biological time

# --- Environment Levels ---
const INIT_NUTRIENT_LEVEL = 7
const INIT_ANTIFUNGAL_LEVEL = 3.05

# --- Diffusion Settings ---
const DIFFUSION_NUTRIENT = 0.10
const DIFFUSION_ANTIFUNGAL = 0.08
const DIFFUSION_ITERATIONS = 15 # Jacobi iterations for the implicit solver

# --- Growth & Metabolism ---
const MU_MAX = 0.34            # Maximum specific growth rate (mu_max)
const MONOD_KS = 5.0           # Half-velocity constant for Monod kinetics
const MAINTENANCE_COEFF = 0.015 # Maintenance coefficient (m)
const YIELD_TRUE = 0.39        # True growth yield (Y)
const NEWBORN_BIOMASS = 1.0    # Biomass of a daughter cell upon bud detachment
const DIVISION_BIOMASS = 2.0   # Threshold to detach a bud
const STARVATION_BIOMASS = 0.5 # Cells die if they shrink to this biomass level due to starvation
const RESERVOIR_FRACTION = 0.05 # Max internal nutrients as a fraction of current biomass

# --- Mechanics & Space ---
const PUSH_PROBABILITY = 0.1 # Probability to mechanically push when locally trapped
const MAX_PUSH_RADIUS = 10    # Max radius a cell can shove others to divide

# --- Antifungal Binding Kinetics ---
const MAX_ANTIFUNGAL_BINDING_LIVE = 0.42  # Capacity for intact, living cells
const MAX_ANTIFUNGAL_BINDING_DEAD = 2.55  # Capacity for dead cells (sponge effect)
const K_ON_ANTIFUNGAL = 0.01              # Adsorption rate constant
const K_OFF_ANTIFUNGAL = 0.005            # Desorption rate constant

# --- Stress, Apoptosis & Necrosis ---
const ANTIFUNGAL_DAMAGE_THRESHOLD = 3      # Threshold to start accumulating damage
const STRESS_START_TIME = 3.30               # Hours of exposure before death risks begin
const APOPTOSIS_DURATION = 2.0               # Hours the apoptosis process takes
const ASSAY_DURATION_HOURS = 200/60            # Calibration time for dose-response percentages

# ==========================================
# --- AGENT TYPES ---
# ==========================================

@agent struct YeastCell(GridAgent{2})
    is_pcd_plus::Bool       # The genetic trait (true = PCD+, false = PCD-)
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

# ==========================================
# --- MODEL PROPERTIES & INITIALIZATION ---
# ==========================================

mutable struct PetriDishProperties
    nutrient_layer::Matrix{Float64}
    ANTIFUNGAL_layer::Matrix{Float64}
    laplacian_kernel::Matrix{Float64}
    neighbor_kernel::Matrix{Float64}
    total_lost_nutrients::Float64 # Tracks respiration, inefficiency, and trapped biomass losses
end

function initialize_model(starting_positions::Vector{Tuple{Int, Int}})
    space = GridSpaceSingle((GRID_SIZE_PX, GRID_SIZE_PX); periodic=false)

    props = PetriDishProperties(
        fill(INIT_NUTRIENT_LEVEL, GRID_SIZE_PX, GRID_SIZE_PX),
        fill(0.0, GRID_SIZE_PX, GRID_SIZE_PX), # Start with 0.0 antifungal layer
        [1/6 2/3 1/6; 2/3 -10/3 2/3; 1/6 2/3 1/6],
        [1/6 2/3 1/6; 2/3 0.0 2/3; 1/6 2/3 1/6],
        0.0 # Initial lost nutrients
    )

    model = StandardABM(YeastCell, space; properties=props, model_step! = complex_model_step!)

    init_internal = NEWBORN_BIOMASS * RESERVOIR_FRACTION

    for pos in starting_positions
        add_agent!(pos, YeastCell, model, START_AS_PCD_PLUS, true, false, 0.0, 0.0, false, false, false, 0.0, NEWBORN_BIOMASS, init_internal, true)
    end

    return model
end

# ==========================================
# --- MUTATION LOGIC ---
# ==========================================

function inject_mutation!(model, location::Symbol, num_mutants::Int)
    alive_agents = [a for a in allagents(model) if a.alive]
    if isempty(alive_agents)
        println("No alive agents available to mutate!")
        return
    end

    cx, cy = (GRID_SIZE_PX + 1) / 2.0, (GRID_SIZE_PX + 1) / 2.0

    if location == :random
        targets = sample(alive_agents, min(num_mutants, length(alive_agents)), replace=false)
    elseif location == :center
        sort!(alive_agents, by = a -> (a.pos[1] - cx)^2 + (a.pos[2] - cy)^2)
        targets = alive_agents[1:min(num_mutants, length(alive_agents))]
    elseif location == :edge
        sort!(alive_agents, by = a -> (a.pos[1] - cx)^2 + (a.pos[2] - cy)^2, rev=true)
        targets = alive_agents[1:min(num_mutants, length(alive_agents))]
    else
        error("Unknown mutation location: $location")
    end

    for target in targets
        old_trait = target.is_pcd_plus
        target.is_pcd_plus = !old_trait 
        
        # If mutating from PCD+ to PCD- while undergoing apoptosis, reset the state
        if !target.is_pcd_plus && target.is_apoptotic
            target.is_apoptotic = false
            target.apoptosis_timer = 0.0
        end
        
        println("  -> Mutated cell at $(target.pos) from PCD$(old_trait ? "+" : "-") to PCD$(target.is_pcd_plus ? "+" : "-")")
    end
end

# ==========================================
# --- ANTIFUNGAL EXPOSURE LOGIC ---
# ==========================================

function get_death_rates(c::Float64)
    if c <= 0.0; return 0.0, 0.0; end
    
    C_vals = (0.0, 4.0, 8.0, 16.0)
    apop_vals = (0.0, 0.20, 0.57, 0.09)
    necro_vals = (0.0, 0.20, 0.33, 0.91)

    target_apop_frac, target_necro_frac = 0.0, 0.0

    if c >= C_vals[end]
        target_apop_frac = apop_vals[end]
        target_necro_frac = necro_vals[end]
    else
        for i in 1:(length(C_vals)-1)
            if c >= C_vals[i] && c <= C_vals[i+1]
                t = (c - C_vals[i]) / (C_vals[i+1] - C_vals[i])
                target_apop_frac = apop_vals[i] + t * (apop_vals[i+1] - apop_vals[i])
                target_necro_frac = necro_vals[i] + t * (necro_vals[i+1] - necro_vals[i])
                break
            end
        end
    end
    
    target_total_frac = min(0.999, target_apop_frac + target_necro_frac)
    if target_total_frac <= 0.0; return 0.0, 0.0; end
    
    hourly_total_rate = -log(1.0 - target_total_frac) / ASSAY_DURATION_HOURS
    ratio_apop = target_apop_frac / target_total_frac
    ratio_necro = target_necro_frac / target_total_frac
    
    return hourly_total_rate * ratio_apop, hourly_total_rate * ratio_necro
end

function set_dead_apoptosis!(agent::YeastCell, model)
    agent.alive = false
    agent.is_apoptotic = false
    agent.dead_apoptosis = true
    
    # --- NUTRIENT RECYCLING ---
    x, y = agent.pos
    model.nutrient_layer[y, x] += agent.internal_nutrients
    lost_biomass_eq = agent.biomass / YIELD_TRUE
    model.total_lost_nutrients += lost_biomass_eq
    agent.biomass = 0.0 
    agent.internal_nutrients = 0.0
end

function set_dead_necrosis!(agent::YeastCell, model)
    agent.alive = false
    agent.dead_necrosis = true
    # Lysis: Instantly dump internal nutrients
    model.nutrient_layer[agent.pos[2], agent.pos[1]] += agent.internal_nutrients
    agent.internal_nutrients = 0.0
end

function apply_stress!(agent::YeastCell, local_ANTIFUNGAL::Float64, model)
    if !agent.alive; return; end

    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    end

    if agent.is_pcd_plus
        if agent.is_apoptotic
            agent.apoptosis_timer += TIME_STEP_DT
            if agent.apoptosis_timer >= APOPTOSIS_DURATION
                set_dead_apoptosis!(agent, model)
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
                            set_dead_necrosis!(agent, model)
                        end
                    end
                end
            end
        end
    else
        # PCD- Strain Logic (Necrosis Only)
        if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME
            rate_apop, rate_necro = get_death_rates(local_ANTIFUNGAL)
            if rate_necro > 0
                prob_death = 1.0 - exp(-rate_necro * TIME_STEP_DT)
                if rand() < prob_death
                    set_dead_necrosis!(agent, model)
                end
            end
        end
    end
end

# ==========================================
# --- CORE MODEL STEP (Execution Logic) ---
# ==========================================

# FIX: Changed to use id_in_position for GridSpaceSingle compatibility
function is_spot_available(p, model)
    isempty(p, model) && return true
    
    id = id_in_position(p, model)
    if id != 0
        # The spot is available only if the occupant is dead
        return !model[id].alive
    end
    
    return true
end

function complex_model_step!(model)
    n_demands = Dict{Int, Float64}()
    f_demands = Dict{Int, Float64}()
    f_releases = Dict{Int, Float64}()

    grid_n_demand = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    grid_f_demand = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)

    # --- PASS 1: Calculate Demands ---
    for agent in allagents(model)
        x, y = agent.pos
        
        # 1. Nutrient Demands (Uptake into Reservoir)
        if agent.alive
            local_n = model.nutrient_layer[y, x]
            max_reservoir = agent.biomass * RESERVOIR_FRACTION
            reservoir_deficit = max(0.0, max_reservoir - agent.internal_nutrients)

            if agent.is_apoptotic
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                n_demands[agent.id] = maintenance_cost + reservoir_deficit
            else
                if agent.biomass < DIVISION_BIOMASS
                    mu = MU_MAX * (local_n / (MONOD_KS + local_n))
                    max_possible_growth_biomass = min(mu * agent.biomass * TIME_STEP_DT, DIVISION_BIOMASS - agent.biomass)
                    growth_demand_n = max_possible_growth_biomass / YIELD_TRUE
                else
                    growth_demand_n = 0.0
                end
                
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                n_demands[agent.id] = growth_demand_n + maintenance_cost + reservoir_deficit
            end
            grid_n_demand[y, x] += n_demands[agent.id]
        end
        
        # 2. Antifungal Reversible Binding
        local_f = model.ANTIFUNGAL_layer[y, x]
        bound_f = agent.bound_ANTIFUNGAL
        current_max = agent.alive ? MAX_ANTIFUNGAL_BINDING_LIVE : MAX_ANTIFUNGAL_BINDING_DEAD
        cap_remaining = max(0.0, current_max - bound_f)

        net_change = (K_ON_ANTIFUNGAL * local_f * cap_remaining - K_OFF_ANTIFUNGAL * bound_f) * TIME_STEP_DT

        if net_change > 0
            f_demands[agent.id] = net_change
            grid_f_demand[y, x] += net_change
        else
            f_releases[agent.id] = min(-net_change, bound_f)
        end
    end

    # Tuple contains (pos, biomass, internal_nutrients, is_pcd_plus_trait)
    newborn_spots = Tuple{Tuple{Int,Int}, Float64, Float64, Bool}[] 

    # --- PASS 2: Allocate & Update ---
    for agent in allagents(model)
        x, y = agent.pos
        
        # 1. Stress Evaluation
        if agent.alive
            apply_stress!(agent, model.ANTIFUNGAL_layer[y, x], model)
        end

        # 2. Nutrients & Metabolism
        if agent.alive && get(n_demands, agent.id, 0.0) > 0
            demand = n_demands[agent.id]
            available_n = model.nutrient_layer[y, x]
            total_demand_n = grid_n_demand[y, x]
            
            alloc_frac = total_demand_n > available_n ? (available_n / total_demand_n) : 1.0
            actual_intake = demand * alloc_frac
            model.nutrient_layer[y, x] -= actual_intake
            
            # Step A: Load nutrients into internal reservoir
            agent.internal_nutrients += actual_intake

            maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

            # Step B: Metabolism (Pull from reservoir)
            if agent.is_apoptotic
                burn = min(agent.internal_nutrients, maintenance_cost)
                agent.internal_nutrients -= burn
                model.total_lost_nutrients += burn
            else
                if agent.internal_nutrients >= maintenance_cost
                    agent.internal_nutrients -= maintenance_cost
                    model.total_lost_nutrients += maintenance_cost

                    if agent.biomass < DIVISION_BIOMASS
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
                    end
                end

                if agent.alive
                    # Step C: Cap reservoir to physical limit (spill excess)
                    max_reservoir = agent.biomass * RESERVOIR_FRACTION
                    if agent.internal_nutrients > max_reservoir
                        excess = agent.internal_nutrients - max_reservoir
                        agent.internal_nutrients = max_reservoir
                        model.nutrient_layer[y, x] += excess
                    end

                    # Step D: Division Logic (Inherits `is_pcd_plus` trait)
                    if agent.biomass >= DIVISION_BIOMASS
                        if !agent.can_divide
                            excess = agent.biomass - DIVISION_BIOMASS
                            agent.biomass = DIVISION_BIOMASS
                            model.total_lost_nutrients += (excess / YIELD_TRUE)
                        else
                            chosen_spot = nothing
                            
                            immediate_hood = collect(nearby_positions(agent.pos, model, 1))
                            empty_immediate = filter(p -> is_spot_available(p, model), immediate_hood)
                            
                            if !isempty(empty_immediate)
                                weights = map(empty_immediate) do p
                                    dist_sq = (x - p[1])^2 + (y - p[2])^2
                                    dist_sq == 1 ? 1.0 : (1.0 / sqrt(2.0))
                                end
                                chosen_spot = sample(empty_immediate, Weights(weights))
                            elseif rand() < PUSH_PROBABILITY
                                block = collect(nearby_positions(agent.pos, model, MAX_PUSH_RADIUS))
                                empty_spots = filter(p -> is_spot_available(p, model), block)
                                
                                if !isempty(empty_spots)
                                    min_dist = minimum((x - p[1])^2 + (y - p[2])^2 for p in empty_spots)
                                    best_spots = filter(p -> (x - p[1])^2 + (y - p[2])^2 == min_dist, empty_spots)
                                    chosen_spot = rand(best_spots)
                                end
                            end

                            if chosen_spot !== nothing
                                agent.biomass -= NEWBORN_BIOMASS 
                                
                                daughter_fraction = NEWBORN_BIOMASS / (agent.biomass + NEWBORN_BIOMASS)
                                daughter_n = agent.internal_nutrients * daughter_fraction
                                agent.internal_nutrients -= daughter_n

                                # Pass the trait genetically to the daughter
                                push!(newborn_spots, (chosen_spot, NEWBORN_BIOMASS, daughter_n, agent.is_pcd_plus))
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

        # 3. Antifungal Binding Resolution
        if get(f_demands, agent.id, 0.0) > 0
            demand = f_demands[agent.id]
            available_f = model.ANTIFUNGAL_layer[y, x]
            total_demand_f = grid_f_demand[y, x]
            
            alloc_frac = total_demand_f > available_f ? (available_f / total_demand_f) : 1.0
            actual_binding = demand * alloc_frac
            
            agent.bound_ANTIFUNGAL += actual_binding
            model.ANTIFUNGAL_layer[y, x] -= actual_binding
        elseif get(f_releases, agent.id, 0.0) > 0
            release = f_releases[agent.id]
            agent.bound_ANTIFUNGAL -= release
            model.ANTIFUNGAL_layer[y, x] += release
        end
    end

    # Add newborns to space
    for (pos, b, n, pcd_trait) in newborn_spots
        if is_spot_available(pos, model)
            if !isempty(pos, model)
                # Spot is occupied by dead cell. Remove it and return bound antifungal to the environment.
                # FIX: use id_in_position for GridSpaceSingle
                id = id_in_position(pos, model)
                if id != 0
                    dead_cell = model[id]
                    model.ANTIFUNGAL_layer[pos[2], pos[1]] += dead_cell.bound_ANTIFUNGAL
                    remove_agent!(dead_cell, model) # Updated to use remove_agent!
                end
            end
            add_agent!(pos, YeastCell, model, pcd_trait, true, false, 0.0, 0.0, false, false, false, 0.0, b, n, true)
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

function get_total_antifungal(model)
    env_mass = sum(model.ANTIFUNGAL_layer)
    agent_mass = sum(a.bound_ANTIFUNGAL for a in allagents(model))
    return env_mass + agent_mass
end

function get_total_nutrients(model)
    env_mass = sum(model.nutrient_layer)
    agent_internal = sum(a.internal_nutrients for a in allagents(model))
    agent_biomass_eq = sum(a.biomass / YIELD_TRUE for a in allagents(model))
    respired_mass = model.total_lost_nutrients
    return env_mass + agent_internal + agent_biomass_eq + respired_mass
end

function generate_subplots(model, title_prefix::String, step::Int)
    alive_agents = [a for a in allagents(model) if a.alive]
    
    # Split healthy cells based on trait
    healthy_plus_x = Float64[a.pos[1] for a in alive_agents if a.is_pcd_plus && !a.is_apoptotic]
    healthy_plus_y = Float64[a.pos[2] for a in alive_agents if a.is_pcd_plus && !a.is_apoptotic]

    healthy_minus_x = Float64[a.pos[1] for a in alive_agents if !a.is_pcd_plus && !a.is_apoptotic]
    healthy_minus_y = Float64[a.pos[2] for a in alive_agents if !a.is_pcd_plus && !a.is_apoptotic]

    apoptotic_x = Float64[a.pos[1] for a in alive_agents if a.is_apoptotic]
    apoptotic_y = Float64[a.pos[2] for a in alive_agents if a.is_apoptotic]

    dead_apoptosis_x = Float64[a.pos[1] for a in allagents(model) if a.dead_apoptosis]
    dead_apoptosis_y = Float64[a.pos[2] for a in allagents(model) if a.dead_apoptosis]

    dead_necrosis_x = Float64[a.pos[1] for a in allagents(model) if a.dead_necrosis]
    dead_necrosis_y = Float64[a.pos[2] for a in allagents(model) if a.dead_necrosis]

    dead_starved_x = Float64[a.pos[1] for a in allagents(model) if a.dead_starvation]
    dead_starved_y = Float64[a.pos[2] for a in allagents(model) if a.dead_starvation]

    num_live = length(alive_agents)
    
    n_plot = copy(model.nutrient_layer)
    n_plot[1, 1] = INIT_NUTRIENT_LEVEL; n_plot[1, 2] = 0.0

    f_plot = copy(model.ANTIFUNGAL_layer)
    f_plot[1, 1] = INIT_ANTIFUNGAL_LEVEL; f_plot[1, 2] = 0.0

    p1 = heatmap(n_plot, title="Nutrients (Step $step)", 
                 color=:viridis, clims=(0, INIT_NUTRIENT_LEVEL), aspect_ratio=:equal)

    p2 = heatmap(f_plot, title="Antifungal", 
                 color=:ice, clims=(0, INIT_ANTIFUNGAL_LEVEL), aspect_ratio=:equal)

    p3 = plot(title="Live Cells: $num_live", 
              xlims=(1, GRID_SIZE_PX), ylims=(1, GRID_SIZE_PX), 
              aspect_ratio=:equal, legend=:outerright)

    if !isempty(healthy_plus_x)
        scatter!(p3, healthy_plus_x, healthy_plus_y, label="PCD+ Alive", color=:blue, markersize=2, markerstrokewidth=0)
    end
    if !isempty(healthy_minus_x)
        scatter!(p3, healthy_minus_x, healthy_minus_y, label="PCD- Alive", color=:red, markersize=2, markerstrokewidth=0)
    end
    if !isempty(apoptotic_x)
        scatter!(p3, apoptotic_x, apoptotic_y, label="Apoptotic (+)", color=:orange, markersize=2.5, markerstrokewidth=0)
    end
    if !isempty(dead_apoptosis_x)
        scatter!(p3, dead_apoptosis_x, dead_apoptosis_y, label="Dead (Apop)", color=:darkgray, markersize=3, markerstrokewidth=0)
    end
    if !isempty(dead_necrosis_x)
        scatter!(p3, dead_necrosis_x, dead_necrosis_y, label="Dead (Necro)", color=:black, markersize=3, markerstrokewidth=0)
    end
    if !isempty(dead_starved_x)
        scatter!(p3, dead_starved_x, dead_starved_y, label="Dead (Starve)", color=:magenta, markersize=3, markerstrokewidth=0)
    end

    ys = [a.pos[2] for a in allagents(model)]
    slice_y = isempty(ys) ? GRID_SIZE_PX ÷ 2 : clamp(round(Int, mean(ys)), 1, GRID_SIZE_PX)
    
    n_slice = model.nutrient_layer[slice_y, :]
    f_slice = model.ANTIFUNGAL_layer[slice_y, :]
    
    p4 = plot(title="Profile Slice (Y=$slice_y)", 
              xlims=(1, GRID_SIZE_PX), ylims=(0, INIT_NUTRIENT_LEVEL),
              legend=:topright, xlabel="X Position", ylabel="Concentration",
              titlefontsize=10)
    
    plot!(p4, 1:GRID_SIZE_PX, n_slice, label="Nutrients", color=:green, linewidth=2)
    plot!(p4, 1:GRID_SIZE_PX, f_slice, label="Antifungal", color=:blue, linewidth=2)
    hline!(p4, [ANTIFUNGAL_DAMAGE_THRESHOLD], label="Damage Threshold", color=:red, linestyle=:dash, linewidth=2)

    return p1, p2, p3, p4
end

# ==========================================
# --- RUN SIMULATION ---
# ==========================================

function main()
    println("Initializing Unified Petri Dish Environment...")
    
    cx = (GRID_SIZE_PX + 1) / 2.0
    cy = (GRID_SIZE_PX + 1) / 2.0
    
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    starting_positions = all_pos[1:min(INITIAL_CELLS, length(all_pos))]
    
    model = initialize_model(starting_positions)
    
    init_af = get_total_antifungal(model)
    init_n = get_total_nutrients(model)

    every_n_steps = 6 
    fps = 10 

    # --- DATA TRACKING ---
    history = Dict(
        :alive_plus => Int[], :alive_minus => Int[], 
        :dead_apop => Int[], :dead_necro_plus => Int[], :dead_necro_minus => Int[],
        :dead_starve => Int[], :total => Int[]
    )

    function record_state!(hist, m)
        push!(hist[:alive_plus], count(a -> a.alive && a.is_pcd_plus, allagents(m)))
        push!(hist[:alive_minus], count(a -> a.alive && !a.is_pcd_plus, allagents(m)))
        push!(hist[:dead_apop], count(a -> a.dead_apoptosis, allagents(m)))
        push!(hist[:dead_necro_plus], count(a -> a.dead_necrosis && a.is_pcd_plus, allagents(m)))
        push!(hist[:dead_necro_minus], count(a -> a.dead_necrosis && !a.is_pcd_plus, allagents(m)))
        push!(hist[:dead_starve], count(a -> a.dead_starvation, allagents(m)))
        push!(hist[:total], nagents(m))
    end

    record_state!(history, model) # Record Step 0

    println("Starting simulation...")
    
    anim = @animate for step in 1:SIMULATION_STEPS
        
        # --- INTRODUCE MUTATION ---
        if step == MUTATION_STEP
            println("\n--- INJECTING MUTATION AT STEP $step ($MUTATION_LOCATION) ---")
            inject_mutation!(model, MUTATION_LOCATION, MUTATION_COUNT)
        end

        # --- INJECT ANTIFUNGAL ---
        if step == ANTIFUNGAL_INJECTION_STEP
            println("\n--- INJECTING ANTIFUNGAL AT STEP $step ---")
            model.ANTIFUNGAL_layer .= INIT_ANTIFUNGAL_LEVEL
            init_af = get_total_antifungal(model) # Update expectations
        end

        Agents.step!(model, 1)
        record_state!(history, model)

        p1_n, p2_f, p3_c, p4_p = generate_subplots(model, "Co-culture", step)
        plot(p1_n, p2_f, p3_c, p4_p, layout=(2, 2), size=(1000, 1000))

        if step % 10 == 0
            println("Progress: Step $step / $SIMULATION_STEPS")
        end
    end every every_n_steps  

    gif_name = "unified_petri_dish.gif"
    gif(anim, gif_name, fps = fps)
    println("Success! GIF saved as: ", gif_name)

    # ==========================================
    # --- GENERATE DYNAMICS PLOTS ---
    # ==========================================
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT

    p_pop = plot(title="Population Over Time", xlabel="Time (hrs)", ylabel="Alive Cells", linewidth=2)
    plot!(p_pop, time_axis, history[:alive_plus], label="PCD+ Alive", color=:blue)
    plot!(p_pop, time_axis, history[:alive_minus], label="PCD- Alive", color=:red)

    # Title updated to reflect that corpses are being removed from the grid over time
    p_death = plot(title="Current Dead Cells Over Time", xlabel="Time (hrs)", ylabel="Dead Cells", linewidth=2)
    plot!(p_death, time_axis, history[:dead_apop], label="Apop (PCD+ only)", color=:orange)
    plot!(p_death, time_axis, history[:dead_necro_plus], label="Necro (PCD+)", color=:black)
    plot!(p_death, time_axis, history[:dead_necro_minus], label="Necro (PCD-)", color=:gray, linestyle=:dash)

    total_alive_series = history[:alive_plus] .+ history[:alive_minus]
    rel_fit_plus = [tot > 0 ? (p / tot) : 0.0 for (p, tot) in zip(history[:alive_plus], total_alive_series)]
    rel_fit_minus = [tot > 0 ? (m / tot) : 0.0 for (m, tot) in zip(history[:alive_minus], total_alive_series)]
    
    p_fit = plot(title="Relative Frequency (Fitness)", xlabel="Time (hrs)", ylabel="Frequency", linewidth=2, ylims=(0, 1.05))
    plot!(p_fit, time_axis, rel_fit_plus, label="PCD+ Frequency", color=:blue)
    plot!(p_fit, time_axis, rel_fit_minus, label="PCD- Frequency", color=:red)

    final_plot = plot(p_pop, p_death, p_fit, layout=(3, 1), size=(800, 1000))
    savefig(final_plot, "population_dynamics.png")
    println("Success! Dynamics plots saved as: population_dynamics.png")

    # ==========================================
    # --- REPORT ---
    # ==========================================
    println("\n==========================================")
    println("--- SIMULATION REPORT ---")
    println("==========================================")
    println("Total Population Size  : $(nagents(model))")
    println(" - PCD+ alive          : $(history[:alive_plus][end])")
    println(" - PCD- alive          : $(history[:alive_minus][end])")
    println(" - Total dead (Apop)   : $(history[:dead_apop][end])")
    println(" - Total dead (Necro)  : $(history[:dead_necro_plus][end] + history[:dead_necro_minus][end])")
    println(" - Total dead (Starve) : $(history[:dead_starve][end])")
    println()
    println("Relative Fitness:")
    println(" - PCD+                : $(round(rel_fit_plus[end], digits=3))")
    println(" - PCD-                : $(round(rel_fit_minus[end], digits=3))")
    println("==========================================")
end

main()