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
const SIMULATION_STEPS = 400
const ANTIFUNGAL_INJECTION_STEP = 73 # 24h
const TIME_STEP_DT = 1.0 / 3.0 # Assuming 1 step = 20 minutes of biological time

# --- NEW: Biomass Capacity Limits ---
const MAX_BIOMASS_PER_PX = 3 # Allows multiple cells to stack up to a maximum biomass per coordinate

# --- Environment Levels ---
const INIT_NUTRIENT_LEVEL = 12 #2.56 units of nutrient for growth (per px)
const INIT_ANTIFUNGAL_LEVEL = 3.1 #3.2
    
# --- Diffusion Settings ---
const DIFFUSION_NUTRIENT = 0.57
const DIFFUSION_ANTIFUNGAL = 0.1
const DIFFUSION_ITERATIONS = 15 # Jacobi iterations for the implicit solver

# --- Growth & Metabolism ---
const MU_MAX = 0.7#0.34            # Maximum specific growth rate (mu_max)
const MONOD_KS = 5.0           # Half-velocity constant for Monod kinetics
const MAINTENANCE_COEFF = 0.015 # Maintenance coefficient (m)
const YIELD_TRUE = 0.39        # True growth yield (Y)
const NEWBORN_BIOMASS = 1.0    # Biomass of a daughter cell upon bud detachment
const DIVISION_BIOMASS = 2.0   # Threshold to detach a bud
const STARVATION_BIOMASS = 0.75 # Cells die if they shrink to this biomass level due to starvation
const RESERVOIR_FRACTION = 0.1 # Max internal nutrients as a fraction of current biomass

# --- Mechanics & Space ---
const PUSH_PROBABILITY = 0.3 # Probability to mechanically push when locally trapped
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
    # GridSpace allows multiple agents to share the same pixel (up to our MAX_BIOMASS limit)
    space = GridSpace((GRID_SIZE_PX, GRID_SIZE_PX); periodic=false)

    props = PetriDishProperties(
        fill(INIT_NUTRIENT_LEVEL, GRID_SIZE_PX, GRID_SIZE_PX),
        fill(0.0, GRID_SIZE_PX, GRID_SIZE_PX), # Start with 0.0 antifungal layer
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
    if c <= 0.0; return 0.0, 0.0; end
    
    C_vals = (0.0, 3, 4.0, 8.0, 16.0)
    apop_vals = (0.0, 0.12, 0.21, 0.57, 0.09)
    necro_vals = (0.0,0.08, 0.19, 0.33, 0.91)

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

function apply_stress!(agent::PCDPlusCell, local_ANTIFUNGAL::Float64, model)
    if !agent.alive; return; end

    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    end

    if agent.is_apoptotic
        agent.apoptosis_timer += TIME_STEP_DT
        if agent.apoptosis_timer >= APOPTOSIS_DURATION
            set_dead_apoptosis!(agent)
            
            # --- NUTRIENT RECYCLING ---
            x, y = agent.pos
            
            # Release 100% of internal reservoir to grid
            model.nutrient_layer[y, x] += agent.internal_nutrients
            agent.internal_nutrients = 0.0
            
            # Corpse (biomass) stays on the grid indefinitely!
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
                        # Lysis: Instantly dump internal nutrients
                        model.nutrient_layer[agent.pos[2], agent.pos[1]] += agent.internal_nutrients
                        agent.internal_nutrients = 0.0
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
        
        if rate_necro > 0
            prob_death = 1.0 - exp(-rate_necro * TIME_STEP_DT)
            if rand() < prob_death
                agent.alive = false
                agent.dead_necrosis = true
                # Lysis: Instantly dump internal nutrients
                model.nutrient_layer[agent.pos[2], agent.pos[1]] += agent.internal_nutrients
                agent.internal_nutrients = 0.0
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

            if is_apoptotic(agent)
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                n_demands[agent.id] = maintenance_cost + reservoir_deficit
            else
                # Contact Inhibition checks
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

    # Contains (pos, biomass, internal_nutrients)
    newborn_spots = Tuple{Tuple{Int,Int}, Float64, Float64}[] 
    
    # Track biomass allocation scheduled during this step to prevent multi-spawning overfill
    planned_biomass = Dict{Tuple{Int,Int}, Float64}()

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
            if is_apoptotic(agent)
                burn = min(agent.internal_nutrients, maintenance_cost)
                agent.internal_nutrients -= burn
                model.total_lost_nutrients += burn
            else
                if agent.internal_nutrients >= maintenance_cost
                    # Pay maintenance
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
                    # Starving
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
                    # Step C: Cap reservoir to physical limit
                    max_reservoir = agent.biomass * RESERVOIR_FRACTION
                    if agent.internal_nutrients > max_reservoir
                        excess = agent.internal_nutrients - max_reservoir
                        agent.internal_nutrients = max_reservoir
                        model.nutrient_layer[y, x] += excess
                    end

                    # Step D: Division Logic (Capacity Based)
                    if agent.biomass >= DIVISION_BIOMASS
                        if !can_divide(agent)
                            excess = agent.biomass - DIVISION_BIOMASS
                            agent.biomass = DIVISION_BIOMASS
                            model.total_lost_nutrients += (excess / YIELD_TRUE) 
                        else
                            chosen_spot = nothing
                            
                            # Check cell's own pixel and immediate neighborhood
                            immediate_hood = [agent.pos; collect(nearby_positions(agent.pos, model, 1))]
                            
                            empty_immediate = filter(p -> begin
                                cur_b = sum((a.biomass for a in agents_in_position(p, model)), init=0.0)
                                cur_b + get(planned_biomass, p, 0.0) + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX
                            end, immediate_hood)
                            
                            if !isempty(empty_immediate)
                                weights = map(empty_immediate) do p
                                    dist_sq = (x - p[1])^2 + (y - p[2])^2
                                    dist_sq == 0 ? 1.0 : (dist_sq == 1 ? 1.0 : (1.0 / sqrt(2.0)))
                                end
                                chosen_spot = sample(empty_immediate, Weights(weights))
                                
                            elseif rand() < PUSH_PROBABILITY
                                # Force Push -> Search in wider radius for capacity
                                block = collect(nearby_positions(agent.pos, model, MAX_PUSH_RADIUS))
                                empty_spots = filter(p -> begin
                                    cur_b = sum((a.biomass for a in agents_in_position(p, model)), init=0.0)
                                    cur_b + get(planned_biomass, p, 0.0) + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX
                                end, block)
                                
                                if !isempty(empty_spots)
                                    min_dist = minimum((x - p[1])^2 + (y - p[2])^2 for p in empty_spots)
                                    best_spots = filter(p -> (x - p[1])^2 + (y - p[2])^2 == min_dist, empty_spots)
                                    
                                    # Weight pushing to prefer emptier pixels over densely packed ones
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

    # Spawning the actual newborns
    for (pos, b, n) in newborn_spots
        cur_b = sum((a.biomass for a in agents_in_position(pos, model)), init=0.0)
        
        # Final capacity double-check before placing agent physically
        if cur_b + b <= MAX_BIOMASS_PER_PX + 0.01 
            if model.is_pcd_plus
                add_agent!(pos, PCDPlusCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, b, n, true)
            else
                add_agent!(pos, PCDMinusCell, model, true, 0.0, false, false, 0.0, b, n)
            end
        else
            # Extremely rare collision fallback
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
    f_plot[1, 1] = INIT_ANTIFUNGAL_LEVEL; f_plot[1, 2] = 0.0

    p1 = Plots.heatmap(n_plot, title="$title_prefix Nutrients (Step $step)", 
                 color=:viridis, clims=(0, INIT_NUTRIENT_LEVEL), aspect_ratio=:equal)

    p2 = Plots.heatmap(f_plot, title="$title_prefix Antifungal", 
                 color=:ice, clims=(0, INIT_ANTIFUNGAL_LEVEL), aspect_ratio=:equal)

    p3 = Plots.plot(title="$title_prefix Live Cells: $num_live", 
              xlims=(1, GRID_SIZE_PX), ylims=(1, GRID_SIZE_PX), 
              aspect_ratio=:equal, legend=:topright)

    healthy_color = model.is_pcd_plus ? :blue : :red
    healthy_label = model.is_pcd_plus ? "PCD+" : "PCD-"

    if !isempty(healthy_x)
        Plots.scatter!(p3, healthy_x, healthy_y, label=healthy_label, color=healthy_color, markersize=2, markerstrokewidth=0)
    end
    if !isempty(apoptotic_x)
        Plots.scatter!(p3, apoptotic_x, apoptotic_y, label="Apoptotic", color=:orange, markersize=2.5, markerstrokewidth=0)
    end
    if !isempty(dead_apoptosis_x)
        Plots.scatter!(p3, dead_apoptosis_x, dead_apoptosis_y, label="Dead (Apop)", color=:darkgray, markersize=3, markerstrokewidth=0)
    end
    if !isempty(dead_necrosis_x)
        Plots.scatter!(p3, dead_necrosis_x, dead_necrosis_y, label="Dead (Necro)", color=:black, markersize=3, markerstrokewidth=0)
    end
    if !isempty(dead_starved_x)
        Plots.scatter!(p3, dead_starved_x, dead_starved_y, label="Dead (Starve)", color=:magenta, markersize=3, markerstrokewidth=0)
    end

    ys = [a.pos[2] for a in allagents(model)]
    slice_y = isempty(ys) ? GRID_SIZE_PX ÷ 2 : clamp(round(Int, mean(ys)), 1, GRID_SIZE_PX)
    
    n_slice = model.nutrient_layer[slice_y, :]
    f_slice = model.ANTIFUNGAL_layer[slice_y, :]
    
    p4 = Plots.plot(title="$title_prefix Profile (Y=$slice_y)", 
              xlims=(1, GRID_SIZE_PX), ylims=(0, INIT_NUTRIENT_LEVEL),
              legend=:topright, xlabel="X Position", ylabel="Concentration",
              titlefontsize=10)
    
    Plots.plot!(p4, 1:GRID_SIZE_PX, n_slice, label="Nutrients", color=:green, linewidth=2)
    Plots.plot!(p4, 1:GRID_SIZE_PX, f_slice, label="Antifungal", color=:blue, linewidth=2)
    Plots.hline!(p4, [ANTIFUNGAL_DAMAGE_THRESHOLD], label="Damage Threshold", color=:red, linestyle=:dash, linewidth=2)

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

    println("Starting dual simulation and recording frames...")
    
    anim = Plots.@animate for step in 1:SIMULATION_STEPS
        
        if step == ANTIFUNGAL_INJECTION_STEP
            println("--- INJECTING ANTIFUNGAL AT STEP $step ---")
            model_plus.ANTIFUNGAL_layer .= INIT_ANTIFUNGAL_LEVEL
            model_minus.ANTIFUNGAL_layer .= INIT_ANTIFUNGAL_LEVEL
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

        Plots.plot(p1_n, p1_f, p1_c, p1_p, p2_n, p2_f, p2_c, p2_p, layout=(2, 4), size=(1600, 800))

        if step % 10 == 0
            println("Progress: Step $step / $SIMULATION_STEPS")
        end
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
    # --- GENERATE REQUESTED DYNAMICS PLOTS ---
    # ==========================================
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT

    p_pop = Plots.plot(title="Population (Live & Dead) Over Time", xlabel="Time (hrs)", ylabel="Cells", linewidth=2)
    Plots.plot!(p_pop, time_axis, history_plus[:alive], label="PCD+ Alive", color=:blue)
    Plots.plot!(p_pop, time_axis, history_plus[:total] .- history_plus[:alive], label="PCD+ Dead", color=:lightblue, linestyle=:dash)
    Plots.plot!(p_pop, time_axis, history_minus[:alive], label="PCD- Alive", color=:red)
    Plots.plot!(p_pop, time_axis, history_minus[:total] .- history_minus[:alive], label="PCD- Dead", color=:pink, linestyle=:dash)

    p_death = Plots.plot(title="Intact Dead Cells On Grid Over Time", xlabel="Time (hrs)", ylabel="Dead Cells", linewidth=2)
    Plots.plot!(p_death, time_axis, history_plus[:dead_apop], label="PCD+ Apop", color=:orange)
    Plots.plot!(p_death, time_axis, history_plus[:dead_necro], label="PCD+ Necro", color=:black)
    Plots.plot!(p_death, time_axis, history_minus[:dead_apop], label="PCD- Apop", color=:orange, linestyle=:dash)
    Plots.plot!(p_death, time_axis, history_minus[:dead_necro], label="PCD- Necro", color=:gray, linestyle=:dash)

    total_alive_series = history_plus[:alive] .+ history_minus[:alive]
    rel_fit_plus = [tot > 0 ? (p / tot) : 0.0 for (p, tot) in zip(history_plus[:alive], total_alive_series)]
    rel_fit_minus = [tot > 0 ? (m / tot) : 0.0 for (m, tot) in zip(history_minus[:alive], total_alive_series)]
    
    p_fit = Plots.plot(title="Relative Fitness Over Time", xlabel="Time (hrs)", ylabel="Frequency", linewidth=2, ylims=(0, 1.05))
    Plots.plot!(p_fit, time_axis, rel_fit_plus, label="PCD+", color=:blue)
    Plots.plot!(p_fit, time_axis, rel_fit_minus, label="PCD-", color=:red)

    final_plot = Plots.plot(p_pop, p_death, p_fit, layout=(3, 1), size=(800, 1000))
    Plots.savefig(final_plot, "population_dynamics.png")
    println("Success! Dynamics plots saved as: population_dynamics.png")
end

main()