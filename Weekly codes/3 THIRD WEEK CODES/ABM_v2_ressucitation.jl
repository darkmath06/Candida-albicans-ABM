using Random
using StatsBase
using Images
using ImageFiltering
using Plots

# ==========================================
# --- GLOBAL EXPERIMENTAL PARAMETERS ---
# ==========================================

const GRID_SIZE_PX = 100 # 1 px = 5 µm, making the physical grid 500 µm across
const INITIAL_CELLS = 1
const SIMULATION_STEPS = 450
const TIME_STEP_DT = 1.0 / 3.0 # Assuming 1 step = 20 minutes of biological time

const INIT_NUTRIENT_LEVEL = 20
const INIT_ANTIFUNGAL_LEVEL = 16.2 # Lowered slightly so the shield effect can trigger before extinction

const DIFFUSION_NUTRIENT = 0.2
const DIFFUSION_ANTIFUNGAL = 0.2
const DIFFUSION_ITERATIONS = 15 # Number of Jacobi iterations for the Crank-Nicolson implicit solver

const MU_MAX = 0.34            # Maximum specific growth rate (mu_max)
const MONOD_KS = 5.0          # Half-velocity constant for Monod kinetics
const MAINTENANCE_COEFF = 0.015 # Maintenance coefficient (m)
const YIELD_TRUE = 0.39       # True growth yield (Y)
const YIELD_ENDOGENOUS = 0.8  # Efficiency of cannibalizing own biomass for energy
const NEWBORN_BIOMASS = 1.0   # Biomass of a daughter cell upon bud detachment
const DIVISION_BIOMASS = 2.0  # Threshold to detach a bud (Mature volume 1.0 + Bud 1.0)
const STARVATION_BIOMASS = 0.1

const PUSH_PROBABILITY = 0.02  # Probability to mechanically push when locally trapped
const MAX_PUSH_RADIUS = 10    # Max radius a cell can shove others to divide

const MAX_ANTIFUNGAL_BINDING_LIVE = 0.42  # Capacity for intact, living cells (outer leaflet only)
const MAX_ANTIFUNGAL_BINDING_DEAD = 2.55 # ~6x capacity for dead cells (internal membranes exposed)
const K_ON_ANTIFUNGAL = 0.01         # Adsorption rate constant (binding)
const K_OFF_ANTIFUNGAL = 0.005       # Desorption rate constant (unbinding/leaching)

const ANTIFUNGAL_DEATH_THRESHOLD = 16.0      # Unified stress threshold for both populations
const STRESS_START_TIME = 3.30                # Hours of exposure before death risks begin
const APOPTOSIS_DURATION = 2.0               # Hours the apoptosis process takes before death (PCD+ ONLY)
const APOPTOSIS_RATE = 0.15                  # Hourly probability rate of entering apoptosis (PCD+ only)
const NECROSIS_RATE = 0.15                   # Hourly probability rate of dying from necrosis (Both)

# --- NEW STRESS RECOVERY PARAMETERS ---
const APOPTOSIS_RECOVERY_PROB = 0.15         # Probability to reverse early apoptosis per hour if stress drops
const RETAIN_DIVISION_PROB = 0.0         # Probability a resuscitated cell can still replicate
const STRESS_CLEARANCE_HALF_LIFE = 2.0       # Hours required to clear 50% of accumulated stress memory
const STRESS_DECAY_RATE = log(2.0) / STRESS_CLEARANCE_HALF_LIFE

# ==========================================
# --- AGENT TYPES ---
# ==========================================

abstract type CellAgent end

mutable struct PCDPlus <: CellAgent
    x::Int
    y::Int
    alive::Bool
    is_apoptotic::Bool
    apoptosis_timer::Float64
    ANTIFUNGAL_exposure_time::Float64
    dead_apoptosis::Bool
    dead_necrosis::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
    can_divide::Bool       
    has_recovered::Bool    
end

# PCDMinus is stripped of all unused fields, severely reducing its memory footprint
mutable struct PCDMinus <: CellAgent
    x::Int
    y::Int
    alive::Bool
    ANTIFUNGAL_exposure_time::Float64
    dead_necrosis::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
end

PCDPlus(x::Int, y::Int; biomass=1.0) = PCDPlus(x, y, true, false, 0.0, 0.0, false, false, false, 0.0, biomass, true, false)
PCDMinus(x::Int, y::Int; biomass=1.0) = PCDMinus(x, y, true, 0.0, false, false, 0.0, biomass)

# --- Accessors for Type-Stable Generic Operations ---
is_apoptotic(a::PCDPlus) = a.is_apoptotic
is_apoptotic(a::PCDMinus) = false

can_divide(a::PCDPlus) = a.can_divide
can_divide(a::PCDMinus) = true

has_recovered(a::PCDPlus) = a.has_recovered
has_recovered(a::PCDMinus) = false

is_dead_apoptosis(a::PCDPlus) = a.dead_apoptosis
is_dead_apoptosis(a::PCDMinus) = false

set_dead_apoptosis!(a::PCDPlus) = begin
    a.alive = false
    a.is_apoptotic = false
    a.dead_apoptosis = true
end
set_dead_apoptosis!(a::PCDMinus) = nothing

# ------------------------------------------
# --- ANTIFUNGAL EXPOSURE LOGIC ---
# ------------------------------------------

function step_ANTIFUNGAL!(agent::PCDPlus, local_ANTIFUNGAL::Float64)
    if !agent.alive
        return
    end

    if local_ANTIFUNGAL >= ANTIFUNGAL_DEATH_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    else
        # Exponential decay of stress memory based on biological half-life
        agent.ANTIFUNGAL_exposure_time *= exp(-STRESS_DECAY_RATE * TIME_STEP_DT)
        if agent.ANTIFUNGAL_exposure_time < 0.01
            agent.ANTIFUNGAL_exposure_time = 0.0
        end
    end

    if agent.is_apoptotic
        # RESUSCITATION LOGIC: Reversal of early apoptosis if stress drops
        if local_ANTIFUNGAL < ANTIFUNGAL_DEATH_THRESHOLD
            # Exponential CDF scaling for discrete time step
            prob_recover = 1.0 - exp(-APOPTOSIS_RECOVERY_PROB * TIME_STEP_DT)
            if rand() < prob_recover
                agent.is_apoptotic = false
                agent.apoptosis_timer = 0.0
                agent.has_recovered = true
                # Cell either regains ability to form colonies or survives as senescent
                agent.can_divide = rand() < RETAIN_DIVISION_PROB
                return # Exit early, cell has recovered this step
            end
        end

        agent.apoptosis_timer += TIME_STEP_DT
        if agent.apoptosis_timer >= APOPTOSIS_DURATION
            agent.alive = false         
            agent.is_apoptotic = false  
            agent.dead_apoptosis = true 
        end
    else
        if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME
            # Probabilistic competing risks for Apoptosis and Necrosis
            prob_apop = 1.0 - exp(-APOPTOSIS_RATE * TIME_STEP_DT)
            prob_necro = 1.0 - exp(-NECROSIS_RATE * TIME_STEP_DT)
            
            roll = rand()
            if roll < prob_apop
                agent.is_apoptotic = true 
            elseif roll < prob_apop + prob_necro
                agent.alive = false
                agent.dead_necrosis = true
            end
        end
    end
end

function step_ANTIFUNGAL!(agent::PCDMinus, local_ANTIFUNGAL::Float64)
    if !agent.alive
        return
    end

    # PCD- Logic
    if local_ANTIFUNGAL >= ANTIFUNGAL_DEATH_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    else
        # Exponential decay of stress memory based on biological half-life
        agent.ANTIFUNGAL_exposure_time *= exp(-STRESS_DECAY_RATE * TIME_STEP_DT)
        if agent.ANTIFUNGAL_exposure_time < 0.01
            agent.ANTIFUNGAL_exposure_time = 0.0
        end
    end

    if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME
        prob_necro = 1.0 - exp(-NECROSIS_RATE * TIME_STEP_DT)
        if rand() < prob_necro
            agent.alive = false
            agent.dead_necrosis = true
        end
    end
end

# ==========================================
# --- MODEL ---
# ==========================================

# Parametric typing isolates memory pools and preserves strict type stability
mutable struct PetriDishModel{T <: CellAgent}
    size::Int
    nutrient_layer::Matrix{Float64}
    ANTIFUNGAL_layer::Matrix{Float64}
    laplacian_kernel::Matrix{Float64}
    neighbor_kernel::Matrix{Float64}
    occupied_layer::Matrix{Bool}
    agents::Vector{T} # Type-stable Vector!
end

function PetriDishModel(AgentType::Type{T}) where {T <: CellAgent}
    size = GRID_SIZE_PX

    nutrient_layer = fill(INIT_NUTRIENT_LEVEL, size, size)
    ANTIFUNGAL_layer = fill(INIT_ANTIFUNGAL_LEVEL, size, size)
    occupied_layer = zeros(Bool, size, size)

    # Improved fully isotropic 9-point Laplacian stencil for true radial diffusion
    laplacian_kernel = Float64[
        1/6   2/3   1/6;
        2/3 -10/3   2/3;
        1/6   2/3   1/6
    ]

    # Companion kernel isolating only the neighbor weights for the Implicit Solver
    neighbor_kernel = Float64[
        1/6   2/3   1/6;
        2/3   0.0   2/3;
        1/6   2/3   1/6
    ]

    agents = T[]

    for i in 1:INITIAL_CELLS
        x = rand(1:size)
        y = rand(1:size)

        while occupied_layer[y, x]
            x = rand(1:size)
            y = rand(1:size)
        end
        occupied_layer[y, x] = true

        push!(agents, T(x, y))
    end

    return PetriDishModel{T}(size, nutrient_layer, ANTIFUNGAL_layer, laplacian_kernel, neighbor_kernel, occupied_layer, agents)
end

function step_environment!(model::PetriDishModel{T}) where {T <: CellAgent}
    new_offspring = T[]

    # Arrays to store simultaneous requests
    num_agents = length(model.agents)
    n_demands = zeros(Float64, num_agents)
    f_demands = zeros(Float64, num_agents)
    f_releases = zeros(Float64, num_agents)

    grid_n_demand = zeros(Float64, model.size, model.size)
    grid_f_demand = zeros(Float64, model.size, model.size)

    # --- PASS 1: Calculate Demands (Simultaneous Evaluation) ---
    for (i, agent) in enumerate(model.agents)
        # 1. Nutrient Demands (Only living cells)
        if agent.alive
            local_n = model.nutrient_layer[agent.y, agent.x]
            maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

            if is_apoptotic(agent)
                n_demands[i] = maintenance_cost
            else
                mu = MU_MAX * (local_n / (MONOD_KS + local_n))
                growth_demand_biomass = mu * agent.biomass * TIME_STEP_DT
                nutrient_for_growth = growth_demand_biomass / YIELD_TRUE
                n_demands[i] = nutrient_for_growth + maintenance_cost
            end
            grid_n_demand[agent.y, agent.x] += n_demands[i]
        end
        
        # 2. Antifungal Reversible Binding (Both Living and Dead Cells)
        local_f = model.ANTIFUNGAL_layer[agent.y, agent.x]
        bound_f = agent.bound_ANTIFUNGAL
        
        # Massive capacity expansion upon death (membrane rupture exposes organelles)
        current_max_capacity = agent.alive ? MAX_ANTIFUNGAL_BINDING_LIVE : MAX_ANTIFUNGAL_BINDING_DEAD
        capacity_remaining = max(0.0, current_max_capacity - bound_f)

        rate_on = K_ON_ANTIFUNGAL * local_f * capacity_remaining
        rate_off = K_OFF_ANTIFUNGAL * bound_f
        net_change = (rate_on - rate_off) * TIME_STEP_DT

        if net_change > 0
            f_demands[i] = net_change
            grid_f_demand[agent.y, agent.x] += net_change
        else
            f_releases[i] = min(-net_change, bound_f)
        end
    end

    # --- PASS 2: Allocate & Update (Simultaneous Execution) ---
    for (i, agent) in enumerate(model.agents)
        
        # 1. Antifungal Exposure Updates
        if agent.alive
            local_f = model.ANTIFUNGAL_layer[agent.y, agent.x]
            step_ANTIFUNGAL!(agent, local_f)
        end

        # 2. Nutrient Allocation
        if agent.alive && n_demands[i] > 0
            available_n = model.nutrient_layer[agent.y, agent.x]
            total_demand_n = grid_n_demand[agent.y, agent.x]
            
            # Proportionally share if demand exceeds available (Simultaneous competition)
            allocation_fraction = total_demand_n > available_n ? (available_n / total_demand_n) : 1.0
            actual_intake = n_demands[i] * allocation_fraction

            model.nutrient_layer[agent.y, agent.x] -= actual_intake

            maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

            if is_apoptotic(agent)
                agent.biomass -= (maintenance_cost - actual_intake) 

                if agent.biomass <= STARVATION_BIOMASS
                    set_dead_apoptosis!(agent)
                end
            else
                if actual_intake >= maintenance_cost
                    leftover_for_growth = actual_intake - maintenance_cost
                    agent.biomass += leftover_for_growth * YIELD_TRUE
                else
                    deficit_nutrient = maintenance_cost - actual_intake
                    agent.biomass -= deficit_nutrient * YIELD_ENDOGENOUS
                end

                if agent.biomass <= STARVATION_BIOMASS
                    agent.alive = false
                    agent.dead_starvation = true
                elseif agent.biomass >= DIVISION_BIOMASS
                    if !can_divide(agent)
                        # Cell has resuscitated but is senescent (cannot replicate)
                        agent.biomass = DIVISION_BIOMASS # Cap biomass
                    else
                        chosen_spot = nothing
                        
                        # 1. Check local immediate neighborhood first (r = 1)
                        local_candidates = Tuple{Int, Int}[]
                        for dx in -1:1, dy in -1:1
                            if dx == 0 && dy == 0; continue; end
                            nx, ny = agent.x + dx, agent.y + dy
                            if 1 <= nx <= model.size && 1 <= ny <= model.size
                                if !model.occupied_layer[ny, nx]
                                    push!(local_candidates, (nx, ny))
                                end
                            end
                        end

                        if !isempty(local_candidates)
                            chosen_spot = rand(local_candidates)
                        else
                            if rand() < PUSH_PROBABILITY
                                for r in 2:MAX_PUSH_RADIUS 
                                    push_candidates = Tuple{Int, Int, Float64}[]
                                    for dx in -r:r, dy in -r:r
                                        if max(abs(dx), abs(dy)) != r; continue; end
                                        nx, ny = agent.x + dx, agent.y + dy
                                        if 1 <= nx <= model.size && 1 <= ny <= model.size
                                            if !model.occupied_layer[ny, nx]
                                                push!(push_candidates, (nx, ny, Float64(dx^2 + dy^2)))
                                            end
                                        end
                                    end
                                    
                                    if !isempty(push_candidates)
                                        min_dist = minimum(c[3] for c in push_candidates)
                                        best_spots = [(c[1], c[2]) for c in push_candidates if c[3] == min_dist]
                                        chosen_spot = rand(best_spots)
                                        break
                                    end
                                end
                            end
                        end

                        if chosen_spot !== nothing
                            # Asymmetric Budding
                            agent.biomass -= NEWBORN_BIOMASS 
                            new_x, new_y = chosen_spot
                            model.occupied_layer[new_y, new_x] = true
                            
                            # Offspring spawns cleanly as Type T
                            push!(new_offspring, T(new_x, new_y; biomass=NEWBORN_BIOMASS))
                        end
                    end
                end
            end
        end

        # 3. Antifungal Binding & Release (Applies to ALL cells now)
        if f_demands[i] > 0
            available_f = model.ANTIFUNGAL_layer[agent.y, agent.x]
            total_demand_f = grid_f_demand[agent.y, agent.x]
            
            allocation_fraction = total_demand_f > available_f ? (available_f / total_demand_f) : 1.0
            actual_binding = f_demands[i] * allocation_fraction
            
            agent.bound_ANTIFUNGAL += actual_binding
            model.ANTIFUNGAL_layer[agent.y, agent.x] -= actual_binding
        elseif f_releases[i] > 0
            agent.bound_ANTIFUNGAL -= f_releases[i]
            model.ANTIFUNGAL_layer[agent.y, agent.x] += f_releases[i]
        end
    end

    append!(model.agents, new_offspring)

    # --- Implicit Crank-Nicolson PDE Solver (Unconditionally Stable) ---
    
    alpha_n = DIFFUSION_NUTRIENT * TIME_STEP_DT
    alpha_f = DIFFUSION_ANTIFUNGAL * TIME_STEP_DT

    # 1. Compute Right Hand Side (Explicit half-step)
    rhs_n = model.nutrient_layer .+ (alpha_n / 2.0) .* imfilter(model.nutrient_layer, centered(model.laplacian_kernel), "replicate")
    rhs_f = model.ANTIFUNGAL_layer .+ (alpha_f / 2.0) .* imfilter(model.ANTIFUNGAL_layer, centered(model.laplacian_kernel), "replicate")

    u_n = copy(model.nutrient_layer)
    u_f = copy(model.ANTIFUNGAL_layer)

    # Precalculate denominator for the matrix division (central weight of Laplacian is -10/3)
    denom_n = 1.0 + (5.0 / 3.0) * alpha_n
    denom_f = 1.0 + (5.0 / 3.0) * alpha_f

    # 2. Jacobi Iteration for the Implicit half-step
    for _ in 1:DIFFUSION_ITERATIONS
        u_n = (rhs_n .+ (alpha_n / 2.0) .* imfilter(u_n, centered(model.neighbor_kernel), "replicate")) ./ denom_n
        u_f = (rhs_f .+ (alpha_f / 2.0) .* imfilter(u_f, centered(model.neighbor_kernel), "replicate")) ./ denom_f
    end

    model.nutrient_layer .= max.(u_n, 0.0)
    model.ANTIFUNGAL_layer .= max.(u_f, 0.0)
end

# Helper to verify Antifungal Conservation
function get_total_antifungal(model::PetriDishModel)
    env_mass = sum(model.ANTIFUNGAL_layer)
    agent_mass = sum(a.bound_ANTIFUNGAL for a in model.agents)
    return env_mass + agent_mass
end

# Generates side-by-side plots modularly
function generate_subplots(model::PetriDishModel{T}, title_prefix::String, step::Int) where {T <: CellAgent}
    alive_agents = [a for a in model.agents if a.alive]
    
    healthy_x = Float64[a.x for a in alive_agents if !is_apoptotic(a) && !has_recovered(a)]
    healthy_y = Float64[a.y for a in alive_agents if !is_apoptotic(a) && !has_recovered(a)]
    
    recovered_x = Float64[a.x for a in alive_agents if has_recovered(a)]
    recovered_y = Float64[a.y for a in alive_agents if has_recovered(a)]

    apoptotic_x = Float64[a.x for a in alive_agents if is_apoptotic(a)]
    apoptotic_y = Float64[a.y for a in alive_agents if is_apoptotic(a)]

    dead_apoptosis_x = Float64[a.x for a in model.agents if is_dead_apoptosis(a)]
    dead_apoptosis_y = Float64[a.y for a in model.agents if is_dead_apoptosis(a)]

    dead_necrosis_x = Float64[a.x for a in model.agents if a.dead_necrosis]
    dead_necrosis_y = Float64[a.y for a in model.agents if a.dead_necrosis]

    dead_starved_x = Float64[a.x for a in model.agents if a.dead_starvation]
    dead_starved_y = Float64[a.y for a in model.agents if a.dead_starvation]

    num_live = length(alive_agents)
    
    # Hack to prevent PlotUtils "No strict ticks found" warning on completely flat empty grids.
    # We anchor the min and max data values in the corner to stabilize the colorbar.
    n_plot = copy(model.nutrient_layer)
    n_plot[1, 1] = INIT_NUTRIENT_LEVEL
    n_plot[1, 2] = 0.0

    f_plot = copy(model.ANTIFUNGAL_layer)
    f_plot[1, 1] = INIT_ANTIFUNGAL_LEVEL
    f_plot[1, 2] = 0.0

    p1 = heatmap(n_plot, 
                 title="$title_prefix Nutrients (Step $step)", 
                 color=:viridis, clims=(0, INIT_NUTRIENT_LEVEL),
                 aspect_ratio=:equal)

    p2 = heatmap(f_plot, 
                 title="$title_prefix Antifungal", 
                 color=:ice, clims=(0, INIT_ANTIFUNGAL_LEVEL),
                 aspect_ratio=:equal)

    p3 = plot(title="$title_prefix Live Cells: $num_live", 
              xlims=(1, model.size), ylims=(1, model.size), 
              aspect_ratio=:equal, legend=:topright)

    healthy_color = T === PCDPlus ? :blue : :red
    healthy_label = T === PCDPlus ? "PCD+" : "PCD-"

    if !isempty(healthy_x)
        scatter!(p3, healthy_x, healthy_y, label=healthy_label, color=healthy_color, markersize=2, markerstrokewidth=0)
    end
    if !isempty(recovered_x)
        scatter!(p3, recovered_x, recovered_y, label="Recovered", color=:cyan, markersize=2, markerstrokewidth=0)
    end
    if !isempty(apoptotic_x)
        scatter!(p3, apoptotic_x, apoptotic_y, label="Apoptotic", color=:orange, markersize=2.5, markerstrokewidth=0)
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

    # --- NEW: 1D Mathematical Concentration Profile (Cross-Section) ---
    # Find the Y-center of the colony to take a slice
    ys = [a.y for a in model.agents]
    slice_y = isempty(ys) ? model.size ÷ 2 : clamp(round(Int, mean(ys)), 1, model.size)
    
    n_slice = model.nutrient_layer[slice_y, :]
    f_slice = model.ANTIFUNGAL_layer[slice_y, :]
    
    p4 = plot(title="$title_prefix Profile (Y=$slice_y)", 
              xlims=(1, model.size), ylims=(0, INIT_NUTRIENT_LEVEL),
              legend=:topright, xlabel="X Position", ylabel="Concentration",
              titlefontsize=10)
    
    # Plot the smooth diffusion curves
    plot!(p4, 1:model.size, n_slice, label="Nutrients", color=:green, linewidth=2)
    plot!(p4, 1:model.size, f_slice, label="Antifungal", color=:blue, linewidth=2)
    
    # Add a dashed line to visualize the stress shield threshold
    hline!(p4, [ANTIFUNGAL_DEATH_THRESHOLD], label="Death Threshold", color=:red, linestyle=:dash, linewidth=2)

    return p1, p2, p3, p4
end

function main()
    println("Initializing independent simulations...")
    
    # Initialize the two independent environments using strictly typed environments
    model_plus = PetriDishModel(PCDPlus)
    model_minus = PetriDishModel(PCDMinus)
    
    init_af_plus = get_total_antifungal(model_plus)
    init_af_minus = get_total_antifungal(model_minus)

    every_n_steps = 6 
    fps = 10 

    println("Starting dual simulation and recording frames...")
    
    anim = @animate for step in 1:SIMULATION_STEPS
        step_environment!(model_plus)
        step_environment!(model_minus)

        # Generate subplots for both environments (Now returns 4 plots each)
        p1_n, p1_f, p1_c, p1_p = generate_subplots(model_plus, "PCD+", step)
        p2_n, p2_f, p2_c, p2_p = generate_subplots(model_minus, "PCD-", step)

        # Compose 2x4 Layout (Added the Math Profiles to the right side)
        plot(p1_n, p1_f, p1_c, p1_p, p2_n, p2_f, p2_c, p2_p, layout=(2, 4), size=(1600, 800))

        if step % 10 == 0
            println("Progress: Step $step / $SIMULATION_STEPS")
        end
    end every every_n_steps  

    gif_name = "dual_petri_dish_simulation.gif"
    gif(anim, gif_name, fps = fps)
    println("Success! GIF saved as: ", gif_name)

    # ==========================================
    # --- CONSERVATION & FREQUENCY REPORT ---
    # ==========================================
    final_af_plus = get_total_antifungal(model_plus)
    final_af_minus = get_total_antifungal(model_minus)
    
    final_pcd_plus = count(a -> a.alive, model_plus.agents)
    final_pcd_minus = count(a -> a.alive, model_minus.agents)

    # --- Mortality Counting ---
    apop_plus = count(is_dead_apoptosis, model_plus.agents)
    necro_plus = count(a -> a.dead_necrosis, model_plus.agents)
    starved_plus = count(a -> a.dead_starvation, model_plus.agents)
    antifungal_deaths_plus = apop_plus + necro_plus

    apop_minus = count(is_dead_apoptosis, model_minus.agents)
    necro_minus = count(a -> a.dead_necrosis, model_minus.agents)
    starved_minus = count(a -> a.dead_starvation, model_minus.agents)
    antifungal_deaths_minus = apop_minus + necro_minus

    println("\n==========================================")
    println("--- INDEPENDENT POPULATION REPORT ---")
    println("==========================================")
    println("Final Live Cells:")
    println("  PCD+ Env : $final_pcd_plus")
    println("  PCD- Env : $final_pcd_minus")
    println("------------------------------------------")
    println("Mortality Breakdown:")
    println("  PCD+ Dead (Antifungal) : $antifungal_deaths_plus  (Apop: $apop_plus, Necro: $necro_plus)")
    println("  PCD+ Dead (Starvation) : $starved_plus")
    println("  PCD- Dead (Antifungal) : $antifungal_deaths_minus  (Apop: $apop_minus, Necro: $necro_minus)")
    println("  PCD- Dead (Starvation) : $starved_minus")
    println("------------------------------------------")
    println("Absolute Fitness (Final / Initial):")
    println("  PCD+ : $(round(final_pcd_plus / INITIAL_CELLS, digits=3))")
    println("  PCD- : $(round(final_pcd_minus / INITIAL_CELLS, digits=3))")
    println("==========================================")
    println("\n==========================================")
    println("--- ANTIFUNGAL MASS CONSERVATION CHECK ---")
    println("==========================================")
    println("PCD+ Env:")
    println("  Initial System Total : $(round(init_af_plus, digits=2))")
    println("  Final System Total   : $(round(final_af_plus, digits=2))")
    println("  Mass Lost            : $(round(init_af_plus - final_af_plus, digits=2))")
    println("PCD- Env:")
    println("  Initial System Total : $(round(init_af_minus, digits=2))")
    println("  Final System Total   : $(round(final_af_minus, digits=2))")
    println("  Mass Lost            : $(round(init_af_minus - final_af_minus, digits=2))")
    println("==========================================\n")
end

main()