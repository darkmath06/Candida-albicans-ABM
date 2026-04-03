using Agents
using Random
using StatsBase
using Images
using ImageFiltering
using Plots

# ==========================================
# --- GLOBAL EXPERIMENTAL PARAMETERS ---
# ==========================================

const GRID_SIZE_PX = 100 # 1 px = 5 µm, making the physical grid 500 µm across
const INITIAL_CELLS = 400
const SIMULATION_STEPS = 450
const TIME_STEP_DT = 1.0 / 3.0 # Assuming 1 step = 20 minutes of biological time

# --- Environment Levels ---
const INIT_NUTRIENT_LEVEL = 12
const INIT_ANTIFUNGAL_LEVEL = 4 # Set to 8.0 to clearly see the Apoptosis/ROS sweet spot!

# --- Diffusion Settings ---
const DIFFUSION_NUTRIENT = 0.2
const DIFFUSION_ANTIFUNGAL = 0.2
const DIFFUSION_ITERATIONS = 15 # Jacobi iterations for the implicit solver

# --- Growth & Metabolism ---
const MU_MAX = 0.34            # Maximum specific growth rate (mu_max)
const MONOD_KS = 5.0           # Half-velocity constant for Monod kinetics
const MAINTENANCE_COEFF = 0.015 # Maintenance coefficient (m)
const YIELD_TRUE = 0.39        # True growth yield (Y)
const NEWBORN_BIOMASS = 1.0    # Biomass of a daughter cell upon bud detachment
const DIVISION_BIOMASS = 2.0   # Threshold to detach a bud
const STARVATION_BIOMASS = 0.1 # Threshold below which a cell dies of starvation
const APOPTOSIS_NUTRIENT_RELEASE_FRAC = 0.2 # Fraction of biomass returned to environment upon apoptotic death

# --- Mechanics & Space ---
const PUSH_PROBABILITY = 0.02 # Probability to mechanically push when locally trapped
const MAX_PUSH_RADIUS = 10    # Max radius a cell can shove others to divide

# --- Antifungal Binding Kinetics ---
const MAX_ANTIFUNGAL_BINDING_LIVE = 0.42  # Capacity for intact, living cells
const MAX_ANTIFUNGAL_BINDING_DEAD = 2.55  # Capacity for dead cells (sponge effect)
const K_ON_ANTIFUNGAL = 0.01              # Adsorption rate constant
const K_OFF_ANTIFUNGAL = 0.005            # Desorption rate constant

# --- Stress, Apoptosis & Necrosis ---
const ANTIFUNGAL_DAMAGE_THRESHOLD = 0.05     # Threshold to start accumulating ROS
const STRESS_START_TIME = 0.25               # Hours of exposure before death risks begin
const APOPTOSIS_DURATION = 2.0               # Hours the apoptosis process takes

# ==========================================
# --- AGENT TYPES ---
# ==========================================

@agent struct PCDPlusCell(GridAgent{2})
    alive::Bool
    is_stressed::Bool      # NEW: Represents cells that are ROS+
    is_apoptotic::Bool
    apoptosis_timer::Float64
    ANTIFUNGAL_exposure_time::Float64
    dead_apoptosis::Bool
    dead_necrosis::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
    can_divide::Bool       
end

@agent struct PCDMinusCell(GridAgent{2})
    alive::Bool
    is_stressed::Bool      # NEW: They still accumulate ROS, but cannot undergo apoptosis
    ANTIFUNGAL_exposure_time::Float64
    dead_necrosis::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
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
end

function initialize_model(AgentType::Type)
    space = GridSpaceSingle((GRID_SIZE_PX, GRID_SIZE_PX); periodic=false)

    props = PetriDishProperties(
        fill(INIT_NUTRIENT_LEVEL, GRID_SIZE_PX, GRID_SIZE_PX),
        fill(INIT_ANTIFUNGAL_LEVEL, GRID_SIZE_PX, GRID_SIZE_PX),
        [1/6 2/3 1/6; 2/3 -10/3 2/3; 1/6 2/3 1/6],
        [1/6 2/3 1/6; 2/3 0.0 2/3; 1/6 2/3 1/6],
        AgentType === PCDPlusCell
    )

    model = StandardABM(AgentType, space; properties=props, model_step! = complex_model_step!)

    center = (GRID_SIZE_PX ÷ 2, GRID_SIZE_PX ÷ 2)
    
    for _ in 1:INITIAL_CELLS
        pos = center
        if !isempty(pos, model)
            for r in 1:GRID_SIZE_PX
                hood = collect(nearby_positions(center, model, r))
                empty_spots = filter(p -> isempty(p, model), hood)
                if !isempty(empty_spots)
                    min_dist = minimum((center[1] - p[1])^2 + (center[2] - p[2])^2 for p in empty_spots)
                    best_spots = filter(p -> (center[1] - p[1])^2 + (center[2] - p[2])^2 == min_dist, empty_spots)
                    pos = rand(best_spots)
                    break
                end
            end
        end
        
        if AgentType === PCDPlusCell
            add_agent!(pos, PCDPlusCell, model, true, false, false, 0.0, 0.0, false, false, false, 0.0, 1.0, true)
        else
            add_agent!(pos, PCDMinusCell, model, true, false, 0.0, false, false, 0.0, 1.0)
        end
    end

    return model
end

# ==========================================
# --- ANTIFUNGAL EXPOSURE LOGIC ---
# ==========================================

# Calculates continuous transition rates calibrated to hit the 200-minute assay endpoints
function get_transition_rates(c::Float64)
    if c <= 0.0
        return 0.0, 0.0, 0.0
    end
    
    # AMB Concentrations
    C_vals = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]
    
    # These hourly rates (k) were pre-calculated using the exponential decay formula
    # k = -ln(1 - TargetFraction) / 3.33 hours to exactly match Graphs A & B.
    
    # Rate of direct Necrosis (Graph A - White Bars)
    k_necro_vals  = [0.0, 0.0, 0.0,   0.067, 0.129, 0.690]
    
    # Rate of entering ROS+ State from Healthy (Graph B)
    k_stress_vals = [0.0, 0.0, 0.360, 0.690, 2.000, 0.150]
    
    # Rate of entering Apoptosis FROM the ROS+ State (Graph A - Black Bars)
    k_apop_vals   = [0.0, 0.0, 0.0,   0.098, 0.759, 1.000]

    # Handle concentrations above 16
    if c >= C_vals[end]
        return k_necro_vals[end], k_stress_vals[end], k_apop_vals[end]
    end
    
    # Linear Interpolation for values in between assay points
    for i in 1:(length(C_vals)-1)
        if c >= C_vals[i] && c <= C_vals[i+1]
            t = (c - C_vals[i]) / (C_vals[i+1] - C_vals[i])
            kn = k_necro_vals[i] + t * (k_necro_vals[i+1] - k_necro_vals[i])
            ks = k_stress_vals[i] + t * (k_stress_vals[i+1] - k_stress_vals[i])
            ka = k_apop_vals[i] + t * (k_apop_vals[i+1] - k_apop_vals[i])
            return kn, ks, ka
        end
    end
    
    return 0.0, 0.0, 0.0
end

function apply_stress!(agent::PCDPlusCell, local_ANTIFUNGAL::Float64, model)
    if !agent.alive; return; end

    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    end

    if agent.is_apoptotic
        # Apoptosis is a point of no return. No recovery/anastasis possible.
        agent.apoptosis_timer += TIME_STEP_DT
        if agent.apoptosis_timer >= APOPTOSIS_DURATION
            set_dead_apoptosis!(agent)
            
            # --- NUTRIENT RECYCLING ---
            x, y = agent.pos
            model.nutrient_layer[y, x] += agent.biomass * APOPTOSIS_NUTRIENT_RELEASE_FRAC
            agent.biomass = 0.0 
        end
        return # Skip further stress rolls
    end

    if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME
        k_necro, k_stress, k_apop = get_transition_rates(local_ANTIFUNGAL)
        
        # 1. Roll for Necrosis (Can happen to healthy or stressed cells if membrane bursts)
        if k_necro > 0
            prob_necro = 1.0 - exp(-k_necro * TIME_STEP_DT)
            if rand() < prob_necro
                agent.alive = false
                agent.dead_necrosis = true
                agent.is_stressed = false # Clears the state
                return
            end
        end
        
        # 2. Roll for Stress (ROS+) -> Only applies if cell is currently healthy
        if !agent.is_stressed
            if k_stress > 0
                prob_stress = 1.0 - exp(-k_stress * TIME_STEP_DT)
                if rand() < prob_stress
                    agent.is_stressed = true
                end
            end
        end
        
        # 3. Roll for Apoptosis -> ONLY applies if the cell is ALREADY stressed (ROS+)
        if agent.is_stressed
            if k_apop > 0
                prob_apop = 1.0 - exp(-k_apop * TIME_STEP_DT)
                if rand() < prob_apop
                    agent.is_apoptotic = true
                    agent.is_stressed = false # Consumed by the apoptosis process
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
        k_necro, k_stress, _ = get_transition_rates(local_ANTIFUNGAL)
        
        # 1. Roll for Necrosis
        if k_necro > 0
            prob_necro = 1.0 - exp(-k_necro * TIME_STEP_DT)
            if rand() < prob_necro
                agent.alive = false
                agent.dead_necrosis = true
                agent.is_stressed = false
                return
            end
        end
        
        # 2. Roll for Stress (ROS+)
        # PCD- cells still experience severe oxidative stress, they just can't execute apoptosis
        if !agent.is_stressed
            if k_stress > 0
                prob_stress = 1.0 - exp(-k_stress * TIME_STEP_DT)
                if rand() < prob_stress
                    agent.is_stressed = true
                end
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
        
        # 1. Nutrient Demands
        if agent.alive
            local_n = model.nutrient_layer[y, x]
            maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

            if is_apoptotic(agent)
                n_demands[agent.id] = maintenance_cost
            else
                mu = MU_MAX * (local_n / (MONOD_KS + local_n))
                growth_demand_biomass = mu * agent.biomass * TIME_STEP_DT
                nutrient_for_growth = growth_demand_biomass / YIELD_TRUE
                n_demands[agent.id] = nutrient_for_growth + maintenance_cost
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

    newborn_spots = Tuple{Tuple{Int,Int}, Float64}[] 

    # --- PASS 2: Allocate & Update ---
    for agent in allagents(model)
        x, y = agent.pos
        
        # 1. Stress Evaluation
        if agent.alive
            apply_stress!(agent, model.ANTIFUNGAL_layer[y, x], model)
        end

        # 2. Nutrients
        if agent.alive && get(n_demands, agent.id, 0.0) > 0
            demand = n_demands[agent.id]
            available_n = model.nutrient_layer[y, x]
            total_demand_n = grid_n_demand[y, x]
            
            alloc_frac = total_demand_n > available_n ? (available_n / total_demand_n) : 1.0
            actual_intake = demand * alloc_frac
            model.nutrient_layer[y, x] -= actual_intake

            maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

            if !is_apoptotic(agent)
                agent.biomass += (actual_intake - maintenance_cost) * YIELD_TRUE

                if agent.biomass <= STARVATION_BIOMASS
                    agent.alive = false
                    agent.dead_starvation = true
                elseif agent.biomass >= DIVISION_BIOMASS
                    if !can_divide(agent)
                        agent.biomass = DIVISION_BIOMASS
                    else
                        chosen_spot = nothing
                        
                        immediate_hood = collect(nearby_positions(agent.pos, model, 1))
                        empty_immediate = filter(p -> isempty(p, model), immediate_hood)
                        
                        if !isempty(empty_immediate)
                            weights = map(empty_immediate) do p
                                dist_sq = (x - p[1])^2 + (y - p[2])^2
                                dist_sq == 1 ? 1.0 : (1.0 / sqrt(2.0))
                            end
                            chosen_spot = sample(empty_immediate, Weights(weights))
                        elseif rand() < PUSH_PROBABILITY
                            block = collect(nearby_positions(agent.pos, model, MAX_PUSH_RADIUS))
                            empty_spots = filter(p -> isempty(p, model), block)
                            
                            if !isempty(empty_spots)
                                min_dist = minimum((x - p[1])^2 + (y - p[2])^2 for p in empty_spots)
                                best_spots = filter(p -> (x - p[1])^2 + (y - p[2])^2 == min_dist, empty_spots)
                                chosen_spot = rand(best_spots)
                            end
                        end

                        if chosen_spot !== nothing
                            agent.biomass -= NEWBORN_BIOMASS 
                            push!(newborn_spots, (chosen_spot, NEWBORN_BIOMASS))
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

    # Spawn new cells
    for (pos, b) in newborn_spots
        if isempty(pos, model)
            if model.is_pcd_plus
                add_agent!(pos, PCDPlusCell, model, true, false, false, 0.0, 0.0, false, false, false, 0.0, b, true)
            else
                add_agent!(pos, PCDMinusCell, model, true, false, 0.0, false, false, 0.0, b)
            end
        end
    end

    # --- PDE Diffusion (Implicit Crank-Nicolson) ---
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
# --- UTILITY & PLOTTING ---
# ==========================================

function get_total_antifungal(model)
    env_mass = sum(model.ANTIFUNGAL_layer)
    agent_mass = sum(a.bound_ANTIFUNGAL for a in allagents(model))
    return env_mass + agent_mass
end

function generate_subplots(model, title_prefix::String, step::Int)
    alive_agents = [a for a in allagents(model) if a.alive]
    
    # Only map cells as healthy if they aren't stressed or apoptotic
    healthy_x = Float64[a.pos[1] for a in alive_agents if !a.is_stressed && !is_apoptotic(a)]
    healthy_y = Float64[a.pos[2] for a in alive_agents if !a.is_stressed && !is_apoptotic(a)]

    stressed_x = Float64[a.pos[1] for a in alive_agents if a.is_stressed && !is_apoptotic(a)]
    stressed_y = Float64[a.pos[2] for a in alive_agents if a.is_stressed && !is_apoptotic(a)]

    apoptotic_x = Float64[a.pos[1] for a in alive_agents if is_apoptotic(a)]
    apoptotic_y = Float64[a.pos[2] for a in alive_agents if is_apoptotic(a)]

    dead_apoptosis_x = Float64[a.pos[1] for a in allagents(model) if is_dead_apoptosis(a)]
    dead_apoptosis_y = Float64[a.pos[2] for a in allagents(model) if is_dead_apoptosis(a)]

    dead_necrosis_x = Float64[a.pos[1] for a in allagents(model) if a.dead_necrosis]
    dead_necrosis_y = Float64[a.pos[2] for a in allagents(model) if a.dead_necrosis]

    dead_starved_x = Float64[a.pos[1] for a in allagents(model) if a.dead_starvation]
    dead_starved_y = Float64[a.pos[2] for a in allagents(model) if a.dead_starvation]

    num_live = length(alive_agents)
    
    n_plot = copy(model.nutrient_layer)
    n_plot[1, 1] = INIT_NUTRIENT_LEVEL; n_plot[1, 2] = 0.0

    f_plot = copy(model.ANTIFUNGAL_layer)
    f_plot[1, 1] = INIT_ANTIFUNGAL_LEVEL; f_plot[1, 2] = 0.0

    p1 = heatmap(n_plot, title="$title_prefix Nutrients (Step $step)", 
                 color=:viridis, clims=(0, INIT_NUTRIENT_LEVEL), aspect_ratio=:equal)

    p2 = heatmap(f_plot, title="$title_prefix Antifungal", 
                 color=:ice, clims=(0, INIT_ANTIFUNGAL_LEVEL), aspect_ratio=:equal)

    p3 = plot(title="$title_prefix Live Cells: $num_live", 
              xlims=(1, GRID_SIZE_PX), ylims=(1, GRID_SIZE_PX), 
              aspect_ratio=:equal, legend=:topright)

    healthy_color = model.is_pcd_plus ? :blue : :red
    healthy_label = model.is_pcd_plus ? "Healthy (PCD+)" : "Healthy (PCD-)"

    if !isempty(healthy_x)
        scatter!(p3, healthy_x, healthy_y, label=healthy_label, color=healthy_color, markersize=2, markerstrokewidth=0)
    end
    if !isempty(stressed_x)
        scatter!(p3, stressed_x, stressed_y, label="Stressed (ROS+)", color=:yellow, markersize=2.2, markerstrokewidth=0)
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

    ys = [a.pos[2] for a in allagents(model)]
    slice_y = isempty(ys) ? GRID_SIZE_PX ÷ 2 : clamp(round(Int, mean(ys)), 1, GRID_SIZE_PX)
    
    n_slice = model.nutrient_layer[slice_y, :]
    f_slice = model.ANTIFUNGAL_layer[slice_y, :]
    
    p4 = plot(title="$title_prefix Profile (Y=$slice_y)", 
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
    println("Initializing strictly typed Agents.jl models (ROS Pathway Version)...")
    model_plus = initialize_model(PCDPlusCell)
    model_minus = initialize_model(PCDMinusCell)
    
    init_af_plus = get_total_antifungal(model_plus)
    init_af_minus = get_total_antifungal(model_minus)

    every_n_steps = 6 
    fps = 10 

    println("Starting dual simulation and recording frames...")
    
    anim = @animate for step in 1:SIMULATION_STEPS
        Agents.step!(model_plus, 1)
        Agents.step!(model_minus, 1)

        p1_n, p1_f, p1_c, p1_p = generate_subplots(model_plus, "PCD+", step)
        p2_n, p2_f, p2_c, p2_p = generate_subplots(model_minus, "PCD-", step)

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
    
    final_pcd_plus = count(a -> a.alive, allagents(model_plus))
    final_pcd_minus = count(a -> a.alive, allagents(model_minus))

    stressed_plus = count(a -> a.is_stressed && !is_apoptotic(a), allagents(model_plus))
    stressed_minus = count(a -> a.is_stressed && !is_apoptotic(a), allagents(model_minus))

    apop_plus = count(is_dead_apoptosis, allagents(model_plus))
    necro_plus = count(a -> a.dead_necrosis, allagents(model_plus))
    starved_plus = count(a -> a.dead_starvation, allagents(model_plus))

    apop_minus = count(is_dead_apoptosis, allagents(model_minus))
    necro_minus = count(a -> a.dead_necrosis, allagents(model_minus))
    starved_minus = count(a -> a.dead_starvation, allagents(model_minus))

    println("\n==========================================")
    println("--- INDEPENDENT POPULATION REPORT ---")
    println("==========================================")
    println("Conservation of Antifungal (Mass Check):")
    println("  PCD+ Initial : $(round(init_af_plus, digits=2))  | Final : $(round(final_af_plus, digits=2))")
    println("  PCD- Initial : $(round(init_af_minus, digits=2))  | Final : $(round(final_af_minus, digits=2))")
    println("------------------------------------------")
    println("Final Live Cells:")
    println("  PCD+ Env : $final_pcd_plus (Currently Stressed/ROS+: $stressed_plus)")
    println("  PCD- Env : $final_pcd_minus (Currently Stressed/ROS+: $stressed_minus)")
    println("------------------------------------------")
    println("Mortality Breakdown:")
    println("  PCD+ Dead (Antifungal) : $(apop_plus + necro_plus)  (Apop: $apop_plus, Necro: $necro_plus)")
    println("  PCD+ Dead (Starvation) : $starved_plus")
    println("  PCD- Dead (Antifungal) : $(apop_minus + necro_minus)  (Apop: $apop_minus, Necro: $necro_minus)")
    println("  PCD- Dead (Starvation) : $starved_minus")
    println("==========================================\n")
end

main()