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
const TIME_STEP_MINS = TIME_STEP_DT * 60.0

const INIT_NUTRIENT_LEVEL = 10
const INIT_ANTIFUNGAL_LEVEL = 0.

const DIFFUSION_NUTRIENT = 0.2
const DIFFUSION_ANTIFUNGAL = 0.2
const DIFFUSION_ITERATIONS = 15

const MU_MAX = 0.34            
const MONOD_KS = 5.0          
const MAINTENANCE_COEFF = 0.015 
const YIELD_TRUE = 0.39       
const YIELD_ENDOGENOUS = 0.8  
const NEWBORN_BIOMASS = 1.0   
const DIVISION_BIOMASS = 2.0  
const STARVATION_BIOMASS = 0.1

const PUSH_PROBABILITY = 0.02 
const MAX_PUSH_RADIUS = 10    

const MAX_ANTIFUNGAL_BINDING_LIVE = 0.42  
const MAX_ANTIFUNGAL_BINDING_DEAD = 2.55 
const K_ON_ANTIFUNGAL = 0.01         
const K_OFF_ANTIFUNGAL = 0.005       

# --- NEW BINDING KINETICS PARAMETERS ---
const K_MAX_BINDING = 1.0            # Max rate for active strong binding (per min)
const K_D_BINDING = 0.64             # Half-max concentration (µg/mL)
const LAG_DOSE_THRESHOLD = 0.25      # Below this, strong binding is delayed
const LAG_TIME_MINS = 30.0           # Minutes before strong binding starts at low doses

# --- NEW STATE THRESHOLDS ---
const STRESS_STRONG_BOND_THRESHOLD = 20.0  # Accumulation required to trigger ROS stress
const RUPTURE_STRONG_BOND_THRESHOLD = 150.0 # Accumulation causing catastrophic cell lysis
const ROS_CRITICAL_LEVEL = 3.3             # Hours of ROS stress before apoptosis/death begins
const APOPTOSIS_DURATION = 2.0             # Hours the apoptosis process takes before death (PCD+ ONLY)
const APOPTOSIS_RATE = 0.15                # Hourly probability of entering apoptosis (PCD+)
const SLOW_DEATH_RATE = 0.1                # Hourly probability of slow death from ROS (PCD-)

const APOPTOSIS_RECOVERY_PROB = 0.15       
const RETAIN_DIVISION_PROB = 0.0         
const STRESS_CLEARANCE_HALF_LIFE = 2.0       
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
    ros_accumulation::Float64
    dead_apoptosis::Bool
    dead_necrosis::Bool
    dead_slow::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
    can_divide::Bool       
    has_recovered::Bool    
    strong_bonds::Float64  # NEW: Tracks irreversible lethal binding
    exposure_time::Float64 # NEW: Tracks total minutes exposed
end

mutable struct PCDMinus <: CellAgent
    x::Int
    y::Int
    alive::Bool
    ros_accumulation::Float64
    dead_necrosis::Bool
    dead_slow::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
    strong_bonds::Float64  # NEW
    exposure_time::Float64 # NEW
end

PCDPlus(x::Int, y::Int; biomass=1.0) = PCDPlus(x, y, true, false, 0.0, 0.0, false, false, false, false, 0.0, biomass, true, false, 0.0, 0.0)
PCDMinus(x::Int, y::Int; biomass=1.0) = PCDMinus(x, y, true, 0.0, false, false, false, 0.0, biomass, 0.0, 0.0)

# --- Accessors for Type-Stable Generic Operations ---
is_apoptotic(a::PCDPlus) = a.is_apoptotic
is_apoptotic(a::PCDMinus) = false

can_divide(a::PCDPlus) = a.can_divide
can_divide(a::PCDMinus) = true

has_recovered(a::PCDPlus) = a.has_recovered
has_recovered(a::PCDMinus) = false

is_dead_apoptosis(a::PCDPlus) = a.dead_apoptosis
is_dead_apoptosis(a::PCDMinus) = false

is_dead_necrosis(a::CellAgent) = a.dead_necrosis
is_dead_slow(a::CellAgent) = a.dead_slow

set_dead_apoptosis!(a::PCDPlus) = begin
    a.alive = false
    a.is_apoptotic = false
    a.dead_apoptosis = true
end
set_dead_apoptosis!(a::PCDMinus) = nothing

# ------------------------------------------
# --- DYNAMIC KINETICS EXPOSURE LOGIC ---
# ------------------------------------------

function calculate_binding_kinetics(agent::CellAgent, local_ANTIFUNGAL::Float64)
    # Track exposure time
    if local_ANTIFUNGAL > 0.01
        agent.exposure_time += TIME_STEP_MINS
    else
        agent.exposure_time = max(0.0, agent.exposure_time - TIME_STEP_MINS * 0.5)
    end

    # Lag phase at very low doses
    if local_ANTIFUNGAL <= LAG_DOSE_THRESHOLD && agent.exposure_time < LAG_TIME_MINS
        return 0.0
    end

    # Saturable active transport (Strong Binding)
    active_transport_rate = (K_MAX_BINDING * local_ANTIFUNGAL) / (K_D_BINDING + local_ANTIFUNGAL)
    
    # Non-linear membrane tearing (Takes over massively at High Doses like 16 µg/mL)
    membrane_tearing_rate = (local_ANTIFUNGAL / 8.0)^3
    
    return active_transport_rate + membrane_tearing_rate
end

function step_ANTIFUNGAL!(agent::PCDPlus, local_ANTIFUNGAL::Float64)
    if !agent.alive; return; end

    binding_rate = calculate_binding_kinetics(agent, local_ANTIFUNGAL)
    agent.strong_bonds += binding_rate * TIME_STEP_MINS

    # 1. Rupture Check (High Dose Direct Necrosis)
    if agent.strong_bonds >= RUPTURE_STRONG_BOND_THRESHOLD
        agent.alive = false
        agent.dead_necrosis = true
        return
    end

    # 2. Stress Check (ROS Accumulation)
    if agent.strong_bonds >= STRESS_STRONG_BOND_THRESHOLD
        agent.ros_accumulation += TIME_STEP_DT
    else
        agent.ros_accumulation *= exp(-STRESS_DECAY_RATE * TIME_STEP_DT)
        if agent.ros_accumulation < 0.01; agent.ros_accumulation = 0.0; end
    end

    # 3. Apoptosis Execution
    if agent.is_apoptotic
        # Resuscitation
        if agent.strong_bonds < STRESS_STRONG_BOND_THRESHOLD
            prob_recover = 1.0 - exp(-APOPTOSIS_RECOVERY_PROB * TIME_STEP_DT)
            if rand() < prob_recover
                agent.is_apoptotic = false
                agent.apoptosis_timer = 0.0
                agent.has_recovered = true
                agent.can_divide = rand() < RETAIN_DIVISION_PROB
                return 
            end
        end

        agent.apoptosis_timer += TIME_STEP_DT
        if agent.apoptosis_timer >= APOPTOSIS_DURATION
            set_dead_apoptosis!(agent)
        end
    else
        if agent.ros_accumulation >= ROS_CRITICAL_LEVEL
            prob_apop = 1.0 - exp(-APOPTOSIS_RATE * TIME_STEP_DT)
            if rand() < prob_apop
                agent.is_apoptotic = true 
            end
        end
    end
end

function step_ANTIFUNGAL!(agent::PCDMinus, local_ANTIFUNGAL::Float64)
    if !agent.alive; return; end

    binding_rate = calculate_binding_kinetics(agent, local_ANTIFUNGAL)
    agent.strong_bonds += binding_rate * TIME_STEP_MINS

    # 1. Rupture Check (Exact same necrotic fate as PCD+ at High Doses)
    if agent.strong_bonds >= RUPTURE_STRONG_BOND_THRESHOLD
        agent.alive = false
        agent.dead_necrosis = true
        return
    end

    # 2. Stress Check (ROS Accumulation)
    if agent.strong_bonds >= STRESS_STRONG_BOND_THRESHOLD
        agent.ros_accumulation += TIME_STEP_DT
    else
        agent.ros_accumulation *= exp(-STRESS_DECAY_RATE * TIME_STEP_DT)
        if agent.ros_accumulation < 0.01; agent.ros_accumulation = 0.0; end
    end

    # 3. Slow Decline Check (Cannot undergo apoptosis)
    if agent.ros_accumulation >= ROS_CRITICAL_LEVEL
        prob_slow_death = 1.0 - exp(-SLOW_DEATH_RATE * TIME_STEP_DT)
        if rand() < prob_slow_death
            agent.alive = false
            agent.dead_slow = true
        end
    end
end

# ==========================================
# --- MODEL ---
# ==========================================

mutable struct PetriDishModel{T <: CellAgent}
    size::Int
    nutrient_layer::Matrix{Float64}
    ANTIFUNGAL_layer::Matrix{Float64}
    laplacian_kernel::Matrix{Float64}
    neighbor_kernel::Matrix{Float64}
    occupied_layer::Matrix{Bool}
    agents::Vector{T} 
end

function PetriDishModel(AgentType::Type{T}) where {T <: CellAgent}
    size = GRID_SIZE_PX
    nutrient_layer = fill(INIT_NUTRIENT_LEVEL, size, size)
    ANTIFUNGAL_layer = fill(INIT_ANTIFUNGAL_LEVEL, size, size)
    occupied_layer = zeros(Bool, size, size)

    laplacian_kernel = Float64[
        1/6   2/3   1/6;
        2/3 -10/3   2/3;
        1/6   2/3   1/6
    ]

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
    num_agents = length(model.agents)
    n_demands = zeros(Float64, num_agents)
    f_demands = zeros(Float64, num_agents)
    f_releases = zeros(Float64, num_agents)

    grid_n_demand = zeros(Float64, model.size, model.size)
    grid_f_demand = zeros(Float64, model.size, model.size)

    # --- PASS 1: Calculate Demands ---
    for (i, agent) in enumerate(model.agents)
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
        
        local_f = model.ANTIFUNGAL_layer[agent.y, agent.x]
        bound_f = agent.bound_ANTIFUNGAL
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

    # --- PASS 2: Allocate & Update ---
    for (i, agent) in enumerate(model.agents)
        if agent.alive
            local_f = model.ANTIFUNGAL_layer[agent.y, agent.x]
            step_ANTIFUNGAL!(agent, local_f)
        end

        if agent.alive && n_demands[i] > 0
            available_n = model.nutrient_layer[agent.y, agent.x]
            total_demand_n = grid_n_demand[agent.y, agent.x]
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
                        agent.biomass = DIVISION_BIOMASS 
                    else
                        chosen_spot = nothing
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
                            agent.biomass -= NEWBORN_BIOMASS 
                            new_x, new_y = chosen_spot
                            model.occupied_layer[new_y, new_x] = true
                            push!(new_offspring, T(new_x, new_y; biomass=NEWBORN_BIOMASS))
                        end
                    end
                end
            end
        end

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

    # --- Implicit Crank-Nicolson PDE Solver ---
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

function get_total_antifungal(model::PetriDishModel)
    env_mass = sum(model.ANTIFUNGAL_layer)
    agent_mass = sum(a.bound_ANTIFUNGAL for a in model.agents)
    return env_mass + agent_mass
end

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

    dead_necrosis_x = Float64[a.x for a in model.agents if is_dead_necrosis(a)]
    dead_necrosis_y = Float64[a.y for a in model.agents if is_dead_necrosis(a)]

    dead_slow_x = Float64[a.x for a in model.agents if is_dead_slow(a)]
    dead_slow_y = Float64[a.y for a in model.agents if is_dead_slow(a)]

    dead_starved_x = Float64[a.x for a in model.agents if a.dead_starvation]
    dead_starved_y = Float64[a.y for a in model.agents if a.dead_starvation]

    num_live = length(alive_agents)
    
    n_plot = copy(model.nutrient_layer); n_plot[1, 1] = INIT_NUTRIENT_LEVEL; n_plot[1, 2] = 0.0
    f_plot = copy(model.ANTIFUNGAL_layer); f_plot[1, 1] = INIT_ANTIFUNGAL_LEVEL; f_plot[1, 2] = 0.0

    p1 = heatmap(n_plot, title="$title_prefix Nutrients (Step $step)", color=:viridis, clims=(0, INIT_NUTRIENT_LEVEL), aspect_ratio=:equal)
    p2 = heatmap(f_plot, title="$title_prefix Antifungal", color=:ice, clims=(0, INIT_ANTIFUNGAL_LEVEL), aspect_ratio=:equal)

    p3 = plot(title="$title_prefix Live Cells: $num_live", xlims=(1, model.size), ylims=(1, model.size), aspect_ratio=:equal, legend=:topright)

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
    if !isempty(dead_slow_x)
        scatter!(p3, dead_slow_x, dead_slow_y, label="Dead (Slow)", color=:brown, markersize=3, markerstrokewidth=0)
    end
    if !isempty(dead_starved_x)
        scatter!(p3, dead_starved_x, dead_starved_y, label="Dead (Starve)", color=:magenta, markersize=3, markerstrokewidth=0)
    end

    ys = [a.y for a in model.agents]
    slice_y = isempty(ys) ? model.size ÷ 2 : clamp(round(Int, mean(ys)), 1, model.size)
    
    n_slice = model.nutrient_layer[slice_y, :]
    f_slice = model.ANTIFUNGAL_layer[slice_y, :]
    
    p4 = plot(title="$title_prefix Profile (Y=$slice_y)", xlims=(1, model.size), ylims=(0, INIT_NUTRIENT_LEVEL), legend=:topright, xlabel="X Position", ylabel="Concentration", titlefontsize=10)
    
    plot!(p4, 1:model.size, n_slice, label="Nutrients", color=:green, linewidth=2)
    plot!(p4, 1:model.size, f_slice, label="Antifungal", color=:blue, linewidth=2)
    
    # Updated dose reference lines
    hline!(p4, [4.0], label="Active Dose (~4 µg)", color=:orange, linestyle=:dash, linewidth=2)
    hline!(p4, [16.0], label="Tearing Dose (~16 µg)", color=:red, linestyle=:dash, linewidth=2)

    return p1, p2, p3, p4
end

function main()
    println("Initializing independent simulations...")
    
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

    final_af_plus = get_total_antifungal(model_plus)
    final_af_minus = get_total_antifungal(model_minus)
    
    final_pcd_plus = count(a -> a.alive, model_plus.agents)
    final_pcd_minus = count(a -> a.alive, model_minus.agents)

    apop_plus = count(is_dead_apoptosis, model_plus.agents)
    necro_plus = count(is_dead_necrosis, model_plus.agents)
    slow_plus = count(is_dead_slow, model_plus.agents)
    starved_plus = count(a -> a.dead_starvation, model_plus.agents)

    apop_minus = count(is_dead_apoptosis, model_minus.agents)
    necro_minus = count(is_dead_necrosis, model_minus.agents)
    slow_minus = count(is_dead_slow, model_minus.agents)
    starved_minus = count(a -> a.dead_starvation, model_minus.agents)

    println("\n==========================================")
    println("--- INDEPENDENT POPULATION REPORT ---")
    println("==========================================")
    println("Final Live Cells:")
    println("  PCD+ Env : $final_pcd_plus")
    println("  PCD- Env : $final_pcd_minus")
    println("------------------------------------------")
    println("Mortality Breakdown:")
    println("  PCD+ (Apop: $apop_plus, Necro: $necro_plus, Slow: $slow_plus, Starve: $starved_plus)")
    println("  PCD- (Apop: $apop_minus, Necro: $necro_minus, Slow: $slow_minus, Starve: $starved_minus)")
    println("------------------------------------------")
    println("Absolute Fitness (Final / Initial):")
    println("  PCD+ : $(round(final_pcd_plus / INITIAL_CELLS, digits=3))")
    println("  PCD- : $(round(final_pcd_minus / INITIAL_CELLS, digits=3))")
    println("==========================================")
end

main()