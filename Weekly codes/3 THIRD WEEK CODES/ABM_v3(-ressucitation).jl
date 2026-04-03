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
const INITIAL_CELLS = 1
const SIMULATION_STEPS = 450
const TIME_STEP_DT = 1.0 / 3.0 # Assuming 1 step = 20 minutes of biological time

# --- Environment Levels ---
const INIT_NUTRIENT_LEVEL = 10
const INIT_ANTIFUNGAL_LEVEL = 0.5

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
const STARVATION_BIOMASS = 0.1 # Unused now that cells don't shrink, kept for compatibility
const APOPTOSIS_NUTRIENT_RELEASE_FRAC = 0.85 # Fraction of biomass returned to environment upon apoptotic death

# --- Mechanics & Space ---
const PUSH_PROBABILITY = 0.02 # Probability to mechanically push when locally trapped
const MAX_PUSH_RADIUS = 10    # Max radius a cell can shove others to divide

# --- Antifungal Binding Kinetics ---
const MAX_ANTIFUNGAL_BINDING_LIVE = 0.42  # Capacity for intact, living cells
const MAX_ANTIFUNGAL_BINDING_DEAD = 2.55  # Capacity for dead cells (sponge effect)
const K_ON_ANTIFUNGAL = 0.01              # Adsorption rate constant
const K_OFF_ANTIFUNGAL = 0.005            # Desorption rate constant

# --- Stress, Apoptosis & Necrosis ---
const ANTIFUNGAL_DAMAGE_THRESHOLD = 0.05      # Threshold to start accumulating damage
const STRESS_START_TIME = 3.30               # Hours of exposure before death risks begin
const APOPTOSIS_DURATION = 2.0               # Hours the apoptosis process takes
const ASSAY_DURATION_HOURS = 24.0            # Calibration time for dose-response percentages

# ==========================================
# --- AGENT TYPES ---
# ==========================================

# Using the Agents.jl macro to define agents that live on a 2D Grid
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
    can_divide::Bool       
end

@agent struct PCDMinusCell(GridAgent{2})
    alive::Bool
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

    for _ in 1:INITIAL_CELLS
        if AgentType === PCDPlusCell
            add_agent_single!(PCDPlusCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, 1.0, true)
        else
            add_agent_single!(PCDMinusCell, model, true, 0.0, false, false, 0.0, 1.0)
        end
    end

    return model
end

# ==========================================
# --- ANTIFUNGAL EXPOSURE LOGIC ---
# ==========================================

# Calculates death rates using continuous mathematical dose-response curves
function get_death_rates(c::Float64)
    if c <= 0.0
        return 0.0, 0.0
    end
    
    # Empirical data points from assay table
    # AmB Concentrations
    C_vals = (0.0, 0.5, 4.0, 8.0, 16.0)
    # Apoptosis Fractions (0 to 1)
    apop_vals = (0.0, 0.15, 0.20, 0.57, 0.09)
    # Necrosis Fractions (0 to 1)
    necro_vals = (0.0, 0.09, 0.20, 0.33, 0.91)

    target_apop_frac = 0.0
    target_necro_frac = 0.0

    # Linear interpolation to exactly match experimental assay points
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
    
    # Cap total fraction slightly below 1.0 to prevent infinite hourly rates (log(0))
    target_total_frac = min(0.999, target_apop_frac + target_necro_frac)
    
    if target_total_frac <= 0.0
        return 0.0, 0.0
    end
    
    # Convert the cumulative overnight fraction into a continuous HOURLY rate
    hourly_total_rate = -log(1.0 - target_total_frac) / ASSAY_DURATION_HOURS
    
    # Split the hourly rate back into apoptosis and necrosis components
    ratio_apop = target_apop_frac / target_total_frac
    ratio_necro = target_necro_frac / target_total_frac
    
    rate_apop = hourly_total_rate * ratio_apop
    rate_necro = hourly_total_rate * ratio_necro
    
    return rate_apop, rate_necro
end

function apply_stress!(agent::PCDPlusCell, local_ANTIFUNGAL::Float64, model)
    if !agent.alive; return; end

    # Damage is now strictly cumulative. It pauses if below threshold, but never decays.
    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    end

    if agent.is_apoptotic
        # Apoptosis is now a point of no return. No recovery/anastasis possible.
        agent.apoptosis_timer += TIME_STEP_DT
        if agent.apoptosis_timer >= APOPTOSIS_DURATION
            set_dead_apoptosis!(agent)
            
            # --- NUTRIENT RECYCLING ---
            x, y = agent.pos
            model.nutrient_layer[y, x] += agent.biomass * APOPTOSIS_NUTRIENT_RELEASE_FRAC
            agent.biomass = 0.0 
        end
    else
        if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME
            # SINGLE ROLL logic based on dose
            rate_apop, rate_necro = get_death_rates(local_ANTIFUNGAL)
            total_rate = rate_apop + rate_necro
            
            if total_rate > 0
                prob_death = 1.0 - exp(-total_rate * TIME_STEP_DT)
                if rand() < prob_death
                    # If the cell is dying, what type of death is it?
                    prob_apop_given_death = rate_apop / total_rate
                    if rand() < prob_apop_given_death
                        agent.is_apoptotic = true 
                    else
                        agent.alive = false
                        agent.dead_necrosis = true
                    end
                end
            end
        end
    end
end

function apply_stress!(agent::PCDMinusCell, local_ANTIFUNGAL::Float64, model)
    if !agent.alive; return; end

    # Damage is purely cumulative
    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    end

    if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME
        rate_apop, rate_necro = get_death_rates(local_ANTIFUNGAL)
        
        # PCD- cells lack the apoptosis machinery. They completely ignore rate_apop.
        # They only die if they succumb to necrosis.
        if rate_necro > 0
            prob_death = 1.0 - exp(-rate_necro * TIME_STEP_DT)
            if rand() < prob_death
                agent.alive = false
                agent.dead_necrosis = true
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
                # Cell only grows if it exceeds maintenance cost. 
                if actual_intake >= maintenance_cost
                    agent.biomass += (actual_intake - maintenance_cost) * YIELD_TRUE
                end

                if agent.biomass >= DIVISION_BIOMASS
                    if !can_divide(agent)
                        agent.biomass = DIVISION_BIOMASS
                    else
                        # Budding/Pushing Resolution 
                        chosen_spot = nothing
                        
                        # Radius 1 search
                        immediate_hood = collect(nearby_positions(agent.pos, model, 1))
                        empty_immediate = filter(p -> isempty(p, model), immediate_hood)
                        
                        if !isempty(empty_immediate)
                            chosen_spot = rand(empty_immediate)
                        elseif rand() < PUSH_PROBABILITY
                            # Search the entire neighborhood block at once
                            block = collect(nearby_positions(agent.pos, model, MAX_PUSH_RADIUS))
                            empty_spots = filter(p -> isempty(p, model), block)
                            
                            if !isempty(empty_spots)
                                # Find the absolute closest empty spot by true Euclidean distance
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

        # 3. Antifungal
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

    for (pos, b) in newborn_spots
        if isempty(pos, model)
            if model.is_pcd_plus
                add_agent!(pos, PCDPlusCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, b, true)
            else
                add_agent!(pos, PCDMinusCell, model, true, 0.0, false, false, 0.0, b)
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
    
    healthy_x = Float64[a.pos[1] for a in alive_agents if !is_apoptotic(a)]
    healthy_y = Float64[a.pos[2] for a in alive_agents if !is_apoptotic(a)]

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
    healthy_label = model.is_pcd_plus ? "PCD+" : "PCD-"

    if !isempty(healthy_x)
        scatter!(p3, healthy_x, healthy_y, label=healthy_label, color=healthy_color, markersize=2, markerstrokewidth=0)
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
    println("Initializing strictly typed Agents.jl models (No-Decay Version)...")
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
    println("  PCD+ Env : $final_pcd_plus")
    println("  PCD- Env : $final_pcd_minus")
    println("------------------------------------------")
    println("Mortality Breakdown:")
    println("  PCD+ Dead (Antifungal) : $(apop_plus + necro_plus)  (Apop: $apop_plus, Necro: $necro_plus)")
    println("  PCD+ Dead (Starvation) : $starved_plus")
    println("  PCD- Dead (Antifungal) : $(apop_minus + necro_minus)  (Apop: $apop_minus, Necro: $necro_minus)")
    println("  PCD- Dead (Starvation) : $starved_minus")
    println("------------------------------------------")
    println("Absolute Fitness (Final / Initial):")
    println("  PCD+ : $(round(final_pcd_plus / INITIAL_CELLS, digits=3))")
    println("  PCD- : $(round(final_pcd_minus / INITIAL_CELLS, digits=3))")
    println("------------------------------------------")
    println("Total Cells (Alive + Dead):")
    println("  PCD+ : $(nagents(model_plus))")
    println("  PCD- : $(nagents(model_minus))")
    println("==========================================\n")
end

main()