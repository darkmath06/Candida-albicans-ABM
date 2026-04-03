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

const INIT_NUTRIENT_LEVEL = 5
const INIT_ANTIFUNGAL_LEVEL = 45.0 # Lowered slightly so the shield effect can trigger before extinction

const DIFFUSION_NUTRIENT = 0.2
const DIFFUSION_ANTIFUNGAL = 0.2
const DIFFUSION_SUBSTEPS = 10 # Substeps added to fix the checkerboard numerical instability

const MU_MAX = 0.2            # Maximum specific growth rate (mu_max)
const MONOD_KS = 5.0          # Half-velocity constant for Monod kinetics
const MAINTENANCE_COEFF = 0.015 # Maintenance coefficient (m)
const YIELD_TRUE = 0.39       # True growth yield (Y)
const YIELD_ENDOGENOUS = 0.8  # Efficiency of cannibalizing own biomass for energy
const DIVISION_BIOMASS = 2.0
const STARVATION_BIOMASS = 0.1

const PUSH_PROBABILITY = 0.02  # Probability to mechanically push when locally trapped
const MAX_PUSH_RADIUS = 10    # Max radius a cell can shove others to divide

const MAX_ANTIFUNGAL_BINDING = 120.0 # Increased from 50 so dead cells are stronger sponges
const BINDING_RATE_ANTIFUNGAL = 4.0  # Doubled so dead cells absorb faster to protect the colony

const PCD_PLUS_DEATH_THRESHOLD = 35.0 # Raised from 30.0 to give them a fairer chance to survive
const PCD_PLUS_DEATH_PROB = 0.30

const PCD_MINUS_DEATH_THRESHOLD = 40.0 # Lowered from 50.0 so they actually trigger necrosis

const PCD_PLUS_APOPTOSIS_START_TIME = 8.0    # Hours of exposure before apoptosis can trigger (PCD+ ONLY)
const PCD_PLUS_APOPTOSIS_DURATION = 2.0      # Hours the apoptosis process takes before death (PCD+ ONLY)
const NECROSIS_START_TIME = 24.0             # Hours of exposure resulting in guaranteed necrotic death

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
end

mutable struct PCDMinus <: CellAgent
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
end

PCDPlus(x::Int, y::Int; biomass=1.0) = PCDPlus(x, y, true, false, 0.0, 0.0, false, false, false, 0.0, biomass)
PCDMinus(x::Int, y::Int; biomass=1.0) = PCDMinus(x, y, true, false, 0.0, 0.0, false, false, false, 0.0, biomass)

# ------------------------------------------
# --- ANTIFUNGAL EXPOSURE LOGIC ---
# ------------------------------------------

function step_ANTIFUNGAL!(agent::PCDPlus, local_ANTIFUNGAL::Float64)
    if !agent.alive
        return
    end

    if local_ANTIFUNGAL >= PCD_PLUS_DEATH_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    else
        # Slowly decay exposure memory instead of instantly resetting it to zero
        agent.ANTIFUNGAL_exposure_time = max(0.0, agent.ANTIFUNGAL_exposure_time - TIME_STEP_DT)
    end

    if agent.is_apoptotic
        agent.apoptosis_timer += TIME_STEP_DT
        if agent.apoptosis_timer >= PCD_PLUS_APOPTOSIS_DURATION
            agent.alive = false         
            agent.is_apoptotic = false  
            agent.dead_apoptosis = true 
        end
    else
        if agent.ANTIFUNGAL_exposure_time >= NECROSIS_START_TIME
            agent.alive = false
            agent.dead_necrosis = true
        elseif agent.ANTIFUNGAL_exposure_time >= PCD_PLUS_APOPTOSIS_START_TIME
            # Probabilistic death rate over time (removes the sudden instant "dice" scatter pattern)
            prob_per_step = PCD_PLUS_DEATH_PROB * (TIME_STEP_DT / 2.0)
            if rand() < prob_per_step
                agent.is_apoptotic = true 
            end
        end
    end
end

function step_ANTIFUNGAL!(agent::PCDMinus, local_ANTIFUNGAL::Float64)
    if !agent.alive
        return
    end

    if local_ANTIFUNGAL >= PCD_MINUS_DEATH_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    else
        agent.ANTIFUNGAL_exposure_time = max(0.0, agent.ANTIFUNGAL_exposure_time - TIME_STEP_DT)
    end

    # Strictly Necrosis only for PCD-
    if agent.ANTIFUNGAL_exposure_time >= NECROSIS_START_TIME
        agent.alive = false
        agent.dead_necrosis = true
    end
end

# ==========================================
# --- MODEL ---
# ==========================================

mutable struct PetriDishModel
    size::Int
    nutrient_layer::Matrix{Float64}
    ANTIFUNGAL_layer::Matrix{Float64}
    laplacian_kernel::Matrix{Float64}
    occupied_layer::Matrix{Bool}
    agents::Vector{CellAgent}
end

function PetriDishModel(AgentType::Type)
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

    agents = CellAgent[]

    for i in 1:INITIAL_CELLS
        x = rand(1:size)
        y = rand(1:size)

        while occupied_layer[y, x]
            x = rand(1:size)
            y = rand(1:size)
        end
        occupied_layer[y, x] = true

        push!(agents, AgentType(x, y))
    end

    return PetriDishModel(size, nutrient_layer, ANTIFUNGAL_layer, laplacian_kernel, occupied_layer, agents)
end

function step_environment!(model::PetriDishModel)
    new_offspring = CellAgent[]

    # Randomize agent update order to prevent directional growth bias
    for agent in shuffle(model.agents)

        if agent.alive

            local_f = model.ANTIFUNGAL_layer[agent.y, agent.x]
            step_ANTIFUNGAL!(agent, local_f)

            if agent.alive
                local_n = model.nutrient_layer[agent.y, agent.x]
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

                if agent.is_apoptotic
                    intake = min(maintenance_cost, local_n)
                    model.nutrient_layer[agent.y, agent.x] -= intake
                    agent.biomass -= (maintenance_cost - intake) 

                    if agent.biomass <= STARVATION_BIOMASS
                        agent.alive = false
                        agent.is_apoptotic = false
                        agent.dead_apoptosis = true
                    end
                else
                    mu = MU_MAX * (local_n / (MONOD_KS + local_n))
                    growth_demand_biomass = mu * agent.biomass * TIME_STEP_DT
                    nutrient_for_growth = growth_demand_biomass / YIELD_TRUE
                    
                    desired_intake = nutrient_for_growth + maintenance_cost
                    actual_intake = min(desired_intake, local_n)
                    
                    model.nutrient_layer[agent.y, agent.x] -= actual_intake

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
                            # Space available locally -> divide easily
                            chosen_spot = rand(local_candidates)
                        else
                            # 2. Local density is high (no immediate empty space).
                            # Cell only has a probability to successfully shove/push.
                            if rand() < PUSH_PROBABILITY
                                for r in 2:MAX_PUSH_RADIUS 
                                    push_candidates = Tuple{Int, Int, Float64}[]
                                    for dx in -r:r, dy in -r:r
                                        # Only check the perimeter of the current radius ring
                                        if max(abs(dx), abs(dy)) != r; continue; end
                                        
                                        nx, ny = agent.x + dx, agent.y + dy
                                        if 1 <= nx <= model.size && 1 <= ny <= model.size
                                            if !model.occupied_layer[ny, nx]
                                                push!(push_candidates, (nx, ny, Float64(dx^2 + dy^2)))
                                            end
                                        end
                                    end
                                    
                                    if !isempty(push_candidates)
                                        # Find strictly the closest spot mathematically in this ring
                                        min_dist = minimum(c[3] for c in push_candidates)
                                        best_spots = [(c[1], c[2]) for c in push_candidates if c[3] == min_dist]
                                        chosen_spot = rand(best_spots)
                                        break
                                    end
                                end
                            end
                        end

                        if chosen_spot !== nothing
                            agent.biomass /= 2.0
                            new_x, new_y = chosen_spot
                            
                            model.occupied_layer[new_y, new_x] = true

                            if isa(agent, PCDPlus)
                                push!(new_offspring, PCDPlus(new_x, new_y; biomass=agent.biomass))
                            else
                                push!(new_offspring, PCDMinus(new_x, new_y; biomass=agent.biomass))
                            end
                        end
                    end
                end
            end
        end

        # Absorption loop - stores taken antifungal inside dead agent
        if !agent.alive && agent.bound_ANTIFUNGAL < MAX_ANTIFUNGAL_BINDING
            capacity_remaining = MAX_ANTIFUNGAL_BINDING - agent.bound_ANTIFUNGAL
            want_to_absorb = min(BINDING_RATE_ANTIFUNGAL * TIME_STEP_DT, capacity_remaining)
            actual_absorption = min(want_to_absorb, model.ANTIFUNGAL_layer[agent.y, agent.x])

            agent.bound_ANTIFUNGAL += actual_absorption
            model.ANTIFUNGAL_layer[agent.y, agent.x] -= actual_absorption
        end
    end

    append!(model.agents, new_offspring)

    # Sub-stepped Diffusion (Prevents checkerboard explicit instability)
    dt_diff = TIME_STEP_DT / DIFFUSION_SUBSTEPS
    for _ in 1:DIFFUSION_SUBSTEPS
        # Changed "reflect" to "replicate" to enforce mass-conserving Zero-Flux boundaries
        laplacian_n = imfilter(model.nutrient_layer, centered(model.laplacian_kernel), "replicate")
        model.nutrient_layer .+= DIFFUSION_NUTRIENT .* laplacian_n .* dt_diff
        model.nutrient_layer .= max.(model.nutrient_layer, 0.0)

        laplacian_f = imfilter(model.ANTIFUNGAL_layer, centered(model.laplacian_kernel), "replicate")
        model.ANTIFUNGAL_layer .+= DIFFUSION_ANTIFUNGAL .* laplacian_f .* dt_diff
        model.ANTIFUNGAL_layer .= max.(model.ANTIFUNGAL_layer, 0.0)
    end
end

# Helper to verify Antifungal Conservation
function get_total_antifungal(model::PetriDishModel)
    env_mass = sum(model.ANTIFUNGAL_layer)
    agent_mass = sum(a.bound_ANTIFUNGAL for a in model.agents)
    return env_mass + agent_mass
end

# Generates side-by-side plots modularly
function generate_subplots(model::PetriDishModel, title_prefix::String, step::Int)
    alive_agents = [a for a in model.agents if a.alive]
    
    healthy_x = Float64[a.x for a in alive_agents if !a.is_apoptotic]
    healthy_y = Float64[a.y for a in alive_agents if !a.is_apoptotic]
    
    apoptotic_x = Float64[a.x for a in alive_agents if a.is_apoptotic]
    apoptotic_y = Float64[a.y for a in alive_agents if a.is_apoptotic]

    dead_apoptosis_x = Float64[a.x for a in model.agents if a.dead_apoptosis]
    dead_apoptosis_y = Float64[a.y for a in model.agents if a.dead_apoptosis]

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

    healthy_color = (length(model.agents) > 0 && isa(model.agents[1], PCDPlus)) ? :blue : :red
    healthy_label = (length(model.agents) > 0 && isa(model.agents[1], PCDPlus)) ? "PCD+" : "PCD-"

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

    return p1, p2, p3
end

function main()
    println("Initializing independent simulations...")
    
    # Initialize the two independent environments
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

        # Generate subplots for both environments
        p1_n, p1_f, p1_c = generate_subplots(model_plus, "PCD+", step)
        p2_n, p2_f, p2_c = generate_subplots(model_minus, "PCD-", step)

        # Compose 2x3 Layout (Row 1: PCD+, Row 2: PCD-)
        plot(p1_n, p1_f, p1_c, p2_n, p2_f, p2_c, layout=(2, 3), size=(1200, 800))

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

    println("\n==========================================")
    println("--- INDEPENDENT POPULATION REPORT ---")
    println("==========================================")
    println("Final Live Cells:")
    println("  PCD+ Env : $final_pcd_plus")
    println("  PCD- Env : $final_pcd_minus")
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