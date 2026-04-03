using Random
using StatsBase
using Images
using ImageFiltering
using Plots

# ==========================================
# --- GLOBAL EXPERIMENTAL PARAMETERS ---
# ==========================================

const GRID_SIZE_UM = 500
const INITIAL_CELLS = 200
const SIMULATION_STEPS = 50
const TIME_STEP_DT = 1.0 # Assuming 1 step = 1 hour of biological time

const INIT_PCD_PLUS_FRACTION = 0.50

const INIT_NUTRIENT_LEVEL = 100.0
const INIT_ANTIFUNGAL_LEVEL = 50.0

const DIFFUSION_NUTRIENT = 0.2
const DIFFUSION_ANTIFUNGAL = 0.8

const MU_MAX = 0.5            # Maximum specific growth rate (mu_max)
const MONOD_KS = 5.0          # Half-velocity constant for Monod kinetics
const MAINTENANCE_COEFF = 0.015 # Maintenance coefficient (m)
const YIELD_TRUE = 0.39       # True growth yield (Y)
const YIELD_ENDOGENOUS = 0.8  # Efficiency of cannibalizing own biomass for energy
const DIVISION_BIOMASS = 2.0
const STARVATION_BIOMASS = 0.1

const MAX_ANTIFUNGAL_BINDING = 15.0
const BINDING_RATE_ANTIFUNGAL = 2.0

const PCD_PLUS_DEATH_THRESHOLD = 30.0
const PCD_PLUS_DEATH_PROB = 0.30

const PCD_MINUS_DEATH_THRESHOLD = 50.0
# Note: PCD_MINUS_DEATH_PROB was removed as PCD- cannot undergo apoptosis

# --- NEW APOPTOSIS & NECROSIS TIMING PARAMETERS ---
const APOPTOSIS_START_TIME = 8.0    # Hours of exposure before apoptosis can trigger
const APOPTOSIS_DURATION = 2.0      # Hours the apoptosis process takes before death
const NECROSIS_START_TIME = 24.0    # Hours of exposure resulting in guaranteed necrotic death

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
    apoptosis_checked::Bool
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
    apoptosis_checked::Bool
    dead_apoptosis::Bool
    dead_necrosis::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
end

# Constructors initialized with not-apoptotic and zeroed states
PCDPlus(x::Int, y::Int; biomass=1.0) = PCDPlus(x, y, true, false, 0.0, 0.0, false, false, false, false, 0.0, biomass)
PCDMinus(x::Int, y::Int; biomass=1.0) = PCDMinus(x, y, true, false, 0.0, 0.0, false, false, false, false, 0.0, biomass)

function _process_ANTIFUNGAL!(agent::CellAgent, local_ANTIFUNGAL::Float64, death_threshold::Float64, death_prob::Float64)
    if !agent.alive
        return
    end

    # Track continuous exposure time if above threshold
    if local_ANTIFUNGAL >= death_threshold
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    else
        # Reset exposure if ANTIFUNGAL levels drop (requires continuous exposure)
        agent.ANTIFUNGAL_exposure_time = 0.0
        agent.apoptosis_checked = false # Reset so they can be checked again if exposed anew
    end

    if agent.is_apoptotic
        # Progress the apoptosis timer
        agent.apoptosis_timer += TIME_STEP_DT
        if agent.apoptosis_timer >= APOPTOSIS_DURATION
            agent.alive = false         # The cell is officially dead
            agent.is_apoptotic = false  # No longer actively dying
            agent.dead_apoptosis = true # Mark as dead via apoptosis for plotting
        end
    else
        # Check for 24h Necrosis limit
        if agent.ANTIFUNGAL_exposure_time >= NECROSIS_START_TIME
            agent.alive = false
            agent.dead_necrosis = true
        
        # Check for 8h Apoptosis trigger (only roll once per continuous exposure)
        elseif agent.ANTIFUNGAL_exposure_time >= APOPTOSIS_START_TIME && !agent.apoptosis_checked
            agent.apoptosis_checked = true
            if rand() < death_prob
                agent.is_apoptotic = true # Irreversibly committed to death pathway
            end
        end
    end
end

function step_ANTIFUNGAL!(agent::PCDPlus, local_ANTIFUNGAL::Float64)
    _process_ANTIFUNGAL!(agent, local_ANTIFUNGAL, PCD_PLUS_DEATH_THRESHOLD, PCD_PLUS_DEATH_PROB)
end

function step_ANTIFUNGAL!(agent::PCDMinus, local_ANTIFUNGAL::Float64)
    if !agent.alive
        return
    end

    # Track continuous exposure time if above threshold
    if local_ANTIFUNGAL >= PCD_MINUS_DEATH_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    else
        # Reset exposure if ANTIFUNGAL levels drop
        agent.ANTIFUNGAL_exposure_time = 0.0
    end

    # PCD- cells strictly undergo necrosis after continuous 24h limit (no apoptosis)
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

function PetriDishModel()
    size = GRID_SIZE_UM

    nutrient_layer = fill(INIT_NUTRIENT_LEVEL, size, size)
    ANTIFUNGAL_layer = fill(INIT_ANTIFUNGAL_LEVEL, size, size)
    occupied_layer = zeros(Bool, size, size)

    laplacian_kernel = Float64[
        0  1  0;
        1 -4  1;
        0  1  0
    ]

    agents = CellAgent[]

    num_pcd_plus = Int(round(INITIAL_CELLS * INIT_PCD_PLUS_FRACTION))

    for i in 1:INITIAL_CELLS
        x = rand(1:size)
        y = rand(1:size)

        # Ensure no overlap at initialization
        while occupied_layer[y, x]
            x = rand(1:size)
            y = rand(1:size)
        end
        occupied_layer[y, x] = true

        if i <= num_pcd_plus
            push!(agents, PCDPlus(x, y))
        else
            push!(agents, PCDMinus(x, y))
        end
    end

    return PetriDishModel(size, nutrient_layer, ANTIFUNGAL_layer, laplacian_kernel, occupied_layer, agents)
end

function step_environment!(model::PetriDishModel)
    new_offspring = CellAgent[]

    for agent in model.agents

        if agent.alive

            local_f = model.ANTIFUNGAL_layer[agent.y, agent.x]
            step_ANTIFUNGAL!(agent, local_f)

            if agent.alive
                local_n = model.nutrient_layer[agent.y, agent.x]
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

                if agent.is_apoptotic
                    # Apoptotic cells don't grow or divide, but the active death process requires ATP (maintenance)
                    intake = min(maintenance_cost, local_n)
                    model.nutrient_layer[agent.y, agent.x] -= intake
                    agent.biomass -= (maintenance_cost - intake) # Shrink if starved during apoptosis

                    if agent.biomass <= STARVATION_BIOMASS
                        agent.alive = false
                        agent.is_apoptotic = false
                        agent.dead_apoptosis = true
                    end

                else
                    # Healthy cells consume based on Monod kinetics and the Pirt equation
                    # 1. Calculate actual growth rate (mu) based on local nutrients
                    mu = MU_MAX * (local_n / (MONOD_KS + local_n))
                    
                    # 2. Calculate demands
                    growth_demand_biomass = mu * agent.biomass * TIME_STEP_DT
                    nutrient_for_growth = growth_demand_biomass / YIELD_TRUE
                    
                    desired_intake = nutrient_for_growth + maintenance_cost
                    actual_intake = min(desired_intake, local_n)
                    
                    # 3. Take nutrients from the environment
                    model.nutrient_layer[agent.y, agent.x] -= actual_intake

                    # 4. Allocate consumed nutrients
                    if actual_intake >= maintenance_cost
                        # Maintenance met! Leftover goes to growth.
                        leftover_for_growth = actual_intake - maintenance_cost
                        agent.biomass += leftover_for_growth * YIELD_TRUE
                    else
                        # Starving! Intake couldn't cover maintenance.
                        # The cell must cannibalize its own biomass to cover the missing energy.
                        deficit_nutrient = maintenance_cost - actual_intake
                        agent.biomass -= deficit_nutrient * YIELD_ENDOGENOUS
                    end

                    if agent.biomass <= STARVATION_BIOMASS
                        agent.alive = false
                        agent.dead_starvation = true

                    elseif agent.biomass >= DIVISION_BIOMASS
                        # Find adjacent empty spots for division (spatial exclusion rule)
                        empty_spots = Tuple{Int, Int}[]
                        for dx in -1:1, dy in -1:1
                            if dx == 0 && dy == 0; continue; end
                            nx, ny = agent.x + dx, agent.y + dy
                            if 1 <= nx <= model.size && 1 <= ny <= model.size
                                if !model.occupied_layer[ny, nx]
                                    push!(empty_spots, (nx, ny))
                                end
                            end
                        end

                        # Only divide if there is physical space available
                        if !isempty(empty_spots)
                            agent.biomass /= 2.0
                            new_x, new_y = rand(empty_spots)
                            
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

        # Dead cells (or husks) passively absorb ANTIFUNGAL but still physically occupy their grid space
        if !agent.alive && agent.bound_ANTIFUNGAL < MAX_ANTIFUNGAL_BINDING
            capacity_remaining = MAX_ANTIFUNGAL_BINDING - agent.bound_ANTIFUNGAL
            want_to_absorb = min(BINDING_RATE_ANTIFUNGAL * TIME_STEP_DT, capacity_remaining)
            actual_absorption = min(want_to_absorb, model.ANTIFUNGAL_layer[agent.y, agent.x])

            agent.bound_ANTIFUNGAL += actual_absorption
            model.ANTIFUNGAL_layer[agent.y, agent.x] -= actual_absorption
        end
    end

    append!(model.agents, new_offspring)

    # Diffusion
    laplacian_n = imfilter(model.nutrient_layer, centered(model.laplacian_kernel), "reflect")
    model.nutrient_layer .+= DIFFUSION_NUTRIENT .* laplacian_n .* TIME_STEP_DT
    model.nutrient_layer .= max.(model.nutrient_layer, 0.0)

    laplacian_f = imfilter(model.ANTIFUNGAL_layer, centered(model.laplacian_kernel), "reflect")
    model.ANTIFUNGAL_layer .+= DIFFUSION_ANTIFUNGAL .* laplacian_f .* TIME_STEP_DT
    model.ANTIFUNGAL_layer .= max.(model.ANTIFUNGAL_layer, 0.0)
end

function main()
    println("Initializing simulation...")
    model = PetriDishModel()
    
    every_n_steps = 2  # Capture every 2nd frame
    fps = 10 

    println("Starting simulation and recording frames...")
    
    anim = @animate for step in 1:SIMULATION_STEPS
        step_environment!(model)

        # Data extraction for plotting
        alive_agents = [a for a in model.agents if a.alive]
        
        # Split into healthy vs. apoptotic for visual distinction
        pcd_plus_healthy_x  = Float64[a.x for a in alive_agents if isa(a, PCDPlus) && !a.is_apoptotic]
        pcd_plus_healthy_y  = Float64[a.y for a in alive_agents if isa(a, PCDPlus) && !a.is_apoptotic]
        
        pcd_minus_healthy_x = Float64[a.x for a in alive_agents if isa(a, PCDMinus) && !a.is_apoptotic]
        pcd_minus_healthy_y = Float64[a.y for a in alive_agents if isa(a, PCDMinus) && !a.is_apoptotic]

        apoptotic_x = Float64[a.x for a in alive_agents if a.is_apoptotic]
        apoptotic_y = Float64[a.y for a in alive_agents if a.is_apoptotic]

        dead_apoptosis_x = Float64[a.x for a in model.agents if a.dead_apoptosis]
        dead_apoptosis_y = Float64[a.y for a in model.agents if a.dead_apoptosis]

        dead_necrosis_x = Float64[a.x for a in model.agents if a.dead_necrosis]
        dead_necrosis_y = Float64[a.y for a in model.agents if a.dead_necrosis]

        dead_starved_x = Float64[a.x for a in model.agents if a.dead_starvation]
        dead_starved_y = Float64[a.y for a in model.agents if a.dead_starvation]

        # Calculate live frequency for the dynamic title
        num_live = length(alive_agents)
        num_pcd_plus_live = length(pcd_plus_healthy_x) + count(a -> isa(a, PCDPlus) && a.is_apoptotic, alive_agents)
        pcd_plus_freq = num_live > 0 ? round((num_pcd_plus_live / num_live) * 100, digits=1) : 0.0

        # Subplot 1: Nutrients
        p1 = heatmap(model.nutrient_layer, 
                     title="Nutrients (Step $step)", 
                     color=:viridis, clims=(0, INIT_NUTRIENT_LEVEL),
                     aspect_ratio=:equal)

        # Subplot 2: ANTIFUNGAL
        p2 = heatmap(model.ANTIFUNGAL_layer, 
                     title="ANTIFUNGAL", 
                     color=:ice, clims=(0, INIT_ANTIFUNGAL_LEVEL),
                     aspect_ratio=:equal)

        # Subplot 3: Cells
        p3 = plot(title="Live: $num_live (PCD+: $pcd_plus_freq%)", 
                  xlims=(1, model.size), ylims=(1, model.size), 
                  aspect_ratio=:equal, legend=:topright)

        # DRAWING ORDER: Plotted bottom-to-top. 
        # Live cells drawn first, dead cells drawn last to ensure they aren't hidden
        if !isempty(pcd_plus_healthy_x)
            scatter!(p3, pcd_plus_healthy_x, pcd_plus_healthy_y, 
                     label="PCD+", color=:blue, markersize=2, markerstrokewidth=0)
        end
        if !isempty(pcd_minus_healthy_x)
            scatter!(p3, pcd_minus_healthy_x, pcd_minus_healthy_y, 
                     label="PCD-", color=:red, markersize=2, markerstrokewidth=0)
        end
        if !isempty(apoptotic_x)
            scatter!(p3, apoptotic_x, apoptotic_y, 
                     label="Apoptotic", color=:orange, markersize=2.5, markerstrokewidth=0)
        end
        if !isempty(dead_apoptosis_x)
            scatter!(p3, dead_apoptosis_x, dead_apoptosis_y, 
                     label="Dead (Apop)", color=:darkgray, markersize=3, markerstrokewidth=0)
        end
        if !isempty(dead_necrosis_x)
            scatter!(p3, dead_necrosis_x, dead_necrosis_y, 
                     label="Dead (Necro)", color=:black, markersize=3, markerstrokewidth=0)
        end
        if !isempty(dead_starved_x)
            scatter!(p3, dead_starved_x, dead_starved_y, 
                     label="Dead (Starve)", color=:magenta, markersize=3, markerstrokewidth=0)
        end

        # Layout combine
        plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))

        if step % 10 == 0
            println("Progress: Step $step / $SIMULATION_STEPS")
        end
    end every every_n_steps  

    # Save the GIF
    gif_name = "petri_dish_simulation.gif"
    gif(anim, gif_name, fps = fps)
    
    println("Success! GIF saved as: ", gif_name)

    # ==========================================
    # --- FITNESS & FREQUENCY TRACKER REPORT ---
    # ==========================================
    initial_pcd_plus = Int(round(INITIAL_CELLS * INIT_PCD_PLUS_FRACTION))
    initial_pcd_minus = INITIAL_CELLS - initial_pcd_plus
    
    final_pcd_plus = count(a -> isa(a, PCDPlus) && a.alive, model.agents)
    final_pcd_minus = count(a -> isa(a, PCDMinus) && a.alive, model.agents)
    
    fitness_plus = initial_pcd_plus > 0 ? final_pcd_plus / initial_pcd_plus : 0.0
    fitness_minus = initial_pcd_minus > 0 ? final_pcd_minus / initial_pcd_minus : 0.0
    
    total_final = final_pcd_plus + final_pcd_minus
    freq_plus = total_final > 0 ? (final_pcd_plus / total_final) * 100 : 0.0
    freq_minus = total_final > 0 ? (final_pcd_minus / total_final) * 100 : 0.0

    println("\n==========================================")
    println("--- GENOTYPE FREQUENCY & FITNESS REPORT ---")
    println("==========================================")
    println("Initial Cells -> PCD+: $initial_pcd_plus | PCD-: $initial_pcd_minus")
    println("Final Live Cells -> PCD+: $final_pcd_plus | PCD-: $final_pcd_minus")
    println("Final Frequency -> PCD+: $(round(freq_plus, digits=2))% | PCD-: $(round(freq_minus, digits=2))%")
    println("------------------------------------------")
    println("Absolute Fitness (Final / Initial):")
    println("  PCD+ Fitness : $(round(fitness_plus, digits=3))")
    println("  PCD- Fitness : $(round(fitness_minus, digits=3))")
    
    if fitness_minus > 0
        println("Relative Fitness (PCD+ / PCD-) : $(round(fitness_plus / fitness_minus, digits=3))")
    else
        println("Relative Fitness (PCD+ / PCD-) : Inf (PCD- extinct)")
    end
    println("==========================================\n")
end

main()