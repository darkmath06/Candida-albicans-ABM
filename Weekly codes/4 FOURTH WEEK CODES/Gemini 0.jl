# Complete, well-commented, runnable code for this single file
using Agents
using Random
using LinearAlgebra
using DataFrames
using CairoMakie # Added for plotting and GIF generation

# ==============================================================================
# 1. AGENT DEFINITIONS
# ==============================================================================

# Wild-type strain capable of apoptosis
@agent struct PCDPlusCell(GridAgent{2})
    health::Symbol              # :alive, :dead_apoptosis, :dead_necrosis, :dead_starvation
    biomass::Float64            # Structural biomass
    internal_nutrients::Float64 # Internal nutrient reservoir
    bound_amb::Float64          # Locally bound antifungal amount
    amb_exposure_time::Int      # Counter for time exposed to AmB above threshold
    time_dead::Int              # Counter tracking how long they have been dead
    in_apoptosis::Bool          # Boolean for being actively in the apoptotic process
    apoptosis_timer::Int        # Timer for the apoptotic process
end

# Mutant strain incapable of apoptosis
@agent struct PCDMinusCell(GridAgent{2})
    health::Symbol              # :alive, :dead_necrosis, :dead_starvation
    biomass::Float64            # Structural biomass
    internal_nutrients::Float64 # Internal nutrient reservoir
    bound_amb::Float64          # Locally bound antifungal amount
    amb_exposure_time::Int      # Counter for time exposed to AmB above threshold
    time_dead::Int              # Counter tracking how long they have been dead
end

# ==============================================================================
# 2. MODEL PROPERTIES & DIFFUSION SOLVER
# ==============================================================================

@kwdef mutable struct PetriDishProperties
    # Spatial layers
    nutrients::Matrix{Float64}
    amb::Matrix{Float64}
    
    # Mass Balance Counters
    total_burned_nutrients::Float64 = 0.0
    total_injected_amb::Float64 = 0.0
    
    # Diffusion coefficients
    D_nut::Float64 = 0.15
    D_amb::Float64 = 0.25
    
    # Biological & Kinetic parameters
    Vmax::Float64 = 2.0
    Km::Float64 = 10.0
    maintenance_cost::Float64 = 0.5
    starvation_threshold::Float64 = 5.0
    growth_efficiency::Float64 = 0.6
    division_threshold::Float64 = 25.0
    push_radius::Int = 2
    
    # Antifungal dynamics
    bmax_live::Float64 = 10.0
    bmax_dead::Float64 = 50.0 # Dead cells act as a massive sponge
    kon::Float64 = 0.05
    koff::Float64 = 0.01
    
    # Stress & Death thresholds
    amb_damage_threshold::Float64 = 2.0
    amb_time_threshold::Int = 5
    death_rate_scalar::Float64 = 0.05
    apoptosis_prob::Float64 = 0.85
    apoptosis_duration::Int = 6
    clearance_time::Int = 20
    
    # Simulation control
    step_counter::Int = 0
    injection_time::Int = 50
    injection_amount::Float64 = 80.0
end

# Implicit Jacobi Diffusion Solver for 2D grids (Neumann boundary conditions)
function jacobi_diffusion!(grid::Matrix{Float64}, D::Float64, iterations::Int=5)
    temp = copy(grid)
    w, h = size(grid)
    for _ in 1:iterations
        for i in 1:w, j in 1:h
            # No-flux boundaries
            up = j == 1 ? 1 : j - 1
            down = j == h ? h : j + 1
            left = i == 1 ? 1 : i - 1
            right = i == w ? w : i + 1
            
            # Implicit update formulation
            temp[i, j] = (grid[i, j] + D * (temp[left, j] + temp[right, j] + temp[i, up] + temp[i, down])) / (1 + 4D)
        end
    end
    grid .= temp
end

# ==============================================================================
# 3. AGENT STEP HELPER FUNCTIONS
# ==============================================================================

# Helper: Pay Maintenance Cost (Strict Mass Balance Version)
function pay_maintenance!(agent, model)
    if agent.internal_nutrients >= model.maintenance_cost
        agent.internal_nutrients -= model.maintenance_cost
        model.total_burned_nutrients += model.maintenance_cost
    else
        # Deficit: burn remaining internal, then burn structural biomass 1:1
        deficit = model.maintenance_cost - agent.internal_nutrients
        model.total_burned_nutrients += agent.internal_nutrients
        agent.internal_nutrients = 0.0
        
        agent.biomass -= deficit
        model.total_burned_nutrients += deficit
    end
    
    # Check if biomass dropped below critical starvation threshold
    if agent.biomass < model.starvation_threshold
        agent.health = :dead_starvation
        # Spill remaining internals, if any
        model.nutrients[agent.pos...] += agent.internal_nutrients
        agent.internal_nutrients = 0.0
        
        # Remaining biomass degrades to "burned/lost" pool
        model.total_burned_nutrients += agent.biomass
        agent.biomass = 0.0
    end
end

# Helper: AmB Binding Dynamics
function bind_amb!(agent, model, bmax)
    local_amb = model.amb[agent.pos...]
    available_sites = max(0.0, bmax - agent.bound_amb)
    
    bind_rate = model.kon * local_amb * available_sites
    unbind_rate = model.koff * agent.bound_amb
    net_change = bind_rate - unbind_rate
    
    if net_change > 0
        actual_bind = min(net_change, local_amb)
        agent.bound_amb += actual_bind
        model.amb[agent.pos...] -= actual_bind
    else
        actual_unbind = min(-net_change, agent.bound_amb)
        agent.bound_amb -= actual_unbind
        model.amb[agent.pos...] += actual_unbind
    end
end

# Helper: Division and Mechanical Pushing
function attempt_division!(agent, model)
    target_pos = nothing
    immediate_neighbors = nearby_positions(agent.pos, model, 1)
    empty_immediate = [p for p in immediate_neighbors if isempty(p, model)]
    
    if !isempty(empty_immediate)
        # Empty space available directly next to the cell
        target_pos = rand(abmrng(model), empty_immediate)
    else
        # Contact Inhibition & Pushing
        # Search for empty spaces within a defined radius
        extended_neighbors = nearby_positions(agent.pos, model, model.push_radius)
        empty_extended = [p for p in extended_neighbors if isempty(p, model)]
        
        if !isempty(empty_extended)
            # Find an empty destination spot
            dest_pos = rand(abmrng(model), empty_extended)
            
            # Select an immediate neighbor to shove into that destination
            occupied_immediate = [p for p in immediate_neighbors if !isempty(p, model)]
            if !isempty(occupied_immediate)
                push_pos = rand(abmrng(model), occupied_immediate)
                cell_to_push_id = id_in_position(push_pos, model)
                
                if cell_to_push_id != 0
                    cell_to_push = model[cell_to_push_id]
                    move_agent!(cell_to_push, dest_pos, model)
                    target_pos = push_pos # The adjacent space is now free for the daughter cell
                end
            end
        end
    end
    
    # Create Daughter Cell
    if target_pos !== nothing
        agent.biomass /= 2.0
        agent.internal_nutrients /= 2.0
        
        if agent isa PCDPlusCell
            add_agent!(target_pos, PCDPlusCell, model, :alive, agent.biomass, agent.internal_nutrients, 0.0, 0, 0, false, 0)
        else
            add_agent!(target_pos, PCDMinusCell, model, :alive, agent.biomass, agent.internal_nutrients, 0.0, 0, 0)
        end
    end
end

# ==============================================================================
# 4. MAIN AGENT STEP FUNCTION
# ==============================================================================

function agent_step!(agent, model)
    # Check if dead and manage clearance / sponge effect
    is_actively_apoptotic = agent isa PCDPlusCell && agent.in_apoptosis
    
    if agent.health != :alive && !is_actively_apoptotic
        agent.time_dead += 1
        
        # Dead Cell Clearance
        if agent.time_dead >= model.clearance_time
            # Release bound AmB back to environment and remove
            model.amb[agent.pos...] += agent.bound_amb
            remove_agent!(agent, model)
            return
        end
        
        # The Sponge Effect: Dead cells continue to bind AmB at higher capacities
        bind_amb!(agent, model, model.bmax_dead)
        return
    end

    # --- Live and Apoptotic Behavior ---
    
    # 1. Apoptosis Timer (PCD+ only)
    if is_actively_apoptotic
        agent.apoptosis_timer += 1
        
        # Pay maintenance cost even during apoptosis
        pay_maintenance!(agent, model)
        if agent.health == :dead_starvation return end # Starved during apoptosis
        
        if agent.apoptosis_timer >= model.apoptosis_duration
            agent.health = :dead_apoptosis
            agent.in_apoptosis = false
            # Dump internal nutrients back to grid, structural biomass lost permanently
            model.nutrients[agent.pos...] += agent.internal_nutrients
            agent.internal_nutrients = 0.0
            
            model.total_burned_nutrients += agent.biomass # Biomass decays
            agent.biomass = 0.0
        end
        return # Skip growth/division if in apoptosis
    end

    # 2. Nutrient Uptake (Monod kinetics)
    local_nutrients = model.nutrients[agent.pos...]
    uptake = model.Vmax * local_nutrients / (model.Km + local_nutrients)
    actual_uptake = min(uptake, local_nutrients)
    
    model.nutrients[agent.pos...] -= actual_uptake
    agent.internal_nutrients += actual_uptake

    # 3. Maintenance and Starvation Check
    pay_maintenance!(agent, model)
    if agent.health == :dead_starvation return end

    # 4. Antifungal (AmB) Dynamics
    bind_amb!(agent, model, model.bmax_live)

    # 5. Stress, Necrosis, and Apoptosis Induction
    local_amb = model.amb[agent.pos...]
    if local_amb > model.amb_damage_threshold
        agent.amb_exposure_time += 1
    else
        agent.amb_exposure_time = max(0, agent.amb_exposure_time - 1)
    end

    if agent.amb_exposure_time > model.amb_time_threshold
        # Concentration-dependent probability of dying
        die_prob = 1.0 - exp(-model.death_rate_scalar * agent.bound_amb)
        
        if rand(abmrng(model)) < die_prob
            if agent isa PCDPlusCell && rand(abmrng(model)) < model.apoptosis_prob
                # Enter orchestrated death (Apoptosis)
                agent.in_apoptosis = true
            else
                # Sudden cell lysis (Necrosis)
                agent.health = :dead_necrosis
                model.nutrients[agent.pos...] += agent.internal_nutrients
                agent.internal_nutrients = 0.0
                
                model.total_burned_nutrients += agent.biomass # Biomass structurally destroyed
                agent.biomass = 0.0
            end
            return
        end
    end

    # 6. Growth (Strict Mass Balance: converting internal to biomass, rest lost to metabolic heat)
    available_for_growth = agent.internal_nutrients
    growth_amount = available_for_growth * model.growth_efficiency
    metabolic_loss = available_for_growth - growth_amount
    
    agent.biomass += growth_amount
    agent.internal_nutrients = 0.0
    model.total_burned_nutrients += metabolic_loss

    # 7. Division (Budding) and Mechanical Pushing
    if agent.biomass >= model.division_threshold
        attempt_division!(agent, model)
    end
end

# ==============================================================================
# 5. MODEL STEP FUNCTION
# ==============================================================================

function model_step!(model)
    model.step_counter += 1
    
    # Apply implicit Jacobi diffusion to continuous layers
    jacobi_diffusion!(model.nutrients, model.D_nut, 5)
    jacobi_diffusion!(model.amb, model.D_amb, 5)
    
    # Inject Antifungal halfway through simulation
    if model.step_counter == model.injection_time
        # Place a heavy dose of AmB into the center of the dish
        center = size(model.amb) .÷ 2
        r = 10
        for i in (center[1]-r):(center[1]+r), j in (center[2]-r):(center[2]+r)
            if (i - center[1])^2 + (j - center[2])^2 <= r^2
                model.amb[i, j] += model.injection_amount
                model.total_injected_amb += model.injection_amount # Track globally
            end
        end
        println("--> Amphotericin B Injected at step $(model.step_counter) <--")
    end
end

# ==============================================================================
# 6. INITIALIZATION & EXECUTION SCRIPT
# ==============================================================================

function setup_model(cell_type::Type, grid_size::Tuple{Int,Int}=(50, 50); initial_pop=50)
    space = GridSpaceSingle(grid_size; periodic=false)
    
    properties = PetriDishProperties(
        nutrients = fill(150.0, grid_size),
        amb = zeros(Float64, grid_size)
    )
    
    model = StandardABM(cell_type, space; 
        agent_step! = agent_step!, 
        model_step! = model_step!,
        properties = properties,
        rng = MersenneTwister(42)
    )
    
    # Seed initial population in the center
    center = grid_size .÷ 2
    for _ in 1:initial_pop
        pos = (
            clamp(center[1] + rand(-3:3), 1, grid_size[1]), 
            clamp(center[2] + rand(-3:3), 1, grid_size[2])
        )
        if isempty(pos, model)
            if cell_type == PCDPlusCell
                add_agent!(pos, PCDPlusCell, model, :alive, 15.0, 5.0, 0.0, 0, 0, false, 0)
            else
                add_agent!(pos, PCDMinusCell, model, :alive, 15.0, 5.0, 0.0, 0, 0)
            end
        end
    end
    
    return model
end

function run_simulations_with_video()
    total_steps = 150
    grid_dims = (60, 60)
    
    println("Initializing PCD+ (Wild-Type) Model for visualization...")
    model = setup_model(PCDPlusCell, grid_dims)
    
    # --- 1. Gather Initial Counter Values ---
    init_env_nut = sum(model.nutrients)
    init_int_nut = sum(a -> a.internal_nutrients, allagents(model); init=0.0)
    init_biomass = sum(a -> a.biomass, allagents(model); init=0.0)
    init_total_nut = init_env_nut + init_int_nut + init_biomass

    # --- 2. Setup Live Visualization (CairoMakie) ---
    println("Generating live graph as 'Gemini_plot.gif'...")
    # Explicitly prefix plotting types to avoid name clashes with other loaded packages
    fig = CairoMakie.Figure(size = (1200, 400))
    ax1 = CairoMakie.Axis(fig[1, 1], title="Petri Dish (Cells)", aspect=CairoMakie.DataAspect())
    ax2 = CairoMakie.Axis(fig[1, 2], title="Nutrient Layer", aspect=CairoMakie.DataAspect())
    ax3 = CairoMakie.Axis(fig[1, 3], title="Antifungal Layer", aspect=CairoMakie.DataAspect())

    # Map colors for cells based on health
    ac(a) = a.health == :alive ? :green : 
           (a.health == :dead_apoptosis ? :purple : 
           (a.health == :dead_necrosis ? :red : :black))
    
    abmobs = ABMObservable(model)
    # Updated to use explicitly full keyword arguments
    abmplot!(ax1, abmobs; agent_color=ac, agent_size=10)
    
    # Observables for the continuous heatmaps
    nut_obs = CairoMakie.Observable(copy(model.nutrients))
    amb_obs = CairoMakie.Observable(copy(model.amb))
    
    CairoMakie.heatmap!(ax2, nut_obs, colormap=:viridis, colorrange=(0, 150))
    CairoMakie.heatmap!(ax3, amb_obs, colormap=:magma, colorrange=(0, 10))
    
    CairoMakie.hidedecorations!(ax1)
    CairoMakie.hidedecorations!(ax2)
    CairoMakie.hidedecorations!(ax3)

    # --- 3. Execute & Record GIF ---
    CairoMakie.record(fig, "Gemini_plot.gif", 0:total_steps; framerate = 15) do i
        if i > 0
            step!(abmobs, 1)
        end
        # Update heatmaps locally using the Observable bracket [] access
        nut_obs[] = copy(abmobs.model[].nutrients)
        amb_obs[] = copy(abmobs.model[].amb)
    end
    println("--> Saved Gemini_plot.gif successfully.")

    # --- 4. Gather Final Counter Values ---
    fin_env_nut = sum(model.nutrients)
    fin_int_nut = sum(a -> a.internal_nutrients, allagents(model); init=0.0)
    fin_biomass = sum(a -> a.biomass, allagents(model); init=0.0)
    fin_burned  = model.total_burned_nutrients
    fin_total_nut = fin_env_nut + fin_int_nut + fin_biomass + fin_burned
    
    fin_env_amb = sum(model.amb)
    fin_bound_amb = sum(a -> a.bound_amb, allagents(model); init=0.0)
    fin_total_amb = fin_env_amb + fin_bound_amb
    
    # --- 5. Output Final Balance Report ---
    println("\n=== Mass Balance: Nutrient Counter ===")
    println("INITIAL TOTAL:     ", round(init_total_nut, digits=2))
    println("  Environment:     ", round(init_env_nut, digits=2))
    println("  Internal:        ", round(init_int_nut, digits=2))
    println("  Biomass:         ", round(init_biomass, digits=2))
    println("----------------------------------------")
    println("FINAL TOTAL:       ", round(fin_total_nut, digits=2))
    println("  Environment:     ", round(fin_env_nut, digits=2))
    println("  Internal:        ", round(fin_int_nut, digits=2))
    println("  Biomass:         ", round(fin_biomass, digits=2))
    println("  Burned/Lost:     ", round(fin_burned, digits=2))
    println("----------------------------------------")
    println("Discrepancy:       ", round(abs(init_total_nut - fin_total_nut), digits=6))

    println("\n=== Mass Balance: Antifungal (AmB) ===")
    println("INITIAL (Injected): ", round(model.total_injected_amb, digits=2))
    println("----------------------------------------")
    println("FINAL TOTAL:        ", round(fin_total_amb, digits=2))
    println("  Environment:      ", round(fin_env_amb, digits=2))
    println("  Bound to Cells:   ", round(fin_bound_amb, digits=2))
    println("----------------------------------------")
    println("Discrepancy:        ", round(abs(model.total_injected_amb - fin_total_amb), digits=6))
end

# Execute the simulation and render the visualization safely handling world-age dynamics
Base.invokelatest(run_simulations_with_video)