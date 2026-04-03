using Agents
using Random
using StatsBase
using Images
using ImageFiltering
using Plots

# ==========================================
# --- GLOBAL EXPERIMENTAL PARAMETERS ---
# ==========================================
const GRID_SIZE_PX = 100 
const INITIAL_CELLS = 1
const SIMULATION_STEPS = 450
const TIME_STEP_DT = 1.0 / 3.0 

const INIT_NUTRIENT_LEVEL = 20.0
const INIT_ANTIFUNGAL_LEVEL = 4

const DIFFUSION_NUTRIENT = 0.2
const DIFFUSION_ANTIFUNGAL = 0.2
const DIFFUSION_ITERATIONS = 15 

const MU_MAX = 0.34            
const MONOD_KS = 5.0           
const MAINTENANCE_COEFF = 0.015 
const YIELD_TRUE = 0.39        
const NEWBORN_BIOMASS = 1.0    
const DIVISION_BIOMASS = 2.0   
const APOPTOSIS_NUTRIENT_RELEASE_FRAC = 0.85 

const PUSH_PROBABILITY = 0.02 
const MAX_PUSH_RADIUS = 10    

const MAX_ANTIFUNGAL_BINDING_LIVE = 0.42  
const MAX_ANTIFUNGAL_BINDING_DEAD = 2.55  
const K_ON_ANTIFUNGAL = 0.01              
const K_OFF_ANTIFUNGAL = 0.005            

const ANTIFUNGAL_DAMAGE_THRESHOLD = 0.5      
const STRESS_START_TIME = 3.30               
const APOPTOSIS_DURATION = 2.0               
const ASSAY_DURATION_HOURS = 24.0            

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
function get_death_rates(c::Float64)
    if c <= 0.0
        return 0.0, 0.0
    end
    
    # Linear interpolation based on assay table
    C_vals = (0.0, 0.5, 4.0, 8.0, 16.0)
    apop_vals = (0.0, 0.15, 0.20, 0.57, 0.09)
    necro_vals = (0.0, 0.09, 0.20, 0.33, 0.91)

    target_apop_frac = 0.0
    target_necro_frac = 0.0

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
    if target_total_frac <= 0.0
        return 0.0, 0.0
    end
    
    hourly_total_rate = -log(1.0 - target_total_frac) / ASSAY_DURATION_HOURS
    ratio_apop = target_apop_frac / target_total_frac
    ratio_necro = target_necro_frac / target_total_frac
    
    return (hourly_total_rate * ratio_apop), (hourly_total_rate * ratio_necro)
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
            model.nutrient_layer[y, x] += agent.biomass * APOPTOSIS_NUTRIENT_RELEASE_FRAC
            agent.biomass = 0.0 
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
        total_rate = rate_apop + rate_necro
        
        if total_rate > 0
            prob_death = 1.0 - exp(-total_rate * TIME_STEP_DT)
            if rand() < prob_death
                agent.alive = false
                agent.dead_necrosis = true
            end
        end
    end
end

# ==========================================
# --- CORE MODEL STEP ---
# ==========================================
function complex_model_step!(model)
    n_demands = Dict{Int, Float64}()
    f_demands = Dict{Int, Float64}()
    f_releases = Dict{Int, Float64}()

    grid_n_demand = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    grid_f_demand = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)

    # PASS 1: Calculate Demands
    for agent in allagents(model)
        x, y = agent.pos
        
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

    # PASS 2: Allocate & Update
    for agent in allagents(model)
        x, y = agent.pos
        
        if agent.alive
            apply_stress!(agent, model.ANTIFUNGAL_layer[y, x], model)
        end

        if agent.alive && get(n_demands, agent.id, 0.0) > 0
            demand = n_demands[agent.id]
            available_n = model.nutrient_layer[y, x]
            total_demand_n = grid_n_demand[y, x]
            
            alloc_frac = total_demand_n > available_n ? (available_n / total_demand_n) : 1.0
            actual_intake = demand * alloc_frac
            model.nutrient_layer[y, x] -= actual_intake

            maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

            if !is_apoptotic(agent)
                if actual_intake >= maintenance_cost
                    agent.biomass += (actual_intake - maintenance_cost) * YIELD_TRUE
                end

                if agent.biomass >= DIVISION_BIOMASS
                    if !can_divide(agent)
                        agent.biomass = DIVISION_BIOMASS
                    else
                        chosen_spot = nothing
                        immediate_hood = collect(nearby_positions(agent.pos, model, 1))
                        empty_immediate = filter(p -> isempty(p, model), immediate_hood)
                        
                        if !isempty(empty_immediate)
                            chosen_spot = rand(empty_immediate)
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

    # PDE Diffusion
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
# --- DATA EXTRACTION & ANALYSIS RUNNER ---
# ==========================================
function main()
    println("Initializing strictly typed Agents.jl models for Data Extraction...")
    model_plus = initialize_model(PCDPlusCell)
    model_minus = initialize_model(PCDMinusCell)
    
    # --- Analytics Arrays ---
    times = Float64[]
    
    # Populations
    live_plus = Int[]; live_minus = Int[]
    
    # Death Modalities
    apop_plus = Int[]; necro_plus = Int[]; necro_minus = Int[]
    
    # Environment & Resource Data
    nut_plus = Float64[]; nut_minus = Float64[]
    bound_af_plus = Float64[]; bound_af_minus = Float64[]

    println("Running simulation fast (No visual rendering) for $SIMULATION_STEPS steps...")
    
    for step in 1:SIMULATION_STEPS
        Agents.step!(model_plus, 1)
        Agents.step!(model_minus, 1)

        # Record Data
        push!(times, step * TIME_STEP_DT)
        
        push!(live_plus, count(a -> a.alive, allagents(model_plus)))
        push!(live_minus, count(a -> a.alive, allagents(model_minus)))
        
        push!(apop_plus, count(is_dead_apoptosis, allagents(model_plus)))
        push!(necro_plus, count(a -> a.dead_necrosis, allagents(model_plus)))
        push!(necro_minus, count(a -> a.dead_necrosis, allagents(model_minus)))
        
        push!(nut_plus, sum(model_plus.nutrient_layer))
        push!(nut_minus, sum(model_minus.nutrient_layer))
        
        push!(bound_af_plus, sum(a.bound_ANTIFUNGAL for a in allagents(model_plus)))
        push!(bound_af_minus, sum(a.bound_ANTIFUNGAL for a in allagents(model_minus)))

        if step % 50 == 0
            println("Progress: Step $step / $SIMULATION_STEPS (Time: $(round(step * TIME_STEP_DT, digits=1)) hrs)")
        end
    end 
    
    println("Simulation Complete. Generating Analytics Dashboard...")
    
    # --- Plot 1: Population Fitness ---
    p1 = plot(times, live_plus, label="PCD+ (Wild-Type)", lw=2, color=:blue, 
              title="Live Population Over Time", ylabel="Cell Count", legend=:topleft)
    plot!(p1, times, live_minus, label="PCD- (Knockout)", lw=2, color=:red, linestyle=:dash)
    
    # --- Plot 2: Death Modalities ---
    p2 = plot(times, apop_plus, label="PCD+ Apoptotic Deaths", lw=2, color=:orange, 
              title="Cumulative Cell Death", ylabel="Cumulative Dead Count", legend=:topleft)
    plot!(p2, times, necro_plus, label="PCD+ Necrotic Deaths", lw=2, color=:black)
    plot!(p2, times, necro_minus, label="PCD- Necrotic Deaths", lw=2, color=:grey, linestyle=:dash)

    # --- Plot 3: Environmental Nutrients (Kin Selection) ---
    p3 = plot(times, nut_plus, label="PCD+ Env. Nutrients", lw=2, color=:green, 
              title="Nutrient Availability (Recycling Benefit)", ylabel="Total Nutrient Mass", legend=:topright)
    plot!(p3, times, nut_minus, label="PCD- Env. Nutrients", lw=2, color=:darkgreen, linestyle=:dash)

    # --- Plot 4: The Sponge Effect ---
    p4 = plot(times, bound_af_plus, label="PCD+ Bound Antifungal", lw=2, color=:purple, 
              title="Total Drug Bound to Cells (Sponge Effect)", xlabel="Time (Hours)", ylabel="Drug Mass", legend=:topleft)
    plot!(p4, times, bound_af_minus, label="PCD- Bound Antifungal", lw=2, color=:magenta, linestyle=:dash)

    # Combine plots into a 2x2 dashboard
    dashboard = plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 800), margin=5Plots.mm)
    
    output_filename = "evolutionary_analytics_dashboard.png"
    savefig(dashboard, output_filename)
    println("Dashboard successfully generated and saved as: ", output_filename)
    
    display(dashboard)
end

main()