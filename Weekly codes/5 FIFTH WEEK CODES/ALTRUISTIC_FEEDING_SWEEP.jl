using Agents
using Random
using StatsBase
using DataFrames
using CSV
using Printf

# ==========================================
# --- GLOBAL EXPERIMENTAL PARAMETERS ---
# ==========================================
# (Variables specifically for the sweep have been moved to the model properties)

const GRID_SIZE_PX = 150 
const INITIAL_CELLS = 1
const SIMULATION_STEPS = 216
const ANTIFUNGAL_INJECTION_STEP = 73 # 24h
const TIME_STEP_DT = 1.0 / 3.0       # 20 minutes

@enum ExposureMode SINGLE_SHOCK CONTINUOUS PULSATED

const ANTIFUNGAL_EXPOSURE_MODE = SINGLE_SHOCK   
const ANTIFUNGAL_PULSE_INTERVAL = 36        
const NUTRIENT_EXPOSURE_MODE = SINGLE_SHOCK
const NUTRIENT_INJECTION_STEP = 0          
const NUTRIENT_PULSE_INTERVAL = 36         
const MAX_BIOMASS_PER_PX = 3 

const DIFFUSION_NUTRIENT = 0.57
const DIFFUSION_ANTIFUNGAL = 0.3     # INCREASED: Drug penetrates the colony faster
const DIFFUSION_ITERATIONS = 15 

const MU_MAX = 0.7            
const MONOD_KS = 5.0           
const MAINTENANCE_COEFF = 0.015 
const YIELD_TRUE = 0.39        
const NEWBORN_BIOMASS = 1.0    
const DIVISION_BIOMASS = 2.0   
const STARVATION_BIOMASS = 0.75 
const RESERVOIR_FRACTION = 0.1 

const PUSH_PROBABILITY = 0.3 
const MAX_PUSH_RADIUS = 10    

const MAX_ANTIFUNGAL_BINDING_LIVE = 0.42  
const MAX_ANTIFUNGAL_BINDING_DEAD = 1.0   # DECREASED: Weakens the "shield" effect so core stops growing
const K_ON_ANTIFUNGAL = 0.01              
const K_OFF_ANTIFUNGAL = 0.005            

const ANTIFUNGAL_DAMAGE_THRESHOLD = 0.64     
const STRESS_START_TIME = 1                
const ASSAY_DURATION_HOURS = 200/60          

const TOTAL_DEATH_MAX = 0.99         
const TOTAL_DEATH_STEEPNESS = 0.8    
const TOTAL_DEATH_C50 = 4.0          

const NECRO_SLOPE = 0.056571         
const NECRO_INTERCEPT = -0.036       

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
    # --- SWEEP VARIABLES ---
    apoptosis_duration::Float64
    apoptosis_leak_rate::Float64
    init_nutrient_level::Float64
    init_antifungal_level::Float64
end

function initialize_model(AgentType::Type, starting_positions::Vector{Tuple{Int, Int}}, 
                          init_nutrients::Float64, init_antifungal::Float64,
                          apop_duration::Float64, apop_leak::Float64)
    
    space = GridSpace((GRID_SIZE_PX, GRID_SIZE_PX); periodic=false)

    props = PetriDishProperties(
        fill(init_nutrients, GRID_SIZE_PX, GRID_SIZE_PX),
        fill(0.0, GRID_SIZE_PX, GRID_SIZE_PX),
        [1/6 2/3 1/6; 2/3 -10/3 2/3; 1/6 2/3 1/6],
        [1/6 2/3 1/6; 2/3 0.0 2/3; 1/6 2/3 1/6],
        AgentType === PCDPlusCell,
        0.0,
        apop_duration,
        apop_leak,
        init_nutrients,
        init_antifungal
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
    if c < ANTIFUNGAL_DAMAGE_THRESHOLD
        return 0.0, 0.0
    end
    
    total_death = TOTAL_DEATH_MAX / (1.0 + exp(-TOTAL_DEATH_STEEPNESS * (c - TOTAL_DEATH_C50)))
    necro_raw = (NECRO_SLOPE * c) + NECRO_INTERCEPT
    target_necro_frac = clamp(necro_raw, 0.0, total_death)
    target_apop_frac = total_death - target_necro_frac
    target_total_frac = min(0.999, total_death)
    
    if target_total_frac <= 0.01; return 0.0, 0.0; end
    
    hourly_total_rate = -log(1.0 - target_total_frac) / ASSAY_DURATION_HOURS
    ratio_apop = target_apop_frac / total_death
    ratio_necro = target_necro_frac / total_death
    
    return hourly_total_rate * ratio_apop, hourly_total_rate * ratio_necro
end

function apply_stress!(agent::PCDPlusCell, local_ANTIFUNGAL::Float64, model)
    if !agent.alive; return; end

    if local_ANTIFUNGAL >= ANTIFUNGAL_DAMAGE_THRESHOLD
        agent.ANTIFUNGAL_exposure_time += TIME_STEP_DT
    end

    if agent.is_apoptotic
        agent.apoptosis_timer += TIME_STEP_DT
        
        # Gradual Nutrient Leak uses the SWEEP property now
        x, y = agent.pos
        leak_fraction = min(1.0, model.apoptosis_leak_rate * TIME_STEP_DT)
        leak_amount = agent.internal_nutrients * leak_fraction
        agent.internal_nutrients -= leak_amount
        model.nutrient_layer[y, x] += leak_amount

        # Duration uses the SWEEP property now
        if agent.apoptosis_timer >= model.apoptosis_duration
            set_dead_apoptosis!(agent)
            model.nutrient_layer[y, x] += agent.internal_nutrients
            agent.internal_nutrients = 0.0
        end
    else
        if agent.ANTIFUNGAL_exposure_time >= STRESS_START_TIME
            rate_apop, rate_necro = get_death_rates(local_ANTIFUNGAL)
            total_rate = rate_apop + rate_necro
            
            if total_rate > 0
                prob_death = 1.0 - exp(-total_rate * TIME_STEP_DT)
                if rand() < prob_death
                    if rand() < (rate_apop / total_rate)
                        agent.is_apoptotic = true 
                    else
                        agent.alive = false
                        agent.dead_necrosis = true
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
                model.nutrient_layer[agent.pos[2], agent.pos[1]] += agent.internal_nutrients
                agent.internal_nutrients = 0.0
            end
        end
    end
end

# ==========================================
# --- SIMPLE FAST DIFFUSION ---
# ==========================================
function diffuse_layer!(layer::Matrix{Float64}, kernel::Matrix{Float64}, neighbor_kernel::Matrix{Float64}, alpha::Float64)
    h, w = size(layer)
    u = copy(layer)
    denom = 1.0 + (5.0 / 3.0) * alpha
    for _ in 1:DIFFUSION_ITERATIONS
        # In-place diffusion approximation for headless runs
        for i in 2:h-1, j in 2:w-1
            u[i, j] = (layer[i,j] + (alpha/2.0) * (u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1])) / denom
        end
    end
    layer .= max.(u, 0.0)
end

# ==========================================
# --- CORE MODEL STEP ---
# ==========================================

function complex_model_step!(model)
    n_demands = Dict{Int, Float64}()
    f_demands = Dict{Int, Float64}()
    f_releases = Dict{Int, Float64}()

    for agent in allagents(model)
        x, y = agent.pos
        if agent.alive
            local_n = model.nutrient_layer[y, x]
            max_reservoir = agent.biomass * RESERVOIR_FRACTION
            reservoir_deficit = max(0.0, max_reservoir - agent.internal_nutrients)

            if is_apoptotic(agent)
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
        end
        
        local_f = model.ANTIFUNGAL_layer[y, x]
        bound_f = agent.bound_ANTIFUNGAL
        current_max = agent.alive ? MAX_ANTIFUNGAL_BINDING_LIVE : MAX_ANTIFUNGAL_BINDING_DEAD
        cap_remaining = max(0.0, current_max - bound_f)

        net_change = (K_ON_ANTIFUNGAL * local_f * cap_remaining - K_OFF_ANTIFUNGAL * bound_f) * TIME_STEP_DT

        if net_change > 0
            f_demands[agent.id] = net_change
        else
            f_releases[agent.id] = min(-net_change, bound_f)
        end
    end

    newborn_spots = Tuple{Tuple{Int,Int}, Float64, Float64}[] 
    planned_biomass = Dict{Tuple{Int,Int}, Float64}()
    randomized_agents = shuffle!(collect(allagents(model)))

    for agent in randomized_agents
        x, y = agent.pos
        
        if agent.alive
            apply_stress!(agent, model.ANTIFUNGAL_layer[y, x], model)
        end

        if agent.alive && get(n_demands, agent.id, 0.0) > 0
            demand = n_demands[agent.id]
            actual_intake = min(demand, model.nutrient_layer[y, x])
            model.nutrient_layer[y, x] -= actual_intake
            agent.internal_nutrients += actual_intake

            maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT

            if is_apoptotic(agent)
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

                if agent.alive && agent.biomass >= DIVISION_BIOMASS
                    if !can_divide(agent)
                        excess = agent.biomass - DIVISION_BIOMASS
                        agent.biomass = DIVISION_BIOMASS
                        model.total_lost_nutrients += (excess / YIELD_TRUE) 
                    else
                        immediate_hood = [agent.pos; collect(nearby_positions(agent.pos, model, 1))]
                        spot_status = map(immediate_hood) do p
                            cur_b = sum((a.biomass for a in agents_in_position(p, model)), init=0.0)
                            (pos = p, bio = cur_b + get(planned_biomass, p, 0.0))
                        end
                        available_spots = filter(s -> s.bio + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX, spot_status)
                        
                        if !isempty(available_spots)
                            chosen_spot = rand([s.pos for s in available_spots])
                            agent.biomass -= NEWBORN_BIOMASS 
                            daughter_fraction = NEWBORN_BIOMASS / (agent.biomass + NEWBORN_BIOMASS)
                            daughter_n = agent.internal_nutrients * daughter_fraction
                            agent.internal_nutrients -= daughter_n
                            planned_biomass[chosen_spot] = get(planned_biomass, chosen_spot, 0.0) + NEWBORN_BIOMASS
                            push!(newborn_spots, (chosen_spot, NEWBORN_BIOMASS, daughter_n))
                        end
                    end
                end
            end
        end

        if get(f_demands, agent.id, 0.0) > 0
            demand = f_demands[agent.id]
            actual_binding = min(demand, model.ANTIFUNGAL_layer[y, x])
            agent.bound_ANTIFUNGAL += actual_binding
            model.ANTIFUNGAL_layer[y, x] -= actual_binding
        elseif get(f_releases, agent.id, 0.0) > 0
            release = f_releases[agent.id]
            agent.bound_ANTIFUNGAL -= release
            model.ANTIFUNGAL_layer[y, x] += release
        end
    end

    for (pos, b, n) in newborn_spots
        cur_b = sum((a.biomass for a in agents_in_position(pos, model)), init=0.0)
        if cur_b + b <= MAX_BIOMASS_PER_PX + 0.01 
            if model.is_pcd_plus
                add_agent!(pos, PCDPlusCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, b, n, true)
            else
                add_agent!(pos, PCDMinusCell, model, true, 0.0, false, false, 0.0, b, n)
            end
        end
    end

    # Fast Diffusion
    alpha_n = DIFFUSION_NUTRIENT * TIME_STEP_DT
    alpha_f = DIFFUSION_ANTIFUNGAL * TIME_STEP_DT
    diffuse_layer!(model.nutrient_layer, model.laplacian_kernel, model.neighbor_kernel, alpha_n)
    diffuse_layer!(model.ANTIFUNGAL_layer, model.laplacian_kernel, model.neighbor_kernel, alpha_f)
end

# ==========================================
# --- EXPERIMENTAL SWEEP LOGIC ---
# ==========================================

function run_idea_A_sweep()
    println("Starting Idea A: Nutrient Altruism Parameter Sweep")
    
    # 1. Define the Environmental Constants
    sweep_nutrients = 8     # MODIFIED: Slightly lower to prevent massive overgrowth before AmB
    sweep_antifungal = 2.8    # MODIFIED: Dialed back slightly so it doesn't kill 100% of cells by step 400

    # 2. Define the Variable Grids
    apop_durations = [0.5, 2.0, 4.0, 8.0]
    apop_leak_rates = [0.1, 0.5, 0.9]
    num_replicates = 10       
    
    # Setup data collection
    results = DataFrame(
        Apoptosis_Duration = Float64[],
        Apoptosis_Leak_Rate = Float64[],
        Replicate = Int[],
        Strain = String[],
        Alive_Cells = Int[],
        Total_Cells = Int[],
        Survivability_Pct = Float64[],
        Total_Live_Biomass = Float64[]
    )

    total_runs = length(apop_durations) * length(apop_leak_rates) * num_replicates
    current_run = 0

    # Calculate center for initialization
    cx = (GRID_SIZE_PX + 1) / 2.0
    cy = (GRID_SIZE_PX + 1) / 2.0
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    starting_positions = all_pos[1:min(INITIAL_CELLS, length(all_pos))]

    # 3. Execute the Nested Loop
    for dur in apop_durations
        for leak in apop_leak_rates
            for rep in 1:num_replicates
                current_run += 1
                if current_run % 10 == 0
                    println("Progress: Run $current_run / $(total_runs * 2)") 
                end

                # --- Run PCD+ Strain ---
                model_plus = initialize_model(PCDPlusCell, starting_positions, sweep_nutrients, sweep_antifungal, dur, leak)
                for step in 1:SIMULATION_STEPS
                    if step == NUTRIENT_INJECTION_STEP
                        model_plus.nutrient_layer .= sweep_nutrients
                    end
                    if step == ANTIFUNGAL_INJECTION_STEP
                        model_plus.ANTIFUNGAL_layer .= sweep_antifungal
                    end
                    Agents.step!(model_plus, 1)
                end
                
                # Collect PCD+ Data
                alive_plus = count(a -> a.alive, allagents(model_plus))
                tot_plus = nagents(model_plus)
                surv_plus = tot_plus > 0 ? (alive_plus / tot_plus) * 100.0 : 0.0
                bio_plus = sum(a.biomass for a in allagents(model_plus) if a.alive; init=0.0)

                push!(results, (dur, leak, rep, "PCD+", alive_plus, tot_plus, surv_plus, bio_plus))

                # --- Run PCD- Strain (Control) ---
                model_minus = initialize_model(PCDMinusCell, starting_positions, sweep_nutrients, sweep_antifungal, dur, leak)
                for step in 1:SIMULATION_STEPS
                    if step == NUTRIENT_INJECTION_STEP
                        model_minus.nutrient_layer .= sweep_nutrients
                    end
                    if step == ANTIFUNGAL_INJECTION_STEP
                        model_minus.ANTIFUNGAL_layer .= sweep_antifungal
                    end
                    Agents.step!(model_minus, 1)
                end
                
                # Collect PCD- Data
                alive_minus = count(a -> a.alive, allagents(model_minus))
                tot_minus = nagents(model_minus)
                surv_minus = tot_minus > 0 ? (alive_minus / tot_minus) * 100.0 : 0.0
                bio_minus = sum(a.biomass for a in allagents(model_minus) if a.alive; init=0.0)

                push!(results, (dur, leak, rep, "PCD-", alive_minus, tot_minus, surv_minus, bio_minus))
            end
        end
    end

    # 4. Export the Data
    output_filename = "IdeaA_NutrientAltruism_Results.csv"
    CSV.write(output_filename, results)
    println("Sweep Complete! Results exported to: $output_filename")
end

# Run the sweep
run_idea_A_sweep()