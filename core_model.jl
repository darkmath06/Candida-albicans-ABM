using Agents
using Random
using StatsBase
using Images
using ImageFiltering

# ==========================================
# --- CONSTANT MECHANICS (Non-Swept) ---
# ==========================================
const GRID_SIZE_PX = 150 
const INITIAL_CELLS = 1
const SIMULATION_STEPS = 600
const ANTIFUNGAL_INJECTION_STEP = 1 
const TIME_STEP_DT = 1.0 / 3.0 

@enum ExposureMode SINGLE_SHOCK CONTINUOUS PULSATED
@enum SpatialMode UNIFORM POINT_SOURCES

const ANTIFUNGAL_EXPOSURE_MODE = SINGLE_SHOCK   
const ANTIFUNGAL_SOURCES = [(105, 45), (45, 105), (45, 45), (105, 105)] 
const ANTIFUNGAL_SOURCE_RADIUS = 2 

const NUTRIENT_EXPOSURE_MODE = SINGLE_SHOCK
const NUTRIENT_INJECTION_STEP = 0          

const MAX_BIOMASS_PER_PX = 3 
const DIFFUSION_NUTRIENT = 0.57
const DIFFUSION_ITERATIONS = 15 

const MU_MAX = 0.7            
const MONOD_KS = 5.0           
const MAINTENANCE_COEFF = 0.015 
const YIELD_TRUE = 0.39        
const NEWBORN_BIOMASS = 1.0    
const DIVISION_BIOMASS = 2.0   
const STARVATION_BIOMASS = 0.75 

const PUSH_PROBABILITY = 0.3 
const MAX_PUSH_RADIUS = 10    
const MAX_ANTIFUNGAL_BINDING_LIVE = 0.42       
const K_ON_ANTIFUNGAL = 0.01              
const K_OFF_ANTIFUNGAL = 0.005            

const ANTIFUNGAL_DAMAGE_THRESHOLD = 0.5      
const STRESS_START_TIME = 0.0                
const ASSAY_DURATION_HOURS = 200/60          

const TOTAL_DEATH_MAX = 0.999        
const TOTAL_DEATH_STEEPNESS = 0.8    
const TOTAL_DEATH_C50 = 2.5          
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
# --- DYNAMIC PROPERTIES (Sweepable) ---
# ==========================================
mutable struct PetriDishProperties
    nutrient_layer::Matrix{Float64}
    ANTIFUNGAL_layer::Matrix{Float64}
    laplacian_kernel::Matrix{Float64}
    neighbor_kernel::Matrix{Float64}
    is_pcd_plus::Bool
    total_lost_nutrients::Float64 
    
    # --- The Parameters We Can Sweep ---
    init_nutrient::Float64
    spatial_mode::SpatialMode
    source_dose::Float64
    max_binding_apop::Float64
    max_binding_necro::Float64
    apop_leak_rate::Float64
    apop_duration::Float64
    apop_point_of_no_return::Float64
    resuscitation_thresh::Float64
    fungistatic_thresh::Float64
    reservoir_fraction::Float64
    diffusion_antifungal::Float64
end

function initialize_model(AgentType::Type, starting_positions::Vector{Tuple{Int, Int}};
    init_nutrient = 8, spatial_mode = POINT_SOURCES, source_dose = 10000.0,
    max_binding_apop = 2.5, max_binding_necro = 0.42, apop_leak_rate = 0.5,
    apop_duration = 2.0, apop_point_of_no_return = 1.0, resuscitation_thresh = 0, fungistatic_thresh = 0.75,
    reservoir_fraction = 0.1, diffusion_antifungal = 0.3)

    space = GridSpace((GRID_SIZE_PX, GRID_SIZE_PX); periodic=false)
    props = PetriDishProperties(
        fill(init_nutrient, GRID_SIZE_PX, GRID_SIZE_PX),
        fill(0.0, GRID_SIZE_PX, GRID_SIZE_PX),
        [1/6 2/3 1/6; 2/3 -10/3 2/3; 1/6 2/3 1/6],
        [1/6 2/3 1/6; 2/3 0.0 2/3; 1/6 2/3 1/6],
        AgentType === PCDPlusCell, 0.0,
        init_nutrient, spatial_mode, source_dose, max_binding_apop, max_binding_necro,
        apop_leak_rate, apop_duration, apop_point_of_no_return, resuscitation_thresh, fungistatic_thresh, 
        reservoir_fraction, diffusion_antifungal
    )

    model = StandardABM(AgentType, space; properties=props, model_step! = complex_model_step!)
    init_internal = NEWBORN_BIOMASS * reservoir_fraction

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
# --- EXECUTION LOGIC ---
# ==========================================
function get_death_rates(c::Float64)
    if c < ANTIFUNGAL_DAMAGE_THRESHOLD; return 0.0, 0.0; end
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
        # Check both the environmental threshold and if they haven't passed the point of no return
        if local_ANTIFUNGAL < model.resuscitation_thresh && agent.apoptosis_timer <= model.apop_point_of_no_return
            agent.is_apoptotic = false
            agent.apoptosis_timer = 0.0
        else
            agent.apoptosis_timer += TIME_STEP_DT
            x, y = agent.pos
            leak_fraction = min(1.0, model.apop_leak_rate * TIME_STEP_DT)
            leak_amount = agent.internal_nutrients * leak_fraction
            agent.internal_nutrients -= leak_amount
            model.nutrient_layer[y, x] += leak_amount

            if agent.apoptosis_timer >= model.apop_duration
                set_dead_apoptosis!(agent)
                model.nutrient_layer[y, x] += agent.internal_nutrients
                agent.internal_nutrients = 0.0
            end
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
                        model.nutrient_layer[agent.pos[2], agent.pos[1]] += agent.internal_nutrients
                        agent.internal_nutrients = 0.0
                        model.ANTIFUNGAL_layer[agent.pos[2], agent.pos[1]] += agent.bound_ANTIFUNGAL
                        agent.bound_ANTIFUNGAL = 0.0
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
        
        # Give PCD- an inherent survival advantage by reducing its death rate to 80%
        # This models the "cost of altruism" for the PCD+ strain!
        adjusted_rate = total_rate * 0.5
        
        if adjusted_rate > 0
            prob_death = 1.0 - exp(-adjusted_rate * TIME_STEP_DT)
            if rand() < prob_death
                agent.alive = false
                agent.dead_necrosis = true
                model.nutrient_layer[agent.pos[2], agent.pos[1]] += agent.internal_nutrients
                agent.internal_nutrients = 0.0
                model.ANTIFUNGAL_layer[agent.pos[2], agent.pos[1]] += agent.bound_ANTIFUNGAL
                agent.bound_ANTIFUNGAL = 0.0
            end
        end
    end
end

function complex_model_step!(model)
    n_demands = Dict{Int, Float64}()
    f_demands = Dict{Int, Float64}()
    f_releases = Dict{Int, Float64}()

    for agent in allagents(model)
        x, y = agent.pos
        if agent.alive
            local_n = model.nutrient_layer[y, x]
            local_f = model.ANTIFUNGAL_layer[y, x]
            max_reservoir = agent.biomass * model.reservoir_fraction
            reservoir_deficit = max(0.0, max_reservoir - agent.internal_nutrients)

            if is_apoptotic(agent)
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                n_demands[agent.id] = maintenance_cost + reservoir_deficit
            else
                if local_f >= model.fungistatic_thresh
                    growth_demand_n = 0.0 
                    reservoir_deficit = 0.0 
                else
                    if agent.biomass < DIVISION_BIOMASS
                        mu = MU_MAX * (local_n / (MONOD_KS + local_n))
                        max_possible_growth_biomass = min(mu * agent.biomass * TIME_STEP_DT, DIVISION_BIOMASS - agent.biomass)
                        growth_demand_n = max_possible_growth_biomass / YIELD_TRUE
                    else
                        growth_demand_n = 0.0
                    end
                end
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                n_demands[agent.id] = growth_demand_n + maintenance_cost + reservoir_deficit
            end
        end
        
        local_f = model.ANTIFUNGAL_layer[y, x]
        bound_f = agent.bound_ANTIFUNGAL
        current_max = if agent.alive
            MAX_ANTIFUNGAL_BINDING_LIVE
        elseif is_dead_apoptosis(agent)
            model.max_binding_apop
        else
            model.max_binding_necro
        end

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

                    local_f = model.ANTIFUNGAL_layer[y, x]
                    if local_f < model.fungistatic_thresh && agent.biomass < DIVISION_BIOMASS
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
                        model.ANTIFUNGAL_layer[y, x] += agent.bound_ANTIFUNGAL
                        agent.bound_ANTIFUNGAL = 0.0
                    end
                end

                if agent.alive
                    max_reservoir = agent.biomass * model.reservoir_fraction
                    if agent.internal_nutrients > max_reservoir
                        excess = agent.internal_nutrients - max_reservoir
                        agent.internal_nutrients = max_reservoir
                        model.nutrient_layer[y, x] += excess
                    end

                    if agent.biomass >= DIVISION_BIOMASS
                        if !can_divide(agent)
                            excess = agent.biomass - DIVISION_BIOMASS
                            agent.biomass = DIVISION_BIOMASS
                            model.total_lost_nutrients += (excess / YIELD_TRUE) 
                        else
                            chosen_spot = nothing
                            immediate_hood = [agent.pos; collect(nearby_positions(agent.pos, model, 1))]
                            spot_status = map(immediate_hood) do p
                                cur_b = sum((a.biomass for a in agents_in_position(p, model)), init=0.0)
                                total_b = cur_b + get(planned_biomass, p, 0.0)
                                (pos = p, bio = total_b)
                            end
                            available_spots = filter(s -> s.bio + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX, spot_status)
                            
                            if !isempty(available_spots)
                                truly_empty = filter(s -> s.bio == 0.0, available_spots)
                                candidate_pool = !isempty(truly_empty) ? truly_empty : available_spots
                                weights = map(candidate_pool) do s
                                    occupied_neighbors = count(nearby_positions(s.pos, model, 1)) do np
                                        cur_b = sum((a.biomass for a in agents_in_position(np, model)), init=0.0)
                                        (cur_b + get(planned_biomass, np, 0.0)) > 0.0
                                    end
                                    return Float64(occupied_neighbors)^3 + 1.0 
                                end
                                chosen_idx = sample(1:length(candidate_pool), Weights(weights))
                                chosen_spot = candidate_pool[chosen_idx].pos
                            elseif rand() < PUSH_PROBABILITY
                                block = collect(nearby_positions(agent.pos, model, MAX_PUSH_RADIUS))
                                empty_spots = filter(p -> begin
                                    cur_b = sum((a.biomass for a in agents_in_position(p, model)), init=0.0)
                                    cur_b + get(planned_biomass, p, 0.0) + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX
                                end, block)
                                if !isempty(empty_spots)
                                    min_dist = minimum((x - p[1])^2 + (y - p[2])^2 for p in empty_spots)
                                    best_spots = filter(p -> (x - p[1])^2 + (y - p[2])^2 <= min_dist + 2, empty_spots)
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
        else
            model.total_lost_nutrients += (b / YIELD_TRUE) + n
        end
    end

    alpha_n = DIFFUSION_NUTRIENT * TIME_STEP_DT
    alpha_f = model.diffusion_antifungal * TIME_STEP_DT
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
# --- HEADLESS EXPERIMENT RUNNER ---
# ==========================================
"""
Runs the simulation entirely in memory (no slow plotting) and returns the final alive count.
Pass keyword arguments to sweep specific parameters.
"""
function run_headless_simulation(AgentType::Type; kwargs...)
    cx, cy = (GRID_SIZE_PX + 1) / 2.0, (GRID_SIZE_PX + 1) / 2.0
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    starting_positions = all_pos[1:min(INITIAL_CELLS, length(all_pos))]
    
    model = initialize_model(AgentType, starting_positions; kwargs...)
    
    injection_step = model.spatial_mode == UNIFORM ? 73 : 1
    
    for step in 1:SIMULATION_STEPS
        if step == injection_step
            if model.spatial_mode == UNIFORM
                model.ANTIFUNGAL_layer .= model.source_dose
            elseif model.spatial_mode == POINT_SOURCES
                for (cx, cy) in ANTIFUNGAL_SOURCES
                    for dx in -ANTIFUNGAL_SOURCE_RADIUS:ANTIFUNGAL_SOURCE_RADIUS
                        for dy in -ANTIFUNGAL_SOURCE_RADIUS:ANTIFUNGAL_SOURCE_RADIUS
                            sx, sy = cx + dx, cy + dy
                            if 1 <= sx <= GRID_SIZE_PX && 1 <= sy <= GRID_SIZE_PX
                                model.ANTIFUNGAL_layer[sy, sx] += model.source_dose
                            end
                        end
                    end
                end
            end
        end
        Agents.step!(model, 1)
    end
    
    final_alive = count(a -> a.alive, allagents(model))
    final_apop = count(a -> is_dead_apoptosis(a), allagents(model))
    final_necro = count(a -> a.dead_necrosis, allagents(model))
    
    return final_alive, final_apop, final_necro
end