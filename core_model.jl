using Agents
using Random
using StatsBase
# Note: Images and ImageFiltering are no longer required as we implemented 
# a hyper-optimized in-place diffusion kernel below.

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
const ANTIFUNGAL_SOURCE_RADIUS = 15 

const NUTRIENT_EXPOSURE_MODE = SINGLE_SHOCK
const NUTRIENT_INJECTION_STEP = 0          

const MAX_BIOMASS_PER_PX = 3.0 
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
    # Temp fields to prevent Dict allocations
    n_demand::Float64
    f_demand::Float64
    f_release::Float64
end

@agent struct PCDMinusCell(GridAgent{2})
    alive::Bool
    ANTIFUNGAL_exposure_time::Float64
    dead_necrosis::Bool
    dead_starvation::Bool
    bound_ANTIFUNGAL::Float64
    biomass::Float64
    internal_nutrients::Float64 
    # Temp fields to prevent Dict allocations
    n_demand::Float64
    f_demand::Float64
    f_release::Float64
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
    pcd_minus_death_modifier::Float64

    # --- Pre-allocated Buffers for Zero-Allocation Loops ---
    rhs_n::Matrix{Float64}
    rhs_f::Matrix{Float64}
    u_n::Matrix{Float64}
    u_f::Matrix{Float64}
    next_u_n::Matrix{Float64}
    next_u_f::Matrix{Float64}
    planned_biomass::Matrix{Float64}
    newborn_spots::Vector{Tuple{Tuple{Int,Int}, Float64, Float64}}
    agent_ids::Vector{Int}
    candidate_spots::Vector{Tuple{Int,Int}}
    candidate_weights::Vector{Float64}
end

function initialize_model(AgentType::Type, starting_positions::Vector{Tuple{Int, Int}};
    init_nutrient = 12.0, spatial_mode = POINT_SOURCES, source_dose = 10000.0,
    max_binding_apop = 2.5, max_binding_necro = 0.42, apop_leak_rate = 0.5,
    apop_duration = 2.0, apop_point_of_no_return = 1.0, resuscitation_thresh = 0.0, fungistatic_thresh = 0.75,
    reservoir_fraction = 0.1, diffusion_antifungal = 0.3, pcd_minus_death_modifier = 0.5)

    space = GridSpace((GRID_SIZE_PX, GRID_SIZE_PX); periodic=false)
    
    # Init zero-allocation buffers
    rhs_n = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    rhs_f = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    u_n = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    u_f = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    next_u_n = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    next_u_f = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    planned_biomass = zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX)
    
    newborn_spots = Tuple{Tuple{Int,Int}, Float64, Float64}[]
    sizehint!(newborn_spots, 2000)
    agent_ids = Int[]
    sizehint!(agent_ids, GRID_SIZE_PX * GRID_SIZE_PX)
    candidate_spots = Tuple{Int,Int}[]
    candidate_weights = Float64[]
    sizehint!(candidate_spots, 500)
    sizehint!(candidate_weights, 500)

    props = PetriDishProperties(
        fill(Float64(init_nutrient), GRID_SIZE_PX, GRID_SIZE_PX),
        fill(0.0, GRID_SIZE_PX, GRID_SIZE_PX),
        [1/6 2/3 1/6; 2/3 -10/3 2/3; 1/6 2/3 1/6],
        [1/6 2/3 1/6; 2/3 0.0 2/3; 1/6 2/3 1/6],
        AgentType === PCDPlusCell, 0.0,
        Float64(init_nutrient), spatial_mode, Float64(source_dose), Float64(max_binding_apop), Float64(max_binding_necro),
        Float64(apop_leak_rate), Float64(apop_duration), Float64(apop_point_of_no_return), Float64(resuscitation_thresh), Float64(fungistatic_thresh), 
        Float64(reservoir_fraction), Float64(diffusion_antifungal), Float64(pcd_minus_death_modifier),
        rhs_n, rhs_f, u_n, u_f, next_u_n, next_u_f, planned_biomass, newborn_spots,
        agent_ids, candidate_spots, candidate_weights
    )

    model = StandardABM(AgentType, space; properties=props, model_step! = complex_model_step!)
    init_internal = NEWBORN_BIOMASS * reservoir_fraction

    for pos in starting_positions
        if AgentType === PCDPlusCell
            add_agent!(pos, PCDPlusCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, NEWBORN_BIOMASS, init_internal, true, 0.0, 0.0, 0.0)
        else
            add_agent!(pos, PCDMinusCell, model, true, 0.0, false, false, 0.0, NEWBORN_BIOMASS, init_internal, 0.0, 0.0, 0.0)
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
        adjusted_rate = total_rate * model.pcd_minus_death_modifier
        
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

# Custom, 0-allocation high performance explicit 2D convolution for Replicate padding
function apply_kernel!(dst::Matrix{Float64}, src::Matrix{Float64}, kernel::Matrix{Float64})
    R, C = size(src)
    k11, k12, k13 = kernel[1,1], kernel[1,2], kernel[1,3]
    k21, k22, k23 = kernel[2,1], kernel[2,2], kernel[2,3]
    k31, k32, k33 = kernel[3,1], kernel[3,2], kernel[3,3]
    
    @inbounds for j in 1:C
        jm1 = j == 1 ? 1 : j - 1
        jp1 = j == C ? C : j + 1
        for i in 1:R
            im1 = i == 1 ? 1 : i - 1
            ip1 = i == R ? R : i + 1
            
            v = src[im1, jm1] * k11 + src[i, jm1] * k21 + src[ip1, jm1] * k31 +
                src[im1, j  ] * k12 + src[i, j  ] * k22 + src[ip1, j  ] * k32 +
                src[im1, jp1] * k13 + src[i, jp1] * k23 + src[ip1, jp1] * k33
            
            dst[i, j] = v
        end
    end
end

function complex_model_step!(model)
    # Reset internal tracking structures
    fill!(model.planned_biomass, 0.0)
    empty!(model.newborn_spots)
    empty!(model.agent_ids)

    # Pre-calculate Demands for all agents
    for agent in allagents(model)
        push!(model.agent_ids, agent.id)
        agent.n_demand = 0.0
        agent.f_demand = 0.0
        agent.f_release = 0.0

        x, y = agent.pos
        if agent.alive
            local_n = model.nutrient_layer[y, x]
            local_f = model.ANTIFUNGAL_layer[y, x]
            max_reservoir = agent.biomass * model.reservoir_fraction
            reservoir_deficit = max(0.0, max_reservoir - agent.internal_nutrients)

            if is_apoptotic(agent)
                maintenance_cost = MAINTENANCE_COEFF * agent.biomass * TIME_STEP_DT
                agent.n_demand = maintenance_cost + reservoir_deficit
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
                agent.n_demand = growth_demand_n + maintenance_cost + reservoir_deficit
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
            agent.f_demand = net_change
        else
            agent.f_release = min(-net_change, bound_f)
        end
    end

    shuffle!(model.agent_ids)

    # Process randomized agent decisions
    for id in model.agent_ids
        agent = model[id]
        x, y = agent.pos
        
        if agent.alive
            apply_stress!(agent, model.ANTIFUNGAL_layer[y, x], model)
        end

        if agent.alive && agent.n_demand > 0
            demand = agent.n_demand
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

                    # DIVISION LOGIC (0 allocations map/filter replacement)
                    if agent.biomass >= DIVISION_BIOMASS
                        if !can_divide(agent)
                            excess = agent.biomass - DIVISION_BIOMASS
                            agent.biomass = DIVISION_BIOMASS
                            model.total_lost_nutrients += (excess / YIELD_TRUE) 
                        else
                            chosen_spot = nothing
                            empty!(model.candidate_spots)
                            empty!(model.candidate_weights)

                            # Search Immediate Hood
                            for np in (agent.pos, nearby_positions(agent.pos, model, 1)...)
                                npx, npy = np
                                cur_b = 0.0
                                for a in agents_in_position(np, model)
                                    cur_b += a.biomass
                                end
                                cur_b += model.planned_biomass[npy, npx]
                                
                                if cur_b + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX
                                    push!(model.candidate_spots, np)
                                    push!(model.candidate_weights, cur_b == 0.0 ? 1.0 : 0.0) # mark trues
                                end
                            end

                            if !isempty(model.candidate_spots)
                                has_empty = false
                                for w in model.candidate_weights
                                    if w == 1.0; has_empty = true; break; end
                                end

                                total_w = 0.0
                                valid_count = 0
                                for i in 1:length(model.candidate_spots)
                                    p = model.candidate_spots[i]
                                    is_empty_spot = (model.candidate_weights[i] == 1.0)

                                    if !has_empty || is_empty_spot
                                        valid_count += 1
                                        model.candidate_spots[valid_count] = p
                                        
                                        occ_neighbors = 0
                                        for nnp in nearby_positions(p, model, 1)
                                            nbio = 0.0
                                            for a in agents_in_position(nnp, model)
                                                nbio += a.biomass
                                            end
                                            nbio += model.planned_biomass[nnp[2], nnp[1]]
                                            if nbio > 0.0
                                                occ_neighbors += 1
                                            end
                                        end
                                        
                                        w = Float64(occ_neighbors)^3 + 1.0
                                        if length(model.candidate_weights) < valid_count
                                            push!(model.candidate_weights, w)
                                        else
                                            model.candidate_weights[valid_count] = w
                                        end
                                        total_w += w
                                    end
                                end

                                # Manual weighted sampling
                                r = rand() * total_w
                                acc = 0.0
                                for i in 1:valid_count
                                    acc += model.candidate_weights[i]
                                    if r <= acc || i == valid_count
                                        chosen_spot = model.candidate_spots[i]
                                        break
                                    end
                                end

                            elseif rand() < PUSH_PROBABILITY
                                # Fallback push logic
                                empty!(model.candidate_spots)
                                empty!(model.candidate_weights)
                                min_dist = typemax(Int)

                                for p in nearby_positions(agent.pos, model, MAX_PUSH_RADIUS)
                                    px, py = p
                                    cur_b = 0.0
                                    for a in agents_in_position(p, model)
                                        cur_b += a.biomass
                                    end
                                    cur_b += model.planned_biomass[py, px]
                                    if cur_b + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX
                                        dist = (agent.pos[1] - px)^2 + (agent.pos[2] - py)^2
                                        if dist < min_dist
                                            min_dist = dist
                                        end
                                        push!(model.candidate_spots, p)
                                    end
                                end

                                if !isempty(model.candidate_spots)
                                    total_w = 0.0
                                    valid_count = 0
                                    for i in 1:length(model.candidate_spots)
                                        p = model.candidate_spots[i]
                                        dist = (agent.pos[1] - p[1])^2 + (agent.pos[2] - p[2])^2
                                        if dist <= min_dist + 2
                                            valid_count += 1
                                            model.candidate_spots[valid_count] = p
                                            
                                            cur_b = 0.0
                                            for a in agents_in_position(p, model)
                                                cur_b += a.biomass
                                            end
                                            cur_b += model.planned_biomass[p[2], p[1]]
                                            w = MAX_BIOMASS_PER_PX - cur_b
                                            
                                            if length(model.candidate_weights) < valid_count
                                                push!(model.candidate_weights, w)
                                            else
                                                model.candidate_weights[valid_count] = w
                                            end
                                            total_w += w
                                        end
                                    end

                                    r = rand() * total_w
                                    acc = 0.0
                                    for i in 1:valid_count
                                        acc += model.candidate_weights[i]
                                        if r <= acc || i == valid_count
                                            chosen_spot = model.candidate_spots[i]
                                            break
                                        end
                                    end
                                end
                            end

                            if chosen_spot !== nothing
                                agent.biomass -= NEWBORN_BIOMASS 
                                daughter_fraction = NEWBORN_BIOMASS / (agent.biomass + NEWBORN_BIOMASS)
                                daughter_n = agent.internal_nutrients * daughter_fraction
                                agent.internal_nutrients -= daughter_n
                                model.planned_biomass[chosen_spot[2], chosen_spot[1]] += NEWBORN_BIOMASS
                                push!(model.newborn_spots, (chosen_spot, NEWBORN_BIOMASS, daughter_n))
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

        if agent.f_demand > 0
            demand = agent.f_demand
            actual_binding = min(demand, model.ANTIFUNGAL_layer[y, x])
            agent.bound_ANTIFUNGAL += actual_binding
            model.ANTIFUNGAL_layer[y, x] -= actual_binding
        elseif agent.f_release > 0
            release = agent.f_release
            agent.bound_ANTIFUNGAL -= release
            model.ANTIFUNGAL_layer[y, x] += release
        end
    end

    for (pos, b, n) in model.newborn_spots
        cur_b = 0.0
        for a in agents_in_position(pos, model)
            cur_b += a.biomass
        end
        if cur_b + b <= MAX_BIOMASS_PER_PX + 0.01 
            if model.is_pcd_plus
                add_agent!(pos, PCDPlusCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, b, n, true, 0.0, 0.0, 0.0)
            else
                add_agent!(pos, PCDMinusCell, model, true, 0.0, false, false, 0.0, b, n, 0.0, 0.0, 0.0)
            end
        else
            model.total_lost_nutrients += (b / YIELD_TRUE) + n
        end
    end

    # Explicit In-Place 2D Diffusion
    alpha_n = DIFFUSION_NUTRIENT * TIME_STEP_DT
    alpha_f = model.diffusion_antifungal * TIME_STEP_DT
    
    # Calculate RHS natively
    apply_kernel!(model.rhs_n, model.nutrient_layer, model.laplacian_kernel)
    apply_kernel!(model.rhs_f, model.ANTIFUNGAL_layer, model.laplacian_kernel)
    
    @inbounds for i in eachindex(model.rhs_n)
        model.rhs_n[i] = model.nutrient_layer[i] + (alpha_n / 2.0) * model.rhs_n[i]
        model.rhs_f[i] = model.ANTIFUNGAL_layer[i] + (alpha_f / 2.0) * model.rhs_f[i]
        model.u_n[i] = model.nutrient_layer[i]
        model.u_f[i] = model.ANTIFUNGAL_layer[i]
    end
    
    denom_n = 1.0 + (5.0 / 3.0) * alpha_n
    denom_f = 1.0 + (5.0 / 3.0) * alpha_f

    for _ in 1:DIFFUSION_ITERATIONS
        apply_kernel!(model.next_u_n, model.u_n, model.neighbor_kernel)
        apply_kernel!(model.next_u_f, model.u_f, model.neighbor_kernel)
        @inbounds for i in eachindex(model.u_n)
            model.u_n[i] = (model.rhs_n[i] + (alpha_n / 2.0) * model.next_u_n[i]) / denom_n
            model.u_f[i] = (model.rhs_f[i] + (alpha_f / 2.0) * model.next_u_f[i]) / denom_f
        end
    end

    @inbounds for i in eachindex(model.nutrient_layer)
        model.nutrient_layer[i] = max(model.u_n[i], 0.0)
        model.ANTIFUNGAL_layer[i] = max(model.u_f[i], 0.0)
    end
end

# ==========================================
# --- HEADLESS EXPERIMENT RUNNER ---
# ==========================================
"""
Runs the simulation entirely in memory and returns the final alive count.
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