using Agents
using Random
using StatsBase
using Images
using ImageFiltering
using Plots # Added for visual petri dish generation

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
@agent struct CandidaCell(GridAgent{2})
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
    
    # --- THE EVOLUTIONARY TRAIT ---
    apoptosis_susceptibility::Float64 

    # Optimization: Store demands on the agent
    n_demand::Float64
    f_demand::Float64
    f_release::Float64
end

is_apoptotic(a::CandidaCell) = a.is_apoptotic
can_divide(a::CandidaCell) = a.can_divide
is_dead_apoptosis(a::CandidaCell) = a.dead_apoptosis

set_dead_apoptosis!(a::CandidaCell) = begin
    a.alive = false
    a.is_apoptotic = false
    a.dead_apoptosis = true
end

# ==========================================
# --- DYNAMIC PROPERTIES (Sweepable) ---
# ==========================================
mutable struct PetriDishProperties
    nutrient_layer::Matrix{Float64}
    ANTIFUNGAL_layer::Matrix{Float64}
    laplacian_kernel::Matrix{Float64}
    neighbor_kernel::Matrix{Float64}
    total_lost_nutrients::Float64 
    
    mutation_rate::Float64
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

    planned_biomass::Matrix{Float64}
    newborn_spots::Vector{Tuple{Tuple{Int,Int}, Float64, Float64, Float64}} 
    shuffled_ids::Vector{Int}
    
    avail_spots::Vector{Tuple{Int,Int}}
    avail_weights::Vector{Float64}
    truly_empty::Vector{Tuple{Int,Int}}
    push_cands::Vector{Tuple{Tuple{Int,Int}, Float64}}
    push_valid::Vector{Tuple{Int,Int}}
    push_weights::Vector{Float64}
end

function initialize_model(starting_positions::Vector{Tuple{Int, Int}};
    initial_traits = nothing,
    mutation_rate = 0.05, init_nutrient = 12, spatial_mode = UNIFORM, source_dose = 1.5,
    max_binding_apop = 2.5, max_binding_necro = 0.42, apop_leak_rate = 0.5,
    apop_duration = 2.0, apop_point_of_no_return = 1.0, resuscitation_thresh = 0, fungistatic_thresh = 0.75,
    reservoir_fraction = 0.1, diffusion_antifungal = 0.3)

    space = GridSpace((GRID_SIZE_PX, GRID_SIZE_PX); periodic=false)
    props = PetriDishProperties(
        fill(init_nutrient, GRID_SIZE_PX, GRID_SIZE_PX),
        fill(0.0, GRID_SIZE_PX, GRID_SIZE_PX),
        [1/6 2/3 1/6; 2/3 -10/3 2/3; 1/6 2/3 1/6],
        [1/6 2/3 1/6; 2/3 0.0 2/3; 1/6 2/3 1/6],
        0.0,
        mutation_rate, init_nutrient, spatial_mode, source_dose, max_binding_apop, max_binding_necro,
        apop_leak_rate, apop_duration, apop_point_of_no_return, resuscitation_thresh, fungistatic_thresh, 
        reservoir_fraction, diffusion_antifungal,
        zeros(Float64, GRID_SIZE_PX, GRID_SIZE_PX),                   
        Tuple{Tuple{Int,Int}, Float64, Float64, Float64}[],           
        Int[],                                                        
        Tuple{Int,Int}[], Float64[], Tuple{Int,Int}[],                
        Tuple{Tuple{Int,Int}, Float64}[], Tuple{Int,Int}[], Float64[] 
    )

    model = StandardABM(CandidaCell, space; properties=props, model_step! = complex_model_step!)
    init_internal = NEWBORN_BIOMASS * reservoir_fraction

    for (i, pos) in enumerate(starting_positions)
        # --- THE FIX ---
        # Instead of randomly seeding new cells, start everyone at exactly 0.5 (unless carrying over from a previous passage)
        initial_trait = initial_traits !== nothing ? initial_traits[i] : 0.5 
        add_agent!(pos, CandidaCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, NEWBORN_BIOMASS, init_internal, true, initial_trait, 0.0, 0.0, 0.0)
    end
    return model
end

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

function apply_stress!(agent::CandidaCell, local_ANTIFUNGAL::Float64, model)
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
                if rand(abmrng(model)) < prob_death
                    prob_apop_given_death = agent.apoptosis_susceptibility
                    if rand(abmrng(model)) < prob_apop_given_death
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

function complex_model_step!(model)
    fill!(model.planned_biomass, 0.0)
    empty!(model.newborn_spots)
    empty!(model.shuffled_ids)

    for agent in allagents(model)
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
        
        push!(model.shuffled_ids, agent.id)
    end

    shuffle!(model.shuffled_ids)

    for id in model.shuffled_ids
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

                    if agent.biomass >= DIVISION_BIOMASS
                        if !can_divide(agent)
                            excess = agent.biomass - DIVISION_BIOMASS
                            agent.biomass = DIVISION_BIOMASS
                            model.total_lost_nutrients += (excess / YIELD_TRUE) 
                        else
                            chosen_spot = nothing
                            empty!(model.avail_spots)
                            empty!(model.avail_weights)
                            empty!(model.truly_empty)
                            
                            function check_and_add_spot(p)
                                cur_b = 0.0
                                for a in agents_in_position(p, model); cur_b += a.biomass; end
                                total_b = cur_b + model.planned_biomass[p[2], p[1]]
                                if total_b + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX
                                    push!(model.avail_spots, p)
                                    if total_b == 0.0
                                        push!(model.truly_empty, p)
                                    end
                                end
                            end
                            
                            check_and_add_spot(agent.pos)
                            for p in nearby_positions(agent.pos, model, 1)
                                check_and_add_spot(p)
                            end
                            
                            if !isempty(model.avail_spots)
                                candidate_pool = !isempty(model.truly_empty) ? model.truly_empty : model.avail_spots
                                
                                for s in candidate_pool
                                    occupied_neighbors = 0
                                    for np in nearby_positions(s, model, 1)
                                        cur_b_np = 0.0
                                        for a in agents_in_position(np, model); cur_b_np += a.biomass; end
                                        if (cur_b_np + model.planned_biomass[np[2], np[1]]) > 0.0
                                            occupied_neighbors += 1
                                        end
                                    end
                                    push!(model.avail_weights, Float64(occupied_neighbors)^3 + 1.0)
                                end
                                
                                chosen_idx = sample(1:length(candidate_pool), Weights(model.avail_weights))
                                chosen_spot = candidate_pool[chosen_idx]
                                
                            elseif rand() < PUSH_PROBABILITY
                                empty!(model.push_cands)
                                empty!(model.push_valid)
                                empty!(model.push_weights)
                                min_dist = Inf
                                
                                for p in nearby_positions(agent.pos, model, MAX_PUSH_RADIUS)
                                    cur_b = 0.0
                                    for a in agents_in_position(p, model); cur_b += a.biomass; end
                                    if cur_b + model.planned_biomass[p[2], p[1]] + NEWBORN_BIOMASS <= MAX_BIOMASS_PER_PX
                                        dist = (x - p[1])^2 + (y - p[2])^2
                                        push!(model.push_cands, (p, Float64(dist)))
                                        if dist < min_dist
                                            min_dist = Float64(dist)
                                        end
                                    end
                                end
                                
                                if !isempty(model.push_cands)
                                    for (p, dist) in model.push_cands
                                        if dist <= min_dist + 2.0
                                            cur_b = 0.0
                                            for a in agents_in_position(p, model); cur_b += a.biomass; end
                                            w = MAX_BIOMASS_PER_PX - (cur_b + model.planned_biomass[p[2], p[1]])
                                            push!(model.push_valid, p)
                                            push!(model.push_weights, w)
                                        end
                                    end
                                    if !isempty(model.push_valid)
                                        chosen_idx = sample(1:length(model.push_valid), Weights(model.push_weights))
                                        chosen_spot = model.push_valid[chosen_idx]
                                    end
                                end
                            end

                            if chosen_spot !== nothing
                                agent.biomass -= NEWBORN_BIOMASS 
                                daughter_fraction = NEWBORN_BIOMASS / (agent.biomass + NEWBORN_BIOMASS)
                                daughter_n = agent.internal_nutrients * daughter_fraction
                                agent.internal_nutrients -= daughter_n
                                model.planned_biomass[chosen_spot[2], chosen_spot[1]] += NEWBORN_BIOMASS
                                push!(model.newborn_spots, (chosen_spot, NEWBORN_BIOMASS, daughter_n, agent.apoptosis_susceptibility))
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
            actual_binding = min(agent.f_demand, model.ANTIFUNGAL_layer[y, x])
            agent.bound_ANTIFUNGAL += actual_binding
            model.ANTIFUNGAL_layer[y, x] -= actual_binding
        elseif agent.f_release > 0
            agent.bound_ANTIFUNGAL -= agent.f_release
            model.ANTIFUNGAL_layer[y, x] += agent.f_release
        end
    end

    for (pos, b, n, parent_trait) in model.newborn_spots
        cur_b = 0.0
        for a in agents_in_position(pos, model)
            cur_b += a.biomass
        end
        if cur_b + b <= MAX_BIOMASS_PER_PX + 0.01 
            mut_noise = randn(abmrng(model)) * model.mutation_rate
            new_trait = clamp(parent_trait + mut_noise, 0.0, 1.0)
            
            add_agent!(pos, CandidaCell, model, true, false, 0.0, 0.0, false, false, false, 0.0, b, n, true, new_trait, 0.0, 0.0, 0.0)
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
function run_headless_simulation(; passages=5, passage_fraction=0.1, record_visuals=false, run_id="Experiment", kwargs...)
    cx, cy = (GRID_SIZE_PX + 1) / 2.0, (GRID_SIZE_PX + 1) / 2.0
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    
    current_traits = nothing
    final_alive, final_apop, final_necro = 0, 0, 0
    mean_susceptibility = NaN 
    
    # Store population history across passages
    trait_history = Float64[]
    alive_history = Int[]
    apop_history = Int[]
    necro_history = Int[]
    
    for passage in 1:passages
        num_starting = current_traits !== nothing ? length(current_traits) : min(INITIAL_CELLS, length(all_pos))
        starting_positions = all_pos[1:num_starting]
        
        model = initialize_model(starting_positions; initial_traits=current_traits, kwargs...)
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
        
        alive_cells = filter(a -> a.alive, collect(allagents(model)))
        final_alive = length(alive_cells)
        final_apop = count(a -> is_dead_apoptosis(a), allagents(model))
        final_necro = count(a -> a.dead_necrosis, allagents(model))
        
        # FIX: Explicitly assign NaN (Not a Number) if extinct to avoid 0.0 averaging bias
        mean_susceptibility = isempty(alive_cells) ? NaN : mean(a.apoptosis_susceptibility for a in alive_cells)
        
        # Record population logs
        push!(trait_history, mean_susceptibility)
        push!(alive_history, final_alive)
        push!(apop_history, final_apop)
        push!(necro_history, final_necro)

        # Output the petri dish visual if requested
        if record_visuals
            mkpath("Project/Figures/PetriDish")
            # Create a heatmap of the background drug
            p_dish = heatmap(1:GRID_SIZE_PX, 1:GRID_SIZE_PX, model.ANTIFUNGAL_layer, 
                             color=:Greys, legend=false, aspect_ratio=1.0, showaxis=false,
                             title="Passage $passage (Mean Trait: $(isnan(mean_susceptibility) ? "Extinct" : round(mean_susceptibility, digits=2)))")
            
            # Plot individual cells on top, colored by trait
            if final_alive > 0
                xs = [a.pos[1] for a in alive_cells]
                ys = [a.pos[2] for a in alive_cells]
                cs = [a.apoptosis_susceptibility for a in alive_cells]
                scatter!(p_dish, xs, ys, marker_z=cs, clims=(0.0, 1.0), marker=:circle, 
                         markersize=2, markerstrokewidth=0, colorbar_title="Trait (0=Self, 1=Alt)", 
                         color=:viridis)
            end
            savefig(p_dish, "Project/Figures/PetriDish/$(run_id)_Passage_$(passage).png")
        end
        
        if passage < passages && final_alive > 0
            num_to_sample = max(1, round(Int, final_alive * passage_fraction))
            sampled_cells = sample(alive_cells, num_to_sample, replace=false)
            current_traits = [a.apoptosis_susceptibility for a in sampled_cells]
        elseif final_alive == 0
            break  # Extinction
        end
    end
    
    return final_alive, final_apop, final_necro, mean_susceptibility, trait_history, alive_history, apop_history, necro_history
end