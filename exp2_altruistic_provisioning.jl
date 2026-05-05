# ==========================================
# EXPERIMENT 2: "Altruistic Provisioning"
# ==========================================
# Goal: Test if controlled nutrient leaking during apoptosis 
# provides a survival advantage in nutrient-scarce environments.

using DataFrames
using CSV
using Printf
using Plots 

# Load the core engine
include("C:/Users/Mathi/Downloads/UVA Ams/Project/Code/core_model.jl")

# --- Custom History Tracker ---
function run_history_simulation(AgentType::Type; kwargs...)
    cx, cy = (GRID_SIZE_PX + 1) / 2.0, (GRID_SIZE_PX + 1) / 2.0
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    starting_positions = all_pos[1:min(INITIAL_CELLS, length(all_pos))]
    
    model = initialize_model(AgentType, starting_positions; kwargs...)
    
    h_alive = Int[]; h_apop = Int[]; h_necro = Int[]
    
    push!(h_alive, count(a -> a.alive, allagents(model)))
    push!(h_apop, count(a -> is_dead_apoptosis(a), allagents(model)))
    push!(h_necro, count(a -> a.dead_necrosis, allagents(model)))
    
    # Determine dynamic injection step: step 73 for Uniform, step 1 (time 0) for Point Sources
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
        
        push!(h_alive, count(a -> a.alive, allagents(model)))
        push!(h_apop, count(a -> is_dead_apoptosis(a), allagents(model)))
        push!(h_necro, count(a -> a.dead_necrosis, allagents(model)))
    end
    
    return h_alive, h_apop, h_necro
end

function run_experiment_2()
    println("--- Starting Experiment 2: Altruistic Provisioning ---")
    
    # 1. Define the parameters for the sweep
    nutrient_levels = [4.0, 8.0, 12.0, 16.0] # Famine to Feast
    reservoir_fractions = [0,0.1,0.3, 0.9]    # Low vs High reservoir capacity
    fixed_dose = 1.0                       # Changed to a lower dose since UNIFORM 100 is instant death
    
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # 2. Prepare DataFrame
    results_df = DataFrame(
        Init_Nutrients = Float64[], 
        Reservoir_Fraction = Float64[], 
        Time_Hours = Float64[],
        Genotype = String[],
        Alive = Int[], 
        Dead_Apop = Int[],
        Dead_Necro = Int[],
        Doubling_Time_Hrs = Float64[]
    )
    
    total_runs = length(nutrient_levels) * length(reservoir_fractions)
    current_run = 1
    plot_grid = []
    
    # Array to store results temporarily so we can calculate max y-axis
    simulation_results = []

    # 3. Execute Sweep
    for nut in nutrient_levels
        for res_frac in reservoir_fractions
            @printf("Running %d/%d (Nutrients: %.1f, Res Capacity: %.1f)...\n", current_run, total_runs, nut, res_frac)
            
            # PCD+ Colony
            alive_plus, apop_plus, necro_plus = run_history_simulation(
                PCDPlusCell, 
                spatial_mode = UNIFORM, 
                source_dose = fixed_dose, 
                init_nutrient = nut,
                reservoir_fraction = res_frac
            )
            
            # PCD- Colony
            alive_minus, apop_minus, necro_minus = run_history_simulation(
                PCDMinusCell, 
                spatial_mode = UNIFORM, 
                source_dose = fixed_dose,
                init_nutrient = nut,
                reservoir_fraction = res_frac
            )
            
            # Calculate Doubling Times
            injection_step = 73 # UNIFORM uses step 73
            t_phase_hours = injection_step * TIME_STEP_DT
            n0_plus = alive_plus[1]
            nt_plus = alive_plus[injection_step + 1] 
            td_plus = (nt_plus > n0_plus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_plus / n0_plus)) : NaN
            
            n0_minus = alive_minus[1]
            nt_minus = alive_minus[injection_step + 1]
            td_minus = (nt_minus > n0_minus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_minus / n0_minus)) : NaN
            
            for i in 1:length(time_axis)
                push!(results_df, (nut, res_frac, time_axis[i], "PCD+", alive_plus[i], apop_plus[i], necro_plus[i], td_plus))
                push!(results_df, (nut, res_frac, time_axis[i], "PCD-", alive_minus[i], apop_minus[i], necro_minus[i], td_minus))
            end
            
            # Cache results for plotting later
            push!(simulation_results, (nut, res_frac, alive_plus, alive_minus, t_phase_hours))
            
            current_run += 1
        end
    end
    
    # 3.5 Generate Subplots with a GLOBAL linked y-axis
    global_max_y = 0.0
    for res in simulation_results
        global_max_y = max(global_max_y, maximum(res[3]), maximum(res[4]))
    end
    # Add 5% padding to the top of the y-axis
    global_max_y = max(global_max_y * 1.05, 1.0)
    
    current_plot = 1
    for nut in nutrient_levels
        for res_frac in reservoir_fractions
            res = simulation_results[current_plot]
            _, _, alive_plus, alive_minus, t_phase_hours = res
            
            # Generate subplot
            p = plot(title="Nut: $nut | Res: $res_frac", titlefontsize=9, legend=false, grid=false, xaxis=false, yaxis=false)
            if res_frac == reservoir_fractions[1]; yaxis!(p, true); ylabel!(p, "Cells"); end
            if nut == nutrient_levels[end]; xaxis!(p, true); xlabel!(p, "Hours"); end
            if current_plot == 1; plot!(p, legend=:topleft, legendfontsize=6); end
            
            # Apply the GLOBAL y-axis limit
            plot!(p, ylims=(0, global_max_y))
            plot!(p, time_axis, alive_plus, color=:blue, linewidth=2, label="PCD+ Alive")
            plot!(p, time_axis, alive_minus, color=:red, linewidth=2, label="PCD- Alive")
            
            push!(plot_grid, p)
            current_plot += 1
        end
    end
    
    # 4. Save Data and Plots (Updated filenames to reflect sweep)
    csv_path = "C:/Users/Mathi/Downloads/UVA Ams/Project/Data/2026-03-23_Exp2_ReservoirSweep_TimeSeries.csv" 
    CSV.write(csv_path, results_df)
    println("\nData successfully saved to: ", csv_path)
    
    println("Generating 4x4 Grid Plot...")
    final_plot = plot(plot_grid..., layout=(length(nutrient_levels), length(reservoir_fractions)), size=(1000, 1000), 
                      plot_title="Altruistic Provisioning: Survival vs Starvation & Reservoir Capacity")
    
    plot_path = "C:/Users/Mathi/Downloads/UVA Ams/Project/Figures/2026-03-23_Exp2_ReservoirSweep_Grid.png"
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: ", plot_path)
    
    println("\n=== Experiment 2 Complete! ===")
end

run_experiment_2()