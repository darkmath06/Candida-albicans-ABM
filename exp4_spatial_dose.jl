# ==========================================
# EXPERIMENT 2: "Spatial Mode vs Dose Sweep"
# ==========================================
# Goal: Compare Uniform AF (injected at 24h) and Point Source AF 
# (injected at step 1) across different dosage levels.

using DataFrames
using CSV
using Printf
using Plots 

# Load the core engine
include(joinpath(@__DIR__, "core_model.jl"))

# --- Custom History Tracker ---
function run_history_simulation(AgentType::Type; kwargs...)
    cx, cy = (GRID_SIZE_PX + 1) / 2.0, (GRID_SIZE_PX + 1) / 2.0
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    starting_positions = all_pos[1:min(INITIAL_CELLS, length(all_pos))]
    
    model = initialize_model(AgentType, starting_positions; kwargs...)
    
    h_alive = Int[]; h_apop = Int[]; h_necro = Int[]
    
    # Baseline (Step 0)
    push!(h_alive, count(a -> a.alive, allagents(model)))
    push!(h_apop, count(a -> is_dead_apoptosis(a), allagents(model)))
    push!(h_necro, count(a -> a.dead_necrosis, allagents(model)))
    
    # Determine the dynamic injection step: 24h = step 72, immediate = step 1
    injection_step = model.spatial_mode == UNIFORM ? 72 : 1
    
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
    println("--- Starting Experiment 2: Spatial Mode vs Dose Sweep ---")
    
    # 1. Define the parameters for the sweep
    spatial_modes = [UNIFORM, POINT_SOURCES]
    uniform_doses = [0.5, 1.0, 1.5, 2.0]       # Dosages for UNIFORM mode
    point_doses = [5.0, 50.0, 200.0, 10000.0]  # Dosages for POINT_SOURCES mode
    fixed_nutrient = 12.0                      # Hold nutrients constant
    fixed_res_frac = 0.7                       # Hold reservoir capacity constant
    
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # 2. Prepare DataFrame
    results_df = DataFrame(
        Spatial_Mode = String[], 
        Dose = Float64[], 
        Time_Hours = Float64[],
        Genotype = String[],
        Alive = Int[], 
        Dead_Apop = Int[],
        Dead_Necro = Int[],
        Doubling_Time_Hrs = Float64[]
    )
    
    total_runs = length(uniform_doses) * length(spatial_modes)
    current_run = 1
    plot_grid = []

    # 3. Execute Sweep
    for i in 1:length(uniform_doses)
        for mode in spatial_modes
            dose = mode == UNIFORM ? uniform_doses[i] : point_doses[i]
            
            @printf("Running %d/%d (Mode: %s, Dose: %.1f)...\n", current_run, total_runs, string(mode), dose)
            
            # PCD+ Colony
            alive_plus, apop_plus, necro_plus = run_history_simulation(
                PCDPlusCell, 
                spatial_mode = mode, 
                source_dose = dose, 
                init_nutrient = fixed_nutrient,
                reservoir_fraction = fixed_res_frac
            )
            
            # PCD- Colony
            alive_minus, apop_minus, necro_minus = run_history_simulation(
                PCDMinusCell, 
                spatial_mode = mode, 
                source_dose = dose,
                init_nutrient = fixed_nutrient,
                reservoir_fraction = fixed_res_frac
            )
            
            # Calculate Doubling Times before stress hits
            injection_step = mode == UNIFORM ? 72 : 1
            t_phase_hours = injection_step * TIME_STEP_DT
            
            n0_plus = alive_plus[1]
            nt_plus = alive_plus[injection_step + 1] 
            td_plus = (nt_plus > n0_plus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_plus / n0_plus)) : NaN
            
            n0_minus = alive_minus[1]
            nt_minus = alive_minus[injection_step + 1]
            td_minus = (nt_minus > n0_minus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_minus / n0_minus)) : NaN
            
            for j in 1:length(time_axis)
                push!(results_df, (string(mode), dose, time_axis[j], "PCD+", alive_plus[j], apop_plus[j], necro_plus[j], td_plus))
                push!(results_df, (string(mode), dose, time_axis[j], "PCD-", alive_minus[j], apop_minus[j], necro_minus[j], td_minus))
            end
            
            # Generate subplot
            p = plot(title="Mode: $mode | Dose: $dose", titlefontsize=9, legend=false, grid=false, xaxis=false, yaxis=false)
            if mode == spatial_modes[1]; yaxis!(p, true); ylabel!(p, "Cells"); end
            if i == length(uniform_doses); xaxis!(p, true); xlabel!(p, "Hours"); end
            if current_run == 1; plot!(p, legend=:topleft, legendfontsize=6); end
            
            plot!(p, time_axis, alive_plus, color=:blue, linewidth=2, label="PCD+ Alive")
            plot!(p, time_axis, alive_minus, color=:red, linewidth=2, label="PCD- Alive")
            
            # Add vertical lines to show injection timing
            vline!(p, [t_phase_hours], color=:gray, linestyle=:dash, alpha=0.5, label="")
            
            push!(plot_grid, p)
            current_run += 1
        end
    end
    
    # 4. Save Data and Plots
    data_dir = joinpath(@__DIR__, "..", "Data")
    mkpath(data_dir) # Automatically creates the "Data" folder if it doesn't exist yet!
    
    csv_path = joinpath(data_dir, "2026-03-25_Exp2_SpatialDoseSweep_TimeSeries.csv") 
    CSV.write(csv_path, results_df)
    println("\nData successfully saved to: ", csv_path)
    
    println("Generating Grid Plot...")
    final_plot = plot(plot_grid..., layout=(length(uniform_doses), length(spatial_modes)), size=(800, 1000), 
                      plot_title="Survival vs Spatial Mode & Dose Severity")
    
    fig_dir = joinpath(@__DIR__, "..", "Figures")
    mkpath(fig_dir) # Automatically creates the "Figures" folder if it doesn't exist yet!
    
    plot_path = joinpath(fig_dir, "2026-03-25_Exp2_SpatialDoseSweep_Grid.png")
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: ", plot_path)
    
    println("\n=== Experiment 2 Complete! ===")
end

run_experiment_2()