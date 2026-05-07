# ==========================================
# EXPERIMENT 5: "Population Size & Spatial Mode"
# ==========================================
# Goal: Test how the initial number of cells affects survival and 
# altruistic provisioning efficiency under different spatial stress modes.

using DataFrames
using CSV
using Printf
using Plots
using Random

# Load the core engine
include(joinpath(@__DIR__, "core_model.jl"))

# --- Custom History Tracker ---
function run_history_simulation(AgentType::Type, pop_size::Int; kwargs...)
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    
    # High local density: pack them in the center
    cx, cy = (GRID_SIZE_PX + 1) / 2.0, (GRID_SIZE_PX + 1) / 2.0
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    starting_positions = all_pos[1:min(pop_size, length(all_pos))]
    
    # 2. Initialize Model
    model = initialize_model(AgentType, starting_positions; kwargs...)
    
    h_alive = Int[]; h_apop = Int[]; h_necro = Int[]
    
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

function run_experiment_5()
    println("--- Starting Experiment 5: Population Size & Spatial Mode Sweep ---")
    
    # 1. Define the parameters for the sweep
    spatial_modes = [UNIFORM, POINT_SOURCES]
    pop_sizes = [100, 500, 5000, 10000]           # From small pioneer colonies to established populations
    uniform_dose = 2.0                       # Smaller dose for uniform exposure
    point_dose = 50.0                        # Challenging dose for localized sources
    fixed_nutrient = 12.0                    # Standard nutrients
    
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # 2. Prepare DataFrame
    results_df = DataFrame(
        Spatial_Mode = String[], 
        Init_Pop_Size = Int[], 
        Time_Hours = Float64[],
        Genotype = String[],
        Alive = Int[], 
        Dead_Apop = Int[],
        Dead_Necro = Int[],
        Doubling_Time_Hrs = Float64[]
    )
    
    total_runs = length(pop_sizes) * length(spatial_modes)
    current_run = 1
    plot_grid = []

    # 3. Execute Sweep
    for size in pop_sizes
        for mode in spatial_modes
            current_dose = mode == UNIFORM ? uniform_dose : point_dose
            @printf("Running %d/%d (Pop: %d, Mode: %s, Dose: %.1f)...\n", current_run, total_runs, size, string(mode), current_dose)
            
            # PCD+ Colony
            alive_plus, apop_plus, necro_plus = run_history_simulation(
                PCDPlusCell, size,
                spatial_mode = mode, 
                source_dose = current_dose, 
                init_nutrient = fixed_nutrient
            )
            
            # PCD- Colony
            alive_minus, apop_minus, necro_minus = run_history_simulation(
                PCDMinusCell, size,
                spatial_mode = mode, 
                source_dose = current_dose,
                init_nutrient = fixed_nutrient
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
                push!(results_df, (string(mode), size, time_axis[j], "PCD+", alive_plus[j], apop_plus[j], necro_plus[j], td_plus))
                push!(results_df, (string(mode), size, time_axis[j], "PCD-", alive_minus[j], apop_minus[j], necro_minus[j], td_minus))
            end
            
            # Generate subplot
            p = plot(title="Pop: $size | $mode", titlefontsize=9, legend=false, grid=false, xaxis=false, yaxis=false)
            if mode == spatial_modes[1]; yaxis!(p, true); ylabel!(p, "Cells"); end
            if size == pop_sizes[end]; xaxis!(p, true); xlabel!(p, "Hours"); end
            if current_run == 1; plot!(p, legend=:topleft, legendfontsize=6); end
            
            plot!(p, time_axis, alive_plus, color=:blue, linewidth=2, label="PCD+ Alive")
            plot!(p, time_axis, alive_minus, color=:red, linewidth=2, label="PCD- Alive")
            
            # Add vertical line to show injection timing
            vline!(p, [t_phase_hours], color=:gray, linestyle=:dash, alpha=0.5, label="")
            
            push!(plot_grid, p)
            current_run += 1
        end
    end
    
    # 4. Save Data and Plots
    data_dir = joinpath(@__DIR__, "..", "Data")
    mkpath(data_dir) # Automatically creates the "Data" folder if it doesn't exist yet!
    csv_path = joinpath(data_dir, "2026-05-11_Exp5_PopSpatial_TimeSeries.csv" )
    CSV.write(csv_path, results_df)
    println("\nData successfully saved to: ", csv_path)
    
    println("Generating Grid Plot...")
    final_plot = plot(plot_grid..., layout=(length(pop_sizes), length(spatial_modes)), size=(800, 1000), 
                      plot_title="Survival vs Population Size & Spatial Stress Mode")
    fig_dir = joinpath(@__DIR__, "..", "Figures")
    mkpath(fig_dir)
    plot_path = joinpath(fig_dir, "2026-05-11_Exp5_PopSpatial_Grid.png")
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: ", plot_path)
    
    println("\n=== Experiment 5 Complete! ===")
end

run_experiment_5()