# ==========================================
# EXPERIMENT 3: "Antifungal Diffusion vs Dose"
# ==========================================
# Goal: Test how the colony survives localized point-source 
# stress depending on whether the drug stays localized (low diffusion) 
# or spreads rapidly (high diffusion) at different severities.

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

function run_experiment_3()
    println("--- Starting Experiment 3: Diffusion vs Dose Sweep (Spatial) ---")
    
    # We test 2 severities for UNIFORM and 2 for POINT_SOURCES to keep a clean 4x4 plot grid
    configs = [
        (UNIFORM, 1.5),
        (UNIFORM, 2.0),
        (POINT_SOURCES, 50.0),
        (POINT_SOURCES, 200.0)
    ]
    diffusion_rates = [0.05, 0.15, 0.3, 0.6] # How fast the drug spreads
    
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # 2. Prepare DataFrame
    results_df = DataFrame(
        Spatial_Mode = String[],
        Dose = Float64[], 
        Diffusion_Rate = Float64[], 
        Time_Hours = Float64[],
        Genotype = String[],
        Alive = Int[], 
        Dead_Apop = Int[],
        Dead_Necro = Int[],
        Doubling_Time_Hrs = Float64[]
    )
    
    total_runs = length(configs) * length(diffusion_rates)
    current_run = 1
    plot_grid = []

    # 3. Execute Sweep
    for (mode, dose) in configs
        for diff_rate in diffusion_rates
            @printf("Running %d/%d (Mode: %s, Dose: %.1f, Diffusion: %.2f)...\n", current_run, total_runs, string(mode), dose, diff_rate)
            
            # PCD+ Colony 
            alive_plus, apop_plus, necro_plus = run_history_simulation(
                PCDPlusCell, 
                spatial_mode = mode, 
                source_dose = dose, 
                diffusion_antifungal = diff_rate
            )
            
            # PCD- Colony 
            alive_minus, apop_minus, necro_minus = run_history_simulation(
                PCDMinusCell, 
                spatial_mode = mode, 
                source_dose = dose,
                diffusion_antifungal = diff_rate
            )
            
            # Calculate Doubling Times before stress hits
            injection_step = mode == UNIFORM ? 73 : 1
            t_phase_hours = injection_step * TIME_STEP_DT
            
            n0_plus = alive_plus[1]
            nt_plus = alive_plus[injection_step + 1] 
            td_plus = (nt_plus > n0_plus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_plus / n0_plus)) : NaN
            
            n0_minus = alive_minus[1]
            nt_minus = alive_minus[injection_step + 1]
            td_minus = (nt_minus > n0_minus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_minus / n0_minus)) : NaN
            
            for i in 1:length(time_axis)
                push!(results_df, (string(mode), dose, diff_rate, time_axis[i], "PCD+", alive_plus[i], apop_plus[i], necro_plus[i], td_plus))
                push!(results_df, (string(mode), dose, diff_rate, time_axis[i], "PCD-", alive_minus[i], apop_minus[i], necro_minus[i], td_minus))
            end
            
            # Generate subplot
            p = plot(title="$mode($(dose)) | Diff: $diff_rate", titlefontsize=7, legend=false, grid=false, xaxis=false, yaxis=false)
            if diff_rate == diffusion_rates[1]; yaxis!(p, true); ylabel!(p, "Cells"); end
            if (mode, dose) == configs[end]; xaxis!(p, true); xlabel!(p, "Hours"); end
            if current_run == 1; plot!(p, legend=:topleft, legendfontsize=5); end
            
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
    
    csv_path = joinpath(data_dir, "2026-03-27_Exp3_DiffusionDose_Spatial_TimeSeries.csv") 
    CSV.write(csv_path, results_df)
    println("\nData successfully saved to: ", csv_path)
    
    println("Generating 4x4 Grid Plot...")
    final_plot = plot(plot_grid..., layout=(length(configs), length(diffusion_rates)), size=(1200, 1000), 
                      plot_title="Survival vs Spatial Dose & Antifungal Diffusion Rate")
    
    fig_dir = joinpath(@__DIR__, "..", "Figures")
    mkpath(fig_dir) # Automatically creates the "Figures" folder if it doesn't exist yet!
    
    plot_path = joinpath(fig_dir, "2026-03-27_Exp3_DiffusionDose_Spatial_Grid.png")
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: ", plot_path)
    
    println("\n=== Experiment 3 Complete! ===")
end

run_experiment_3()