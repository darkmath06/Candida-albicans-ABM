# ==========================================
# EXPERIMENT 6: "Resuscitation vs No Resuscitation"
# ==========================================
# Goal: Test the colony's ability to bounce back from a transient 
# wave of antifungal stress (point sources that diffuse and dilute) 
# by comparing models where apoptotic resuscitation is enabled vs disabled.

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
    
    # Dynamic injection step: 24h = step 72 for Uniform, immediate = step 1 for Point Sources
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

function run_experiment_6()
    println("--- Starting Experiment 6: Resuscitation OFF vs ON (Spatial Sweep) ---")
    
    # 1. Define the parameters for the sweep
    spatial_modes = [UNIFORM, POINT_SOURCES]
    uniform_doses = [0.5, 1.0, 1.5, 2.0]        # Low doses for Uniform
    point_doses = [10.0, 25.0, 50.0, 100.0]     # High doses for Point Sources
    
    # 0.0 means concentration will never be < threshold, so resuscitation is OFF
    # 1.0 means cells resuscitate once AmB drops below 1.0 µg/ml
    resuscitation_thresholds = [0.0, 1.0] 
    
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # 2. Prepare DataFrame
    results_df = DataFrame(
        Spatial_Mode = String[],
        Dose = Float64[], 
        Resuscitation_State = String[], 
        Time_Hours = Float64[],
        Genotype = String[],
        Alive = Int[], 
        Dead_Apop = Int[],
        Dead_Necro = Int[],
        Doubling_Time_Hrs = Float64[]
    )
    
    # Total combinations: 4 severity levels x 2 spatial modes x 2 resuscitation states = 16 runs
    total_runs = length(uniform_doses) * length(spatial_modes) * length(resuscitation_thresholds)
    current_run = 1
    plot_grid = []

    # 3. Execute Sweep
    for i in 1:length(uniform_doses)
        for mode in spatial_modes
            dose = mode == UNIFORM ? uniform_doses[i] : point_doses[i]
            for res_thresh in resuscitation_thresholds
                state_label = res_thresh == 0.0 ? "OFF" : "ON"
                
                @printf("Running %d/%d (Mode: %s, Dose: %.1f, Resuscitation: %s)...\n", current_run, total_runs, string(mode), dose, state_label)
                
                # PCD+ Colony (Capable of resuscitation)
                alive_plus, apop_plus, necro_plus = run_history_simulation(
                    PCDPlusCell, 
                    spatial_mode = mode, 
                    source_dose = dose, 
                    resuscitation_thresh = res_thresh,
                    ANTIFUNGAL_layer = 1.5
                )
                
                # PCD- Colony (Baseline: Cannot undergo apoptosis, so resuscitation parameter does nothing)
                alive_minus, apop_minus, necro_minus = run_history_simulation(
                    PCDMinusCell, 
                    spatial_mode = mode, 
                    source_dose = dose,
                    resuscitation_thresh = res_thresh,
                    ANTIFUNGAL_layer = 1.5
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
                    push!(results_df, (string(mode), dose, state_label, time_axis[j], "PCD+", alive_plus[j], apop_plus[j], necro_plus[j], td_plus))
                    push!(results_df, (string(mode), dose, state_label, time_axis[j], "PCD-", alive_minus[j], apop_minus[j], necro_minus[j], td_minus))
                end
                
                # Generate subplot
                p = plot(title="$mode($(dose)) | Res: $state_label", titlefontsize=7, legend=false, grid=false, xaxis=false, yaxis=false)
                if mode == spatial_modes[1] && res_thresh == resuscitation_thresholds[1]; yaxis!(p, true); ylabel!(p, "Cells"); end
                if i == length(uniform_doses); xaxis!(p, true); xlabel!(p, "Hours"); end
                if current_run == 1; plot!(p, legend=:topleft, legendfontsize=5); end
                
                plot!(p, time_axis, alive_plus, color=:blue, linewidth=2, label="PCD+ Alive")
                plot!(p, time_axis, alive_minus, color=:red, linewidth=2, label="PCD- Alive")
                
                # Add vertical line to show injection timing
                vline!(p, [t_phase_hours], color=:gray, linestyle=:dash, alpha=0.5, label="")
                
                push!(plot_grid, p)
                current_run += 1
            end
        end
    end
    
    # 4. Save Data and Plots
    data_dir = joinpath(@__DIR__, "..", "Data")
    mkpath(data_dir) # Automatically creates the "Data" folder if it doesn't exist yet!
    
    csv_path = joinpath(data_dir, "2026-03-25_Exp6_Resuscitation_Spatial_TimeSeries.csv") 
    CSV.write(csv_path, results_df)
    println("\nData successfully saved to: ", csv_path)
    
    println("Generating 4x4 Grid Plot...")
    final_plot = plot(plot_grid..., layout=(length(uniform_doses), length(spatial_modes) * length(resuscitation_thresholds)), size=(1200, 1000), 
                      plot_title="Survival Dynamics: Resuscitation (OFF vs ON) across Spatial Modes")
    
    fig_dir = joinpath(@__DIR__, "..", "Figures")
    mkpath(fig_dir) # Automatically creates the "Figures" folder if it doesn't exist yet!
    
    plot_path = joinpath(fig_dir, "2026-03-25_Exp6_Resuscitation_Spatial_Grid.png")
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: ", plot_path)
    
    println("\n=== Experiment 6 Complete! ===")
end

run_experiment_6()