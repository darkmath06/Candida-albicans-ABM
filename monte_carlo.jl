# ==========================================
# EXPERIMENT 7: Stochastic Survival Analysis (Monte Carlo)
# ==========================================
# Goal: Run 100 identical simulations to observe the variance 
# in survival outcomes due to the model's inherent stochasticity.

using DataFrames
using CSV
using Printf
using Plots
using Random
using Statistics

# Load the core engine
include("core_model.jl")

# --- Fast History Tracker ---
# We only need the 'alive' count for this analysis to save memory over 100 runs
function run_fast_history(AgentType::Type; kwargs...)
    cx, cy = (GRID_SIZE_PX + 1) / 2.0, (GRID_SIZE_PX + 1) / 2.0
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    starting_positions = all_pos[1:min(INITIAL_CELLS, length(all_pos))]
    
    model = initialize_model(AgentType, starting_positions; kwargs...)
    
    h_alive = Int[]
    push!(h_alive, count(a -> a.alive, allagents(model)))
    
    # Dynamic injection step
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
    end
    
    return h_alive
end

function run_experiment_7()
    println("--- Starting Experiment 7: Stochastic Monte Carlo Simulation ---")
    
    # 1. Setup Parameters
    NUM_RUNS = 10
    
    # Choose a "borderline" scenario where survival is ambiguous. 
    # You may need to tweak this dose to find the "sweet spot" of stochasticity.
    test_mode = UNIFORM
    test_dose = 2 
    
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # Arrays to hold all trajectories
    all_plus_trajectories = []
    all_minus_trajectories = []
    
    # 2. Execute 100 Runs
    for run in 1:NUM_RUNS
        @printf("Executing Run %d / %d...\n", run, NUM_RUNS)
        
        # We do NOT set a random seed here, so every run uses a different natural seed
        
        # PCD+ Colony
        alive_plus = run_fast_history(PCDPlusCell, spatial_mode=test_mode, source_dose=test_dose)
        push!(all_plus_trajectories, alive_plus)
        
        # PCD- Colony
        alive_minus = run_fast_history(PCDMinusCell, spatial_mode=test_mode, source_dose=test_dose)
        push!(all_minus_trajectories, alive_minus)
    end
    
    # 3. Calculate Survival Statistics
    # A colony "survived" if its final alive count at step 600 is > 0
    survived_plus = sum(traj[end] > 0 for traj in all_plus_trajectories)
    survived_minus = sum(traj[end] > 0 for traj in all_minus_trajectories)
    
    surv_rate_plus = (survived_plus / NUM_RUNS) * 100
    surv_rate_minus = (survived_minus / NUM_RUNS) * 100
    
    println("\n=== FINAL SURVIVAL RATES ===")
    @printf("PCD+ Survival: %.1f%% (%d/%d runs)\n", surv_rate_plus, survived_plus, NUM_RUNS)
    @printf("PCD- Survival: %.1f%% (%d/%d runs)\n", surv_rate_minus, survived_minus, NUM_RUNS)
    
    # Calculate averages
    avg_plus = mean(all_plus_trajectories)
    avg_minus = mean(all_minus_trajectories)

    # 4. Generate Plots
    println("\nGenerating Stochasticity Plots...")
    
    # Plot A: The Spaghetti Plot
    p_traj = plot(title="Stochastic Trajectories ($NUM_RUNS Runs)\nDose: $test_dose | Mode: $test_mode", 
                  xlabel="Time (Hours)", ylabel="Alive Cells", grid=true, legend=false)
                  
    # Plot all faint lines
    for i in 1:NUM_RUNS
        plot!(p_traj, time_axis, all_plus_trajectories[i], color=:blue, alpha=0.1, linewidth=1)
        plot!(p_traj, time_axis, all_minus_trajectories[i], color=:red, alpha=0.1, linewidth=1)
    end
    
    # Plot the thick average lines on top
    plot!(p_traj, time_axis, avg_plus, color=:blue, linewidth=3, label="PCD+ Average")
    plot!(p_traj, time_axis, avg_minus, color=:red, linewidth=3, label="PCD- Average")
    
    # Plot B: Bar Chart of Survival Rates
    p_bar = bar(["PCD+", "PCD-"], [surv_rate_plus, surv_rate_minus], 
                title="Colony Survival Rate", ylabel="Survival %",
                color=[:blue, :red], legend=false, ylims=(0, 105))
                
    # Combine into a final layout
    final_plot = plot(p_traj, p_bar, layout=(1, 2), size=(1000, 500), margin=5Plots.mm)
    
    plot_path = "Project/Figures/2026-03-31_Exp7_Stochastic_Survival.png"
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: ", plot_path)
    
    println("\n=== Experiment 7 Complete! ===")
end

run_experiment_7()