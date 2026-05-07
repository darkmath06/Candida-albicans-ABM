# ==========================================
# EXPERIMENT 7: Stochastic Survival Analysis (Monte Carlo)
# ==========================================
# Goal: Run Monte Carlo simulations sweeping through different
# Antifungal Doses to find the probabilistic tipping point of survival.

using DataFrames
using CSV
using Printf
using Plots
using StatsPlots # Required for groupedbar
using Random
using Statistics

# Load the core engine robustly
include(joinpath(@__DIR__, "core_model.jl"))

# --- Fast History Tracker ---
# We only need the 'alive' count for this analysis to save memory over hundreds of runs
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
    println("--- Starting Experiment 7: Monte Carlo Dose Sweep ---")
    
    # 1. Setup Parameters
    NUM_RUNS = 10 # Number of stochastic runs per configuration
    test_doses = [0.8, 1.0, 1.2, 1.5] # The parameter we are sweeping
    
    test_mode = UNIFORM 
    
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # Arrays to store survival rates for the final bar chart
    pcd_plus_rates = Float64[]
    pcd_minus_rates = Float64[]
    plot_grid = []
    
    total_sims = length(test_doses) * 2 * NUM_RUNS
    println("Total individual simulations to run: $total_sims")
    
    # Cache to store results for the global Y-axis scaling later
    simulation_results = []

    # 2. Execute the Monte Carlo Sweep
    for dose in test_doses
        println("\n--- Testing Dose: $(dose) ---")
        all_plus_trajectories = []
        all_minus_trajectories = []
        
        for run in 1:NUM_RUNS
            if run % 5 == 0 || run == 1
                @printf("  -> Executing Run %d / %d...\n", run, NUM_RUNS)
            end
            
            # PCD+ Colony
            alive_plus = run_fast_history(PCDPlusCell, spatial_mode=test_mode, source_dose=dose)
            push!(all_plus_trajectories, alive_plus)
            
            # PCD- Colony 
            alive_minus = run_fast_history(PCDMinusCell, spatial_mode=test_mode, source_dose=dose)
            push!(all_minus_trajectories, alive_minus)
        end
        
        # Cache this data for plotting later
        push!(simulation_results, (dose, all_plus_trajectories, all_minus_trajectories))
        
        # 3. Calculate Statistics for this Dose
        # STRICT SURVIVAL RULE: Population must be > 0 AND must be stable/growing over the last 10 hours (30 steps).
        survived_plus = sum((traj[end] > 0 && traj[end] >= traj[max(1, end-30)]) for traj in all_plus_trajectories)
        survived_minus = sum((traj[end] > 0 && traj[end] >= traj[max(1, end-30)]) for traj in all_minus_trajectories)
        
        rate_plus = (survived_plus / NUM_RUNS) * 100
        rate_minus = (survived_minus / NUM_RUNS) * 100
        
        push!(pcd_plus_rates, rate_plus)
        push!(pcd_minus_rates, rate_minus)
        
        @printf("  Result Summary -> PCD+: %.1f%% | PCD-: %.1f%%\n", rate_plus, rate_minus)
    end
    
    # 3.5 Calculate Global Y-Axis Maximum for Spaghetti Plots
    global_max_y = 0.0
    for res in simulation_results
        _, plus_trajs, minus_trajs = res
        for traj in plus_trajs; global_max_y = max(global_max_y, maximum(traj)); end
        for traj in minus_trajs; global_max_y = max(global_max_y, maximum(traj)); end
    end
    global_max_y = max(global_max_y * 1.05, 1.0) # Add 5% padding
    
    # 4. Generate Spaghetti Plots
    for res in simulation_results
        dose, all_plus_trajectories, all_minus_trajectories = res
        
        avg_plus = mean(all_plus_trajectories)
        avg_minus = mean(all_minus_trajectories)
        
        p_traj = plot(title="Antifungal Dose: $(dose)", titlefontsize=10, 
                        xlabel=(dose >= test_doses[end-1] ? "Time (Hours)" : ""), 
                        ylabel=(dose == test_doses[1] || dose == test_doses[3] ? "Alive Cells" : ""), 
                        grid=true, legend=false, ylims=(0, global_max_y))
                        
        # Add legend only to the first plot
        if dose == test_doses[1]
            plot!(p_traj, legend=:topright, legendfontsize=6)
        end
                        
        # Plot all faint lines
        for i in 1:NUM_RUNS
            plot!(p_traj, time_axis, all_plus_trajectories[i], color=:blue, alpha=0.1, linewidth=1, label="")
            plot!(p_traj, time_axis, all_minus_trajectories[i], color=:red, alpha=0.1, linewidth=1, label="")
        end
        
        # Plot the thick average lines on top
        plot!(p_traj, time_axis, avg_plus, color=:blue, linewidth=3, label="PCD+ Avg")
        plot!(p_traj, time_axis, avg_minus, color=:red, linewidth=3, label="PCD- Avg")
        
        push!(plot_grid, p_traj)
    end

    # 5. Compile and Save Plots robustly
    println("\nGenerating Final Plots...")
    
    # Ensure Figures directory exists
    fig_dir = joinpath(@__DIR__, "..", "Figures")
    mkpath(fig_dir)
    
    # Save the 2x2 Spaghetti Grid
    spaghetti_plot = plot(plot_grid..., layout=(2, 2), size=(1000, 800), 
                          plot_title="Stochastic Trajectories vs Antifungal Dose ($NUM_RUNS Runs/panel)")
    spag_path = joinpath(fig_dir, "Exp7_Spaghetti_Grid.png")
    savefig(spaghetti_plot, spag_path)
    println("Spaghetti Grid saved to: ", spag_path)
    
    # Generate and Save the Grouped Bar Chart
    dose_labels = string.(test_doses)
    p_bar = groupedbar([pcd_plus_rates pcd_minus_rates], 
                labels=["PCD+" "PCD-"],
                color=[:blue :red],
                xticks=(1:length(test_doses), dose_labels),
                xlabel="Antifungal Dose Severity",
                ylabel="Survival Rate (%)",
                title="Stochastic Survival Probability vs Dose",
                legend=:topright, ylims=(0, 105), size=(700, 500))
                
    bar_path = joinpath(fig_dir, "Exp7_Survival_Bar.png")
    savefig(p_bar, bar_path)
    println("Survival Bar Chart saved to: ", bar_path)
    
    println("\n=== Experiment 7 Complete! ===")
end

run_experiment_7()