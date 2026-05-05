# ==========================================
# EXPERIMENT 7: Stochastic Survival Analysis (Monte Carlo)
# ==========================================
# Goal: Run Monte Carlo simulations sweeping through the 
# PCD- Death Rate Modifier to find the exact "Cost of Altruism" 
# tipping point where Kin Shielding beats the Selfish Advantage.

using DataFrames
using CSV
using Printf
using Plots
using StatsPlots # Required for groupedbar
using Random
using Statistics

# Load the core engine
include("core_model.jl")

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
    println("--- Starting Experiment 7: Monte Carlo Modifier Sweep ---")
    
    # 1. Setup Parameters
    NUM_RUNS = 50 # Adjust to 10 for a quick test if it takes too long
    
    # We use Point Sources because spatial gradients are where the Sponge Effect shines
    test_mode = UNIFORM
    test_dose = 1.5 
    

    death_modifiers = [0.1, 0.2, 0.3, 1]
    
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # Arrays to store survival rates for the final bar chart
    pcd_plus_rates = Float64[]
    pcd_minus_rates = Float64[]
    plot_grid = []
    
    total_sims = length(death_modifiers) * 2 * NUM_RUNS
    println("Total individual simulations to run: $total_sims")
    
    # 2. Execute the Sweep
    for mod in death_modifiers
        println("\n==================================")
        @printf("Testing PCD- Death Modifier: %.1f\n", mod)
        println("==================================")
        
        all_plus_trajectories = []
        all_minus_trajectories = []
        
        for run in 1:NUM_RUNS
            if run % 20 == 0 || run == 1
                @printf("  -> Executing Run %d / %d...\n", run, NUM_RUNS)
            end
            
            # PCD+ Colony (PCD+ ignores the modifier, but we pass it anyway for consistency)
            alive_plus = run_fast_history(PCDPlusCell, spatial_mode=test_mode, source_dose=test_dose, pcd_minus_death_modifier=mod)
            push!(all_plus_trajectories, alive_plus)
            
            # PCD- Colony (This is where the modifier actually applies)
            alive_minus = run_fast_history(PCDMinusCell, spatial_mode=test_mode, source_dose=test_dose, pcd_minus_death_modifier=mod)
            push!(all_minus_trajectories, alive_minus)
        end
        
        # 3. Calculate Statistics for this Modifier
        # STRICT SURVIVAL RULE: Population must be > 0 AND must be stable/growing over the last 10 hours (30 steps).
        # If the population is lower than it was 10 hours ago, it is crashing and counts as a failure.
        survived_plus = sum((traj[end] > 0 && traj[end] >= traj[max(1, end-30)]) for traj in all_plus_trajectories)
        survived_minus = sum((traj[end] > 0 && traj[end] >= traj[max(1, end-30)]) for traj in all_minus_trajectories)
        
        rate_plus = (survived_plus / NUM_RUNS) * 100
        rate_minus = (survived_minus / NUM_RUNS) * 100
        
        push!(pcd_plus_rates, rate_plus)
        push!(pcd_minus_rates, rate_minus)
        
        @printf("\nResult at Mod %.1f:\n", mod)
        @printf("  PCD+ Survival: %.1f%%\n", rate_plus)
        @printf("  PCD- Survival: %.1f%%\n", rate_minus)
        
        # 4. Generate Spaghetti Plot for this Modifier
        avg_plus = mean(all_plus_trajectories)
        avg_minus = mean(all_minus_trajectories)
        
        p_traj = plot(title="PCD- Death Rate: $(mod)x", titlefontsize=10, 
                      xlabel=(mod >= 0.8 ? "Time (Hours)" : ""), 
                      ylabel=(mod == 0.2 || mod == 0.8 ? "Alive Cells" : ""), 
                      grid=true, legend=false)
                      
        # Add legend only to the first plot
        if mod == death_modifiers[1]
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

    # 5. Compile and Save Plots
    println("\nGenerating Final Plots...")
    
    # Save the 2x2 Spaghetti Grid
    spaghetti_plot = plot(plot_grid..., layout=(2, 2), size=(1000, 800), 
                          plot_title="Stochastic Trajectories vs PCD- Penalty ($NUM_RUNS Runs/panel)")
    spag_path = "Project/Figures/2026-04-07_Exp7_Spaghetti_Grid.png"
    savefig(spaghetti_plot, spag_path)
    println("Spaghetti Grid saved to: ", spag_path)
    
    # Generate and Save the Grouped Bar Chart
    dose_labels = string.(death_modifiers)
    p_bar = groupedbar([pcd_plus_rates pcd_minus_rates], 
                labels=["PCD+" "PCD-"],
                color=[:blue :red],
                xticks=(1:length(death_modifiers), dose_labels),
                xlabel="PCD- Death Rate Modifier (Selfish Advantage)",
                ylabel="Survival Rate (%)",
                title="The Cost of Altruism: Survival vs Individual Penalty",
                legend=:topleft, ylims=(0, 105), size=(700, 500))
                
    bar_path = "Project/Figures/2026-04-07_Exp7_Survival_Bar.png"
    savefig(p_bar, bar_path)
    println("Survival Bar Chart saved to: ", bar_path)
    
    println("\n=== Experiment 7 Complete! ===")
end


run_experiment_7()