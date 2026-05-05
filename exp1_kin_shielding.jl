# ==========================================
# EXPERIMENT 1: The "Kin Shielding" Sweep
# ==========================================
# Goal: Test if the PCD+ "Sponge Effect" works against 
# uniform blanket exposures vs localized point-source doses.

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
    
    # Initialize history arrays
    h_alive = Int[]
    h_apop = Int[]
    h_necro = Int[]
    
    # Record Step 0
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
        
        # Record data for this step
        push!(h_alive, count(a -> a.alive, allagents(model)))
        push!(h_apop, count(a -> is_dead_apoptosis(a), allagents(model)))
        push!(h_necro, count(a -> a.dead_necrosis, allagents(model)))
    end
    
    return h_alive, h_apop, h_necro
end

function run_experiment_1()
    println("--- Starting Experiment 1: Kin Shielding (Uniform Dose Sweep) ---")
    
    # Test 4 uniform doses to keep a clean 4x4 plot grid
    uniform_doses = [0.5, 1.0, 1.5, 2.0]
    apop_capacities = [0.5, 1.5, 2.5, 4.0] 
    
    # Time axis for plotting (converting steps to hours)
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # 2. Prepare a "Long Format" DataFrame
    results_df = DataFrame(
        Spatial_Mode = String[],
        Dose = Float64[], 
        Apop_Capacity = Float64[], 
        Time_Hours = Float64[],
        Genotype = String[],
        Alive = Int[], 
        Dead_Apop = Int[],
        Dead_Necro = Int[],
        Doubling_Time_Hrs = Float64[]
    )
    
    total_runs = length(uniform_doses) * length(apop_capacities)
    current_run = 1
    
    # Array to hold our 16 subplots
    plot_grid = []
    
    # Array to store results temporarily so we can calculate max y-axis
    simulation_results = []

    # 3. Execute the Sweep Loop
    for dose in uniform_doses
        for cap in apop_capacities
            @printf("Running %d/%d (Mode: UNIFORM, Dose: %.1f, Capacity: %.1f)...\n", current_run, total_runs, dose, cap)
            
            # Run PCD+ colony
            alive_plus, apop_plus, necro_plus = run_history_simulation(
                PCDPlusCell, spatial_mode = UNIFORM, source_dose = dose, max_binding_apop = cap
            )
            
            # Run PCD- colony
            alive_minus, apop_minus, necro_minus = run_history_simulation(
                PCDMinusCell, spatial_mode = UNIFORM, source_dose = dose, max_binding_apop = cap
            )
            
            # --- Calculate Doubling Times (pre-injection) ---
            injection_step = 73 # UNIFORM is step 73
            t_phase_hours = injection_step * TIME_STEP_DT
            
            n0_plus = alive_plus[1]
            nt_plus = alive_plus[injection_step + 1] # +1 because array index 1 is Step 0
            td_plus = (nt_plus > n0_plus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_plus / n0_plus)) : NaN
            
            n0_minus = alive_minus[1]
            nt_minus = alive_minus[injection_step + 1]
            td_minus = (nt_minus > n0_minus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_minus / n0_minus)) : NaN
            
            # Append history to DataFrame
            for i in 1:length(time_axis)
                push!(results_df, ("UNIFORM", dose, cap, time_axis[i], "PCD+", alive_plus[i], apop_plus[i], necro_plus[i], td_plus))
                push!(results_df, ("UNIFORM", dose, cap, time_axis[i], "PCD-", alive_minus[i], apop_minus[i], necro_minus[i], td_minus))
            end
            
            # Cache results for plotting later
            push!(simulation_results, (dose, cap, alive_plus, alive_minus, t_phase_hours))
            
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
    for dose in uniform_doses
        for cap in apop_capacities
            res = simulation_results[current_plot]
            _, _, alive_plus, alive_minus, t_phase_hours = res
            
            # Generate the subplot for this specific combination
            p = plot(title="Dose: $dose | Cap: $cap", titlefontsize=8, legend=false, grid=false, xaxis=false, yaxis=false)
            
            # Add axes/labels to the edges
            if cap == apop_capacities[1]; yaxis!(p, true); ylabel!(p, "Cells"); end
            if dose == uniform_doses[end]; xaxis!(p, true); xlabel!(p, "Hours"); end
            if current_plot == 1; plot!(p, legend=:topleft, legendfontsize=5); end
            
            # Apply the GLOBAL y-axis limit
            plot!(p, ylims=(0, global_max_y))
            
            plot!(p, time_axis, alive_plus, color=:blue, linewidth=2, label="PCD+ Alive")
            plot!(p, time_axis, alive_minus, color=:red, linewidth=2, label="PCD- Alive")
            
            # Add vertical line to show exactly when the drug hits
            vline!(p, [t_phase_hours], color=:gray, linestyle=:dash, alpha=0.5, label="")
            
            push!(plot_grid, p)
            current_plot += 1
        end
    end
    
    # 4. Save the Data
    csv_path = "C:/Users/Mathi/Downloads/UVA Ams/Project/Data/2026-03-27_Exp1_KinShielding_UniformDose_TimeSeries.csv" 
    CSV.write(csv_path, results_df)
    println("\nData successfully saved to: ", csv_path)
    
    # 5. Compile and Save the Plot
    println("Generating 4x4 Grid Plot...")
    final_plot = plot(plot_grid..., layout=(length(uniform_doses), length(apop_capacities)), size=(1200, 1000), 
                      plot_title="Sponge Effect: Survival vs Uniform Dose & Apop Binding Capacity")
    
    plot_path = "C:/Users/Mathi/Downloads/UVA Ams/Project/Figures/2026-03-27_Exp1_KinShielding_UniformDose_Grid.png"
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: ", plot_path)
    
    println("\n=== Experiment 1 Complete! ===")
end

run_experiment_1()