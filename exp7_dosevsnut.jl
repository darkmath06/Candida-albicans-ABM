# =========================================================
# EXPERIMENT 3: "Uniform AF Dose vs Initial Nutrient Sweep"
# =========================================================
# Goal: Test only the UNIFORM spatial mode, sweeping across 
# different antifungal doses and different starting nutrient levels.

using DataFrames
using CSV
using Printf
using Plots 

# Load the core engine using the absolute path based on your terminal location
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
    
    # Injection step for UNIFORM is 24h = step 72
    injection_step = 72 
    
    for step in 1:SIMULATION_STEPS
        if step == injection_step
            # We are only testing UNIFORM mode in this script
            model.ANTIFUNGAL_layer .= model.source_dose
        end
        
        Agents.step!(model, 1)
        
        push!(h_alive, count(a -> a.alive, allagents(model)))
        push!(h_apop, count(a -> is_dead_apoptosis(a), allagents(model)))
        push!(h_necro, count(a -> a.dead_necrosis, allagents(model)))
    end
    
    return h_alive, h_apop, h_necro
end

function run_experiment_3()
    println("--- Starting Experiment 3: Uniform Dose vs Nutrient Sweep ---")
    
    # 1. Define the parameters for the sweep
    uniform_doses = [0.5, 1.0, 1.5, 2.0]        # Dosages for UNIFORM mode
    nutrient_levels = [6.0, 10.0, 14.0, 18.0]   # Varying starting nutrient capacities
    fixed_res_frac = 0.1                        # Hold reservoir capacity constant
    max_binding_apop = 0.42
    time_axis = (0:SIMULATION_STEPS) .* TIME_STEP_DT
    
    # 2. Prepare DataFrame
    results_df = DataFrame(
        Dose = Float64[], 
        Nutrient_Level = Float64[],
        Time_Hours = Float64[],
        Genotype = String[],
        Alive = Int[], 
        Dead_Apop = Int[],
        Dead_Necro = Int[],
        Doubling_Time_Hrs = Float64[]
    )
    
    total_runs = length(uniform_doses) * length(nutrient_levels)
    current_run = 1
    plot_grid = []
    
    # Array to store results temporarily so we can calculate max y-axis per row
    simulation_results = []

    # 3. Execute Sweep
    for dose in uniform_doses
        for nutrient in nutrient_levels
            
            @printf("Running %d/%d (Dose: %.1f, Nutrient: %.1f)...\n", current_run, total_runs, dose, nutrient)
            
            # PCD+ Colony
            alive_plus, apop_plus, necro_plus = run_history_simulation(
                PCDPlusCell, 
                spatial_mode = UNIFORM, # Fixed to Uniform
                source_dose = dose, 
                init_nutrient = nutrient,
                reservoir_fraction = fixed_res_frac,
                max_binding_apop = 0.42
            )
            
            # PCD- Colony
            alive_minus, apop_minus, necro_minus = run_history_simulation(
                PCDMinusCell, 
                spatial_mode = UNIFORM, # Fixed to Uniform
                source_dose = dose,
                init_nutrient = nutrient,
                reservoir_fraction = fixed_res_frac
            )
            
            # Calculate Doubling Times before stress hits
            injection_step = 72
            t_phase_hours = injection_step * TIME_STEP_DT
            
            n0_plus = alive_plus[1]
            nt_plus = alive_plus[injection_step + 1] 
            td_plus = (nt_plus > n0_plus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_plus / n0_plus)) : NaN
            
            n0_minus = alive_minus[1]
            nt_minus = alive_minus[injection_step + 1]
            td_minus = (nt_minus > n0_minus && t_phase_hours > 0) ? (t_phase_hours * log(2) / log(nt_minus / n0_minus)) : NaN
            
            for j in 1:length(time_axis)
                push!(results_df, (dose, nutrient, time_axis[j], "PCD+", alive_plus[j], apop_plus[j], necro_plus[j], td_plus))
                push!(results_df, (dose, nutrient, time_axis[j], "PCD-", alive_minus[j], apop_minus[j], necro_minus[j], td_minus))
            end
            
            # Store data for plotting later
            push!(simulation_results, (dose, nutrient, alive_plus, alive_minus, t_phase_hours))
            
            current_run += 1
        end
    end
    
    # 3.5 Generate Subplots with row-linked y-axes
    current_plot = 1
    for dose in uniform_doses
        # Find maximum cell count across all nutrient levels for this specific dose
        row_max_y = 0
        for res in simulation_results
            if res[1] == dose
                row_max_y = max(row_max_y, maximum(res[3]), maximum(res[4]))
            end
        end
        
        # Add 5% padding to the top of the y-axis (and ensure it's at least 1 to avoid plot errors)
        row_max_y = max(row_max_y * 1.05, 1.0)
        
        for nutrient in nutrient_levels
            # Retrieve the simulation data for this exact configuration
            res = simulation_results[current_plot]
            _, _, alive_plus, alive_minus, t_phase_hours = res
            
            # Generate subplot
            p = plot(title="Dose: $dose | Nutrients: $nutrient", titlefontsize=8, legend=false, grid=false, xaxis=false, yaxis=false)
            
            # Add labels to axes on the left and bottom edges of the grid
            if nutrient == nutrient_levels[1]; yaxis!(p, true); ylabel!(p, "Cells"); end
            if dose == uniform_doses[end]; xaxis!(p, true); xlabel!(p, "Hours"); end
            
            # Add a legend only to the first plot
            if current_plot == 1; plot!(p, legend=:topleft, legendfontsize=6); end
            
            # Apply the row-specific y-axis limit
            plot!(p, ylims=(0, row_max_y))
            
            plot!(p, time_axis, alive_plus, color=:blue, linewidth=2, label="PCD+ Alive")
            plot!(p, time_axis, alive_minus, color=:red, linewidth=2, label="PCD- Alive")
            
            # Add vertical lines to show injection timing
            vline!(p, [t_phase_hours], color=:gray, linestyle=:dash, alpha=0.5, label="")
            
            push!(plot_grid, p)
            current_plot += 1
        end
    end
    
    # 4. Save Data and Plots
    data_dir = joinpath(@__DIR__, "..", "Data")
    mkpath(data_dir) # Automatically creates the "Data" folder if it doesn't exist yet!
    
    csv_path = joinpath(data_dir, "Exp3_UniformDoseNutrientSweep_TimeSeries.csv") 
    CSV.write(csv_path, results_df)
    println("\nData successfully saved to: ", csv_path)
    
    println("Generating Grid Plot...")
    # Layout size adjusts to the number of combinations
    final_plot = plot(plot_grid..., layout=(length(uniform_doses), length(nutrient_levels)), size=(1000, 800), 
                      plot_title="Survival vs Uniform Dose & Initial Nutrients")
    
    fig_dir = joinpath(@__DIR__, "..", "Figures")
    mkpath(fig_dir) # Automatically creates the "Figures" folder if it doesn't exist yet!
    
    plot_path = joinpath(fig_dir, "Exp3_UniformDoseNutrientSweep_Grid.png")
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: ", plot_path)
    
    println("\n=== Experiment 3 Complete! ===")
end

run_experiment_3()