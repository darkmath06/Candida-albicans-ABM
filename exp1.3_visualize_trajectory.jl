using Plots 
include(joinpath(@__DIR__, "core_model_genetic.jl"))


function visualize_evolutionary_trajectory()
    println("Running 15 replicate simulations to visualize evolutionary variance...")
    
    passages_num = 5 
    num_replicates = 5
    
    # 1. Initialize the empty plot with all our formatting
    p = plot(
        title = "Evolutionary Trajectory of Apoptotic Trait (15 Replicates)",
        xlabel = "Passage Number",
        ylabel = "Mean Apoptosis Probability",
        ylims = (0, 1.0),
        xticks = 0:passages_num, 
        legend = :topleft
    )
    
    # Optional: Draw the starting expectation line first so it sits behind the data
    hline!(p, [0.5], label="Initial (0.5)", linestyle=:dash, color=:gray, linewidth=2)
    
    # 2. Loop through the experiment 15 times
    for i in 1:num_replicates
        alive, apop, necro, mean_susc, history = run_headless_simulation(
            spatial_mode = UNIFORM,
            source_dose = 0.75,
            mutation_rate = 0.05, # Try changing this to 0.01 later to see the chaos!
            passages = passages_num,
            passage_fraction = 0.1
        )
        
        # Prepend the starting state
        full_history = vcat(0.5, history)
        passage_indices = 0:(length(full_history) - 1)
        
        # 3. Add this specific run's line to the plot
        # We set label=false so we don't get 15 identical legend entries
        # We use alpha=0.4 to make the lines slightly transparent
        plot!(p, passage_indices, full_history, 
              label=false, 
              color=:indigo, 
              alpha=0.4, 
              linewidth=2,
              marker=:circle)
    end
    
    fig_dir = joinpath(@__DIR__, "..", "Figures")
    mkpath(fig_dir) # Automatically creates the "Figures" folder if it doesn't exist yet!
    
    plot_path = joinpath(fig_dir, "2026-04-09_Evolutionary_Trajectory_15x.png")
    savefig(p, plot_path)
    
    println("\nSaved trajectory visualization to: ", plot_path)
end

visualize_evolutionary_trajectory()