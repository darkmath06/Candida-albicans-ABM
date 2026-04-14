using Plots
include("core_model_genetic.jl")

function visualize_evolutionary_trajectory()
    println("Running single parameter simulation to visualize evolutionary trajectory...")
    
    # We choose parameters that apply strong selection pressure 
    # (e.g. source_dose=0.8 with a uniform spatial mode to repeatedly challenge the population)
    passages_num = 10 
    
    alive, apop, necro, mean_susc, history = run_headless_simulation(
        spatial_mode = UNIFORM,
        source_dose = 0.8,
        mutation_rate = 0.05,
        passages = passages_num,
        passage_fraction = 0.1
    )
    
    # The history array contains the mean susceptibility at the end of each passage.
    # If the population goes extinct early, length(history) will be less than passages_num.
    passage_indices = 1:length(history)
    
    println("Trajectory captured across ", length(history), " passages.")
    println("Values: ", history)
    
    p = plot(
        passage_indices, history,
        title = "Evolutionary Trajectory of Apoptotic Trait",
        xlabel = "Passage Number",
        ylabel = "Mean Apoptosis Susceptibility",
        label = "Mean Population Trait",
        marker = :circle,
        linewidth = 2,
        ylims = (0, 1.0),
        color = :indigo
    )
    
    # Optionally overlay a horizontal line representing the starting average (expected ~0.5)
    hline!(p, [0.5], label="Initial Expectation", linestyle=:dash, color=:gray)
    
    plot_path = "../Figures/2026-04-09_Evolutionary_Trajectory.png"
    # Make sure the directory exists
    mkpath("../Figures")
    savefig(p, plot_path)
    
    println("\nSaved trajectory visualization to: ", plot_path)
end

visualize_evolutionary_trajectory()
