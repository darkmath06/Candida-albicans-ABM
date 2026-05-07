# ==========================================
# Candida albicans Evolutionary Parameter Sweep
# ==========================================
# This script imports your ABM and runs experiments
# across a range of parameters to find evolutionary tipping points.

using DataFrames
using CSV
using Plots
using Statistics

# Import the model (pointing to where your CandidaCell logic currently is)
include(joinpath(@__DIR__, "core_model_genetic.jl"))

function run_evolutionary_sweep(;
    # Define the ranges for the parameters you want to test
    dose_range = [0.5,1.0,1.5],
    mutation_rates = [0.05],
    replicates = 15, # Number of times to run each parameter combination to account for randomness
    passages = 5,
    passage_fraction = 0.1
)
    # Initialize an empty DataFrame to store our results
    results = DataFrame(
        AntifungalDose = Float64[],
        MutationRate = Float64[],
        Replicate = Int[],
        FinalAlive = Int[],
        FinalApop = Int[],
        FinalNecro = Int[],
        MeanSusceptibility = Float64[]
    )

    total_runs = length(dose_range) * length(mutation_rates) * replicates
    current_run = 0

    println("Starting Evolutionary Parameter Sweep...")
    println("Total simulation runs scheduled: $total_runs")

    # Loop through all parameter combinations
    for mut in mutation_rates
        for dose in dose_range
            for rep in 1:replicates
                current_run += 1
                println("Progress: Run $current_run / $total_runs | Dose: $dose, Mut: $mut, Rep: $rep")
                
                # Execute the headless simulation from core_model_genetic.jl
                # Passing UNIFORM mode and our swept dose parameter
                alive, apop, necro, mean_susc, _ = run_headless_simulation(
                    spatial_mode = UNIFORM,
                    source_dose = dose,
                    mutation_rate = mut,
                    passages = passages,
                    passage_fraction = passage_fraction
                )
                
                # Record the results
                push!(results, (dose, mut, rep, alive, apop, necro, mean_susc))
            end
        end
    end

    println("Sweep complete! Data collected.")
    return results
end

function run_evolutionary_experiment()
    # Run the sweep using the concentration doses requested
    results_df = run_evolutionary_sweep(
        dose_range = [0.5,1.0,1.5], 
        mutation_rates = [0.01,0.05, 0.1], 
        replicates = 15,
        passages = 5,
        passage_fraction = 0.1
    )

    # Aggregate the data (calculate mean and std across replicates)
    grouped_data = combine(groupby(results_df, [:AntifungalDose, :MutationRate]), 
                           :MeanSusceptibility => mean => :Avg_Evolved_Susceptibility,
                           :MeanSusceptibility => std => :Std_Evolved_Susceptibility,
                           :FinalAlive => mean => :Avg_Alive)

    # Handle any potential NaN values in standard deviation if a run fails or has 1 replicate
    grouped_data.Std_Evolved_Susceptibility = coalesce.(grouped_data.Std_Evolved_Susceptibility, 0.0)

    # Create a plot showing the evolutionary trait divergence
    p1 = plot(
        title = "Evolution of Apoptosis vs. Antifungal Concentration\n(After 5 Passages)",
        xlabel = "Uniform Antifungal Concentration (µg/ml)",
        ylabel = "Mean Apoptosis Susceptibility (Trait Value)",
        legend = :outertopright,
        ylims = (0, 1.0) # Trait goes from 0 to 1
    )

    for mut in unique(grouped_data.MutationRate)
        subset_data = filter(row -> row.MutationRate == mut, grouped_data)
        # Added yerror to include the standard deviation whiskers
        plot!(p1, subset_data.AntifungalDose, subset_data.Avg_Evolved_Susceptibility, 
              yerror = subset_data.Std_Evolved_Susceptibility,
              label = "Mut Rate: $mut", marker = :circle, linewidth=2)
    end

    # 4. Save Data and Plots
    data_dir = joinpath(@__DIR__, "..", "Data")
    mkpath(data_dir) # Automatically creates the "Data" folder if it doesn't exist yet!

    csv_path = joinpath(data_dir, "2026-04-21_Evolutionary_Dose_Sweep_Results.csv") 
    CSV.write(csv_path, results_df)
    println("\nData successfully saved to: ", csv_path)
    
    println("Generating Plot...")
    final_plot = p1 
    
    fig_dir = joinpath(@__DIR__, "..", "Figures")
    mkpath(fig_dir) # Automatically creates the "Figures" folder if it doesn't exist yet!
    
    plot_path = joinpath(fig_dir, "2026-04-21_Evolutionary_Dose_Sweep_Plot.png")
    savefig(final_plot, plot_path)
    println("Plot successfully saved to: ", plot_path)
    
    println("\n=== Evolutionary Sweep Complete! ===")
end

run_evolutionary_experiment()