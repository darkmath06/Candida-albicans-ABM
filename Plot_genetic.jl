# ==========================================
# POST-SIMULATION ANALYSIS: BOXPLOTS
# ==========================================
# Standalone script to generate boxplots from CSV data.
# Optimized to handle "Extinction" events (NaNs) without crashing.

using CSV
using DataFrames
using StatsPlots
using Statistics

function generate_evolutionary_boxplots()
    csv_path = "Project/Data/2026-04-21_Evolutionary_Dose_Sweep_Results.csv" 
    
    if !isfile(csv_path)
        println("Error: File not found at $csv_path.")
        return
    end

    # 1. Load data
    df = CSV.read(csv_path, DataFrame)
    
    # 2. Force-Clean the Data
    # We create a categorical label for the X-axis to prevent scaling issues
    # and handle cases where a group is 100% NaNs.
    df_clean = DataFrame(DoseLabel=String[], MutationRate=Float64[], Trait=Float64[])
    
    for sdf in groupby(df, [:AntifungalDose, :MutationRate])
        dose_val = sdf.AntifungalDose[1]
        mut_val = sdf.MutationRate[1]
        
        # Get all non-missing, non-NaN values
        raw_vals = sdf.MeanSusceptibility
        valid_vals = filter(x -> !ismissing(x) && !isnan(x), raw_vals)
        
        if isempty(valid_vals)
            # FORCE PLOT: If a group is entirely extinct, we add one dummy value 
            # outside the plot range so the math engine sees "data" but the box is invisible.
            push!(df_clean, (DoseLabel=string(dose_val), MutationRate=mut_val, Trait=-0.05))
        else
            for v in valid_vals
                push!(df_clean, (DoseLabel=string(dose_val), MutationRate=mut_val, Trait=v))
            end
        end
    end
    
    println("Plotting $(nrow(df_clean)) rows of processed data...")
    
    # 3. Generate Plot
    # We use 'boxplot' with 'group' as it's more stable for single-categorical axes
    p = @df df_clean boxplot(
        :DoseLabel, 
        :Trait, 
        group = :MutationRate,
        title = "Evolution of Apoptosis vs. Antifungal Dose\n(Distribution Analysis)",
        xlabel = "Antifungal Dose (µg/ml)",
        ylabel = "Mean Apoptosis Probability (Trait)",
        legend = :outertopright,
        ylims = (0.0, 1.0), # This hides our -0.05 placeholders
        linewidth = 1.2,
        fillalpha = 0.5,
        outliers = true,
        marker = (3, :circle, 0.4), # Jittered dots to see individual replicates
        framestyle = :box
    )
    
    # 4. Save and Show
    plot_path = "Project/Figures/Evolutionary_Boxplot_Final.png"
    mkpath("Project/Figures")
    savefig(p, plot_path)
    println("Boxplot saved to: ", plot_path)
    display(p)
end

generate_evolutionary_boxplots()