# ==========================================
# POST-SIMULATION ANALYSIS: BOXPLOTS
# ==========================================
# This is a standalone script to generate boxplots directly 
# from the saved CSV data without running the core model.

using CSV
using DataFrames
using StatsPlots # Required for the @df macro and grouped boxplots

function generate_evolutionary_boxplots()
    # The file path to the CSV you want to analyze
    csv_path = "Project/Data/2026-04-20_Evolutionary_Dose_Sweep_Results.csv" 
    
    println("Loading data from: ", csv_path)
    
    # Safety check
    if !isfile(csv_path)
        println("Error: File not found at $csv_path.")
        println("Make sure the CSV is in the exact same folder as this script, or update the path!")
        return
    end

    # Read the CSV into a DataFrame
    df = CSV.read(csv_path, DataFrame)
    
    # 1. Clean the Data
    # Filter out any NaN or Missing values (this happens when a population goes completely extinct)
    df_valid = filter(row -> !ismissing(row.MeanSusceptibility) && !isnan(row.MeanSusceptibility), df)
    
    # 2. Extract X-Axis Ticks
    # Automatically find which diffusion rates were tested to make the x-axis look clean
    tested_rates = sort(unique(df_valid.AntifungalDose))
    
    println("Data cleaned. Generating Boxplot...")
    
    # 3. Create the Grouped Boxplot
    final_boxplot = @df df_valid groupedboxplot(
        :AntifungalDose, 
        :MeanSusceptibility, 
        group = :MutationRate,
        title = "Evolution of Apoptosis vs. Antifungal Dose\n(Boxplot Distribution)",
        xlabel = "Antifungal Dose",
        ylabel = "Mean Apoptosis Probability (Trait Value)",
        legend = :outertopright,
        ylims = (0.0, 1.0),
        xticks = tested_rates, 
        linewidth = 1.5,
        outliers = true,     # Show outlier dots
        framestyle = :box
    )
    
    # 4. Save the Output
    plot_path = "2026-04-14_Evolutionary_Dose_Sweep_Boxplot.png"
    savefig(final_boxplot, plot_path)
    println("Boxplot successfully saved to: ", plot_path)
    
    # Display the plot in your viewer/REPL
    display(final_boxplot)
end

# Run the function
generate_evolutionary_boxplots()