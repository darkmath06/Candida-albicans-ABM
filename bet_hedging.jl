# ==========================================
# BET HEDGING EXPERIMENT: SINGLE SHOCK VS PULSATED
# ==========================================
# This script compares evolutionary outcomes in stable vs. fluctuating environments.
# It measures:
# 1. Mean Trait (Average strategy)
# 2. Inter-replicate Variance (Difference between experiment runs)
# 3. Intra-population Diversity (The spread of traits within a single dish)

using DataFrames
using CSV
using Plots
using Statistics
using Agents
using StatsBase

# Ensure the core model logic is loaded
include("core_model_genetic.jl")

# --- CUSTOM SIMULATION RUNNER ---
# This runner allows us to calculate Intra-population SD at the end of each run
function run_bet_hedging_simulation(; 
    exposure_mode = SINGLE_SHOCK, 
    dose = 1.5, 
    mut = 0.05, 
    passages = 5, 
    passage_fraction = 0.1
)
    cx, cy = (GRID_SIZE_PX + 1) / 2.0, (GRID_SIZE_PX + 1) / 2.0
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    
    current_traits = nothing
    final_alive = 0
    mean_trait = NaN 
    intra_std = NaN

    for passage in 1:passages
        num_starting = current_traits !== nothing ? length(current_traits) : min(INITIAL_CELLS, length(all_pos))
        starting_positions = all_pos[1:num_starting]
        
        # Initialize model with standard parameters
        model = initialize_model(
            starting_positions; 
            initial_traits = current_traits, 
            mutation_rate = mut, 
            source_dose = dose,
            spatial_mode = UNIFORM 
        )
        
        for step in 1:SIMULATION_STEPS
            # Manual Drug Injection Logic
            if exposure_mode == SINGLE_SHOCK
                if step == 73 
                    model.ANTIFUNGAL_layer .= dose
                end
            elseif exposure_mode == PULSATED
                # Pulse: 12-hour drug on, 36-hour drug off (48hr cycle)
                if (step % 144) < 36
                    model.ANTIFUNGAL_layer .= max.(model.ANTIFUNGAL_layer, dose)
                end
            end
            
            Agents.step!(model, 1)
        end
        
        # End of passage analysis
        alive_cells = filter(a -> a.alive, collect(allagents(model)))
        final_alive = length(alive_cells)
        
        if final_alive > 0
            traits = [a.apoptosis_susceptibility for a in alive_cells]
            mean_trait = mean(traits)
            intra_std = length(traits) > 1 ? std(traits) : 0.0
        else
            mean_trait = NaN
            intra_std = NaN
        end
        
        # Passaging
        if passage < passages && final_alive > 0
            num_to_sample = max(1, round(Int, final_alive * passage_fraction))
            sampled_cells = sample(alive_cells, num_to_sample, replace=false)
            current_traits = [a.apoptosis_susceptibility for a in sampled_cells]
        elseif final_alive == 0
            break
        end
    end
    
    return final_alive, mean_trait, intra_std
end

function run_bet_hedging_comparison(;
    dose_range = [0.5, 1.0, 1.5],
    mutation_rates = [0.05],
    replicates = 10,
    passages = 5,
    passage_fraction = 0.1
)
    # Results now include IntraPopStd
    results = DataFrame(
        ExposureMode = String[],
        AntifungalDose = Float64[],
        MutationRate = Float64[],
        Replicate = Int[],
        FinalAlive = Int[],
        MeanSusceptibility = Float64[],
        IntraPopStd = Float64[]
    )

    modes_to_test = [SINGLE_SHOCK, PULSATED]
    total_runs = length(modes_to_test) * length(dose_range) * length(mutation_rates) * replicates
    current_run = 0

    println("Starting Bet Hedging Analysis...")
    
    for mode in modes_to_test
        mode_str = string(mode)
        for mut in mutation_rates
            for dose in dose_range
                for rep in 1:replicates
                    current_run += 1
                    println("Progress: [$current_run/$total_runs] | Mode: $mode_str, Dose: $dose, Rep: $rep")
                    
                    alive, mean_t, i_std = run_bet_hedging_simulation(
                        exposure_mode = mode, 
                        dose = dose,
                        mut = mut,
                        passages = passages,
                        passage_fraction = passage_fraction
                    )
                    
                    push!(results, (mode_str, dose, mut, rep, alive, mean_t, i_std))
                end
            end
        end
    end

    mkpath("Project/Data")
    CSV.write("Project/Data/Bet_Hedging_Results_Detailed.csv", results)

    # --- AGGREGATE & PLOT ---
    df_clean = filter(row -> !isnan(row.MeanSusceptibility), results)
    
    # Calculate Summary Stats
    summary = combine(groupby(df_clean, [:ExposureMode, :AntifungalDose]), 
                      :MeanSusceptibility => mean => :AvgTrait,
                      :MeanSusceptibility => std => :InterStd,       # INTER: Variation between runs
                      :IntraPopStd => mean => :AvgIntraStd)         # INTRA: Average diversity in a dish

    summary.InterStd = coalesce.(summary.InterStd, 0.0)

    # Panel 1: Mean Trait
    p1 = plot(title="1. Population Mean Trait", ylabel="Mean Trait", ylims=(0, 1), legend=:topright)

    # Panel 2: Inter-replicate Variance (How much replicates differ)
    p2 = plot(title="2. Inter-replicate Variance", ylabel="StdDev of Means", legend=false)

    # Panel 3: Intra-population Diversity (How spread out one dish is)
    p3 = plot(title="3. Intra-population Diversity", ylabel="Avg Internal StdDev", xlabel="Antifungal Dose", legend=false)

    for mode in unique(summary.ExposureMode)
        sub = filter(r -> r.ExposureMode == mode, summary)
        
        plot!(p1, sub.AntifungalDose, sub.AvgTrait, yerror=sub.InterStd, label=mode, marker=:circle)
        plot!(p2, sub.AntifungalDose, sub.InterStd, label=mode, marker=:square, linestyle=:dash)
        plot!(p3, sub.AntifungalDose, sub.AvgIntraStd, label=mode, marker=:diamond, linestyle=:dot)
    end

    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 1000), margin=5Plots.mm)
    
    mkpath("Project/Figures")
    savefig(final_plot, "Project/Figures/Bet_Hedging_Detailed_Analysis.png")
    println("Analysis complete. Plot saved to Project/Figures/Bet_Hedging_Detailed_Analysis.png")
    
    return results
end

run_bet_hedging_comparison()