# ==========================================
# Dedicated Timelapse Generator
# ==========================================
# Run this script to generate side-by-side GIF timelapses
# of the petri dish and the evolving trait graph for specific parameters.

using Plots
using Statistics
using Agents
using StatsBase

# Include the core model mechanics (make sure they are in the same folder!)
include("core_model_genetic.jl")
theme(:default) # Or :default for light mode


# Helper function to map traits to the requested custom discrete colors
function get_trait_color(trait::Float64)
    if trait < 0.2
        return :red
    elseif trait < 0.4
        return :orange
    elseif trait < 0.6
        return :yellow
    elseif trait < 0.8
        return :green
    else
        return :blue
    end
end

# Generates a single timelapse with a side-by-side layout
function run_custom_timelapse(dose::Float64, mut::Float64; 
                              passages=5, passage_fraction=0.1, 
                              capture_interval=10, fps=10, run_id="Custom_Timelapse")
    
    # Calculate starting setup based on the core model constants
    cx, cy = (GRID_SIZE_PX + 1) / 2.0, (GRID_SIZE_PX + 1) / 2.0
    all_pos = [(x, y) for x in 1:GRID_SIZE_PX for y in 1:GRID_SIZE_PX]
    sort!(all_pos, by = pos -> (pos[1] - cx)^2 + (pos[2] - cy)^2)
    
    current_traits = nothing
    
    # Track the global step and mean trait over the entire experiment for the line graph
    global_hours = Float64[]
    mean_trait_history = Float64[]
    std_trait_history = Float64[] # Added to track standard deviation
    
    # Setup animation and folders
    anim = Animation()
    mkpath("Project/Figures/PetriDish")
    
    println("Starting timelapse generation for Dose: $dose, Mut: $mut...")
    
    for passage in 1:passages
        num_starting = current_traits !== nothing ? length(current_traits) : min(INITIAL_CELLS, length(all_pos))
        starting_positions = all_pos[1:num_starting]
        
        # Initialize the model using the specific parameters for this timelapse
        model = initialize_model(starting_positions; initial_traits=current_traits, spatial_mode=UNIFORM, source_dose=dose, mutation_rate=mut)
        injection_step = model.spatial_mode == UNIFORM ? 73 : 1
        
        for step in 1:SIMULATION_STEPS
            if step == injection_step
                if model.spatial_mode == UNIFORM
                    model.ANTIFUNGAL_layer .= model.source_dose
                end
            end
            
            Agents.step!(model, 1)

            # Capture frames based on interval
            if step % capture_interval == 0 || step == SIMULATION_STEPS
                alive_cells_now = filter(a -> a.alive, collect(allagents(model)))
                
                # Extract traits to calculate mean and std
                traits_now = [a.apoptosis_susceptibility for a in alive_cells_now]
                mean_trait_now = isempty(traits_now) ? NaN : mean(traits_now)
                # Calculate standard deviation (0.0 if only 1 cell, NaN if extinct)
                std_trait_now = length(traits_now) > 1 ? std(traits_now) : (isempty(traits_now) ? NaN : 0.0)

                # Store data for the line graph (converted to hours)
                current_global_step = (passage - 1) * SIMULATION_STEPS + step
                push!(global_hours, current_global_step * TIME_STEP_DT)
                push!(mean_trait_history, mean_trait_now)
                push!(std_trait_history, std_trait_now) # Store the standard deviation

                # --- LEFT PANEL: Petri Dish ---
                # Added Mean Trait to the title, disabled colorbar for heatmap, but enabled the legend
                current_hour = round(step * TIME_STEP_DT, digits=1)
                title_text = "Passage $passage, Hour $current_hour\nDose: $dose, Mut: $mut\nMean: $(isnan(mean_trait_now) ? "Extinct" : round(mean_trait_now, digits=3)) | Std: $(isnan(std_trait_now) ? "-" : round(std_trait_now, digits=3))"
                
                p_dish = heatmap(1:GRID_SIZE_PX, 1:GRID_SIZE_PX, model.ANTIFUNGAL_layer, 
                                 color=:Greys, colorbar=false, legend=:topright, 
                                 legendfontsize=7, aspect_ratio=1.0, showaxis=false,
                                 title=title_text, titlefontsize=10)
                
                # Define labels for the legend
                color_labels = Dict(:red => "< 0.2", :orange => "0.2 - 0.4", 
                                    :yellow => "0.4 - 0.6", :green => "0.6 - 0.8", :blue => "> 0.8")
                
                # Plot invisible NaNs first to keep the legend stable across all frames
                for c in [:red, :orange, :yellow, :green, :blue]
                    scatter!(p_dish, [NaN], [NaN], color=c, label=color_labels[c], 
                             marker=:circle, markersize=3, markerstrokewidth=0)
                end

                if !isempty(alive_cells_now)
                    color_groups = Dict(:red => (Int[], Int[]), :orange => (Int[], Int[]), 
                                        :yellow => (Int[], Int[]), :green => (Int[], Int[]), :blue => (Int[], Int[]))
                    
                    for a in alive_cells_now
                        c = get_trait_color(a.apoptosis_susceptibility)
                        push!(color_groups[c][1], a.pos[1])
                        push!(color_groups[c][2], a.pos[2])
                    end

                    for (c, (xs, ys)) in color_groups
                        if !isempty(xs)
                            # label="" prevents duplicate entries in the legend
                            scatter!(p_dish, xs, ys, color=c, marker=:circle, 
                                     markersize=2, markerstrokewidth=0, label="")
                        end
                    end
                end
                
                # --- RIGHT PANEL: Live Mean Trait Line Graph ---
                p_line = plot(global_hours, mean_trait_history,
                              ribbon=std_trait_history, fillalpha=0.3, # This creates the shaded Std Dev area
                              title="Population Trait Evolution",
                              xlabel="Global Time (Hours)",
                              ylabel="Apoptosis Prob (Mean ± Std)",
                              ylims=(0.0, 1.0),
                              xlims=(0, passages * SIMULATION_STEPS * TIME_STEP_DT),
                              legend=false, linewidth=3, color=:black)
                
                # Combine them side-by-side
                combined_plot = plot(p_dish, p_line, layout=(1,2), size=(800, 400))
                frame(anim, combined_plot)
            end
        end
        
        alive_cells = filter(a -> a.alive, collect(allagents(model)))
        final_alive = length(alive_cells)
        
        if passage < passages && final_alive > 0
            # Sample cells to seed the next passage
            num_to_sample = max(1, round(Int, final_alive * passage_fraction))
            sampled_cells = sample(alive_cells, num_to_sample, replace=false)
            current_traits = [a.apoptosis_susceptibility for a in sampled_cells]
        elseif final_alive == 0
            println("Extinction recorded in video at passage $passage.")
            break 
        end
    end
     
    # Save the final compiled GIF
    output_path = "Project/Figures/PetriDish/$(run_id).gif"
    gif(anim, output_path, fps=fps)
    println("Saved timelapse to: $output_path")
end

# ==========================================
# --- RUN YOUR CUSTOM EXPERIMENT HERE ---
# ==========================================

# Define the specific combination you want to visualize:
TARGET_DOSE = 1.0 #make sure it is an decimal
TARGET_MUTATION_RATE = 0.05
PASSAGES_TO_RUN = 5

# Create a clear name for the resulting file
EXPERIMENT_NAME = "Visual_Dose_$(TARGET_DOSE)_Mut_$(TARGET_MUTATION_RATE)"

# Run it!
run_custom_timelapse(
    TARGET_DOSE, 
    TARGET_MUTATION_RATE; 
    passages = PASSAGES_TO_RUN,
    run_id = EXPERIMENT_NAME,
    capture_interval = 10, # Takes a picture every 10 simulation steps
    fps = 10               # Playback speed of the final GIF
)