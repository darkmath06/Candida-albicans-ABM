using Random
using StatsBase
using Images
using ImageFiltering
using Plots

# ==========================================
# --- GLOBAL PARAMETERS (1 px = 5 μm) ---
# ==========================================

# Total physical size remains 500μm x 500μm.
# Since 1 px = 5μm, the grid is now 100x100.
const GRID_SIZE = 100         
const CELL_DIAMETER_UM = 5    # 5μm diameter cells
const CELL_PIXEL_SIZE = 1     # 1 pixel represents exactly 5μm

const INITIAL_CELLS = 2     # Starting population
const INIT_PCD_PLUS_FRACTION = 0.50

const INIT_NUTRIENT_LEVEL = 100.0
const INIT_ANTIFUNGAL_LEVEL = 50.0

# ==========================================
# --- AGENT TYPES ---
# ==========================================

abstract type CellAgent end

mutable struct PCDPlus <: CellAgent
    x::Int
    y::Int
    alive::Bool
    biomass::Float64
end

mutable struct PCDMinus <: CellAgent
    x::Int
    y::Int
    alive::Bool
    biomass::Float64
end

# Constructors
PCDPlus(x, y; biomass=1.0) = PCDPlus(x, y, true, biomass)
PCDMinus(x, y; biomass=1.0) = PCDMinus(x, y, true, biomass)

# ==========================================
# --- MODEL STRUCTURE ---
# ==========================================

mutable struct PetriDishModel
    size::Int
    nutrient_layer::Matrix{Float64}
    ANTIFUNGAL_layer::Matrix{Float64}
    occupied_layer::Matrix{Bool} # Tracking pixel occupancy
    agents::Vector{CellAgent}
end

"""
    is_space_clear(model_size, occupied_layer, x, y)

Checks if the specific pixel (x, y) is available. 
In this scale (1px = 5um), a cell occupies exactly 1 pixel.
"""
function is_space_clear(model_size::Int, occupied_layer::Matrix{Bool}, x::Int, y::Int)
    if x < 1 || x > model_size || y < 1 || y > model_size
        return false # Out of bounds
    end
    # Since 1px = 1 cell, we just check if this specific pixel is taken
    return !occupied_layer[y, x]
end

function PetriDishModel()
    s = GRID_SIZE
    n_layer = fill(INIT_NUTRIENT_LEVEL, s, s)
    f_layer = fill(INIT_ANTIFUNGAL_LEVEL, s, s)
    occ_layer = zeros(Bool, s, s)
    
    agents = CellAgent[]
    num_plus = Int(round(INITIAL_CELLS * INIT_PCD_PLUS_FRACTION))

    # Initialize agents with pixel-level exclusion
    for i in 1:INITIAL_CELLS
        placed = false
        attempts = 0
        while !placed && attempts < 1000
            rx, ry = rand(1:s), rand(1:s)
            if is_space_clear(s, occ_layer, rx, ry)
                occ_layer[ry, rx] = true
                if i <= num_plus
                    push!(agents, PCDPlus(rx, ry))
                else
                    push!(agents, PCDMinus(rx, ry))
                end
                placed = true
            end
            attempts += 1
        end
    end

    return PetriDishModel(s, n_layer, f_layer, occ_layer, agents)
end

# ==========================================
# --- INITIAL STATE PLOTTING ---
# ==========================================

function plot_initial_state()
    println("Initializing 100x100 Environment (1px = 5μm)...")
    model = PetriDishModel()

    # Subplot 1: Nutrient Heatmap
    p1 = heatmap(model.nutrient_layer, 
                 title="Nutrients (Initial)", 
                 color=:viridis, 
                 aspect_ratio=:equal,
                 clims=(0, INIT_NUTRIENT_LEVEL))

    # Subplot 2: ANTIFUNGAL Heatmap
    p2 = heatmap(model.ANTIFUNGAL_layer, 
                 title="ANTIFUNGAL (Initial)", 
                 color=:ice, 
                 aspect_ratio=:equal,
                 clims=(0, INIT_ANTIFUNGAL_LEVEL))
    
    # Subplot 3: Cells (Pixel-based visualization)
    p3 = plot(title="Initial Cell Positions", 
              xlims=(0.5, model.size + 0.5), ylims=(0.5, model.size + 0.5), 
              aspect_ratio=:equal, legend=:outertopright)

    plus_x = [a.x for a in model.agents if isa(a, PCDPlus)]
    plus_y = [a.y for a in model.agents if isa(a, PCDPlus)]
    minus_x = [a.x for a in model.agents if isa(a, PCDMinus)]
    minus_y = [a.y for a in model.agents if isa(a, PCDMinus)]

    # Markersize is adjusted to look correct for a 100x100 grid
    if !isempty(plus_x)
        scatter!(p3, plus_x, plus_y, label="PCD+", color=:blue, markersize=4, markerstrokewidth=0.5)
    end
    if !isempty(minus_x)
        scatter!(p3, minus_x, minus_y, label="PCD-", color=:red, markersize=4, markerstrokewidth=0.5)
    end

    # Combine into a 3-pane layout
    final_plot = plot(p1, p2, p3, layout=(1, 3), size=(1400, 450))
    
    display(final_plot)
    println("Simulation grid of $(GRID_SIZE)x$(GRID_SIZE) plotted.")
end

plot_initial_state()