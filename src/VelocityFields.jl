# VelocityFields.jl

module VelocityFields

include("dataset.jl")
include("geometry.jl")

# Local imports
using .DataSetModule
using .Geometry

# External imports
using Packing3D
using LinearAlgebra
using Plots
using Distributed

export generate_field, plot_field, Plane, Field

# Define a Field struct to hold the computed averaged velocity field and associated metadata.
struct Field
    avg_field::Array{Float64,3}        # The averaged velocity field (size: [n_bins_x, n_bins_y, 2])
    origin::Tuple{Float64,Float64}       # True lower-left coordinate (x_min, y_min) of the field
    bin_size::Float64                  # The size of each bin, in the plane’s coordinates
end

# --- Helper function for per-file processing ---
# Processes an individual file to compute a dictionary of bin accumulations.
function process_file_helper(file::String, plane::Geometry.Plane, bin_size_reciprocal::Float64)
    local_dict = Dict{Tuple{Int,Int}, Tuple{Vector{Float64}, Int}}()
    # Transform file data into the plane’s coordinate system.
    data = Geometry.transform_file_data(file, plane)
    pts = data[:points]
    if !haskey(data[:point_data], :v)
        @warn "File $file does not contain velocity data under key :v. Skipping file."
        return local_dict
    end
    vels = data[:point_data][:v]
    num_particles = size(pts, 1)
    
    for i in 1:num_particles
        x = pts[i, 1]  # x' coordinate
        y = pts[i, 2]  # y' coordinate
        # Compute provisional bin indices using floor division.
        i_bin = floor(Int, x * bin_size_reciprocal)
        j_bin = floor(Int, y * bin_size_reciprocal)
        # Accumulate the in-plane velocity (first two components).
        vel = vels[i, 1:2]
        key = (i_bin, j_bin)
        if haskey(local_dict, key)
            sum_vel, count = local_dict[key]
            local_dict[key] = (sum_vel .+ vel, count + 1)
        else
            local_dict[key] = (copy(vel), 1)
        end
    end
    return local_dict
end

# --- Merge function for dictionaries ---
# Combines two dictionaries by adding velocity sums and counts for common keys.
function merge_dicts(d1::Dict{Tuple{Int,Int}, Tuple{Vector{Float64}, Int}},
                     d2::Dict{Tuple{Int,Int}, Tuple{Vector{Float64}, Int}})
    for (key, (sum_vel, count)) in d2
        if haskey(d1, key)
            s, c = d1[key]
            d1[key] = (s .+ sum_vel, c + count)
        else
            d1[key] = (sum_vel, count)
        end
    end
    return d1
end

"""
    generate_field(dataset_dir::String, plane::Plane; bin_size::Union{Float64,Nothing}=nothing)

Processes all VTK files in the provided directory (`dataset_dir`) to generate a Field instance.
For each file, it:
  1. Reads the file and transforms its data into the coordinate system defined by `plane`.
  2. Computes provisional bin indices (based on a given bin_size or one estimated from the first file)
     and accumulates the in-plane (x′, y′) velocity vectors.

The function uses a parallel map–reduce strategy:
  - A helper function processes each file independently (using pmap).
  - The resulting local dictionaries are merged to form a global accumulator.
  - Global min/max bin indices are used to compute the true lower-left coordinate (origin) and grid dimensions.
  - A dense grid (initialized with NaNs for empty bins) is filled with averaged velocity values.

Returns a Field instance that encapsulates:
  - `avg_field`: the 3D average velocity field (size: [n_bins_x, n_bins_y, 2]),
  - `origin`: a tuple (x_min, y_min) with the true lower-left coordinates,
  - `bin_size`: the bin size used.
"""
function generate_field(dataset_dir::String, plane::Geometry.Plane; bin_size::Union{Float64,Nothing}=nothing)
    # Load dataset info using our DataSet module.
    ds = DataSet(dataset_dir)

    # If no bin_size is provided, estimate one using the first file.
    if bin_size === nothing
        first_file = ds.files[1]
        data_first = Geometry.transform_file_data(first_file, plane)
        pts = data_first[:points]
        xs = pts[:, 1] # use the x' coordinates
        span = maximum(xs) - minimum(xs)
        bin_size = 0.05 * span  # 5% of the x-span; adjust as needed.
    end

    # Precompute reciprocal for efficiency.
    bin_size_reciprocal = 1 / bin_size

    # Process each file in parallel using pmap.
    # Each worker processes one file and returns a dictionary of bin accumulations.
    local_dicts = pmap(file -> process_file_helper(file, plane, bin_size_reciprocal), ds.files)

    # Merge all local dictionaries into a single global dictionary.
    global_dict = reduce(merge_dicts, local_dicts)

    # Extract the global min and max provisional bin indices.
    all_i_bins = [k[1] for k in keys(global_dict)]
    all_j_bins = [k[2] for k in keys(global_dict)]
    global_min_i = minimum(all_i_bins)
    global_min_j = minimum(all_j_bins)
    global_max_i = maximum(all_i_bins)
    global_max_j = maximum(all_j_bins)

    n_bins_i = global_max_i - global_min_i + 1
    n_bins_j = global_max_j - global_min_j + 1

    # Compute the true lower-left coordinates.
    x_min = global_min_i * bin_size
    y_min = global_min_j * bin_size

    # Initialize the dense array for average velocity with NaN values.
    avg_field = fill(NaN, n_bins_i, n_bins_j, 2)

    # Populate the dense array: for each bin key, compute and store the average velocity.
    for (key, (sum_vel, count)) in global_dict
        i_bin, j_bin = key
        ii = i_bin - global_min_i + 1
        jj = j_bin - global_min_j + 1
        avg_field[ii, jj, :] .= sum_vel ./ count
    end

    return Field(avg_field, (x_min, y_min), bin_size)
end

"""
    plot_field(field::Field; arrow_length::Union{Float64, Nothing}=nothing)

Generates a plot of the velocity field stored in the given Field object.

Arguments:
  - `field`: A Field instance containing:
       - `avg_field`: a 3D array of size [n_bins_x, n_bins_y, 2] of averaged velocity vectors.
       - `origin`: a tuple (x_min, y_min) representing the true lower-left coordinate.
       - `bin_size`: the bin size used during field generation.
  - `arrow_length`: an optional fixed arrow length for the quiver plot. 
    If not provided, it defaults to 70% of the bin_size.

Plotting Details:
  - A heatmap is created using the magnitude (speed) of the averaged velocity in each bin.
  - A quiver plot overlays constant-length arrows (colored white) to indicate the direction of velocity.
  - Bin centers are computed using the true origin so the field aligns with the proper spatial coordinates.

Returns nothing; the plot is displayed on screen.
"""
function plot_field(field::Field; arrow_length::Union{Float64, Nothing}=nothing)
    avg_field = field.avg_field
    origin = field.origin
    bin_size = field.bin_size

    n_bins_x, n_bins_y, two = size(avg_field)
    if two != 2
        error("Expected avg_field to have size [n_bins_x, n_bins_y, 2]")
    end

    if isnothing(arrow_length)
        arrow_length = 0.7 * bin_size  # 70% of bin_size
    end

    # Unpack the true origin.
    x_origin, y_origin = origin

    # Compute bin-center coordinates using the true origin.
    x_coords = [x_origin + (i - 0.5) * bin_size for i in 1:n_bins_x]
    y_coords = [y_origin + (j - 0.5) * bin_size for j in 1:n_bins_y]

    # Preallocate arrays for magnitude and arrow components.
    mag = zeros(n_bins_x, n_bins_y)
    arrow_u = zeros(n_bins_x, n_bins_y)
    arrow_v = zeros(n_bins_x, n_bins_y)

    for i in 1:n_bins_x, j in 1:n_bins_y
        v = avg_field[i, j, :]
        mag[i, j] = norm(v)
        theta = (mag[i, j] > 0) ? atan(v[2], v[1]) : 0.0
        arrow_u[i, j] = arrow_length * cos(theta)
        arrow_v[i, j] = arrow_length * sin(theta)
    end

    # Create the heatmap colored by velocity magnitude.
    p = heatmap(x_coords, y_coords, mag',
        aspect_ratio = 1,
        colorbar_title = "Speed",
        xlabel = "x′", ylabel = "y′",
        title = "Averaged Velocity Field")

    # Create meshgrid-like arrays for quiver.
    X = repeat(x_coords, 1, n_bins_y)
    Y = repeat(y_coords', n_bins_x, 1)

    # Overlay arrows.
    quiver!(vec(X), vec(Y),
        quiver = (vec(arrow_u), vec(arrow_v)),
        color = :white, lw = 1, arrow = true)

    display(p)

    plot_name = "velocity_field.png"
    println("Saved plot $plot_name")
    flush(stdout)
    savefig(p, plot_name)

    return nothing
end

end # module VelocityFields
