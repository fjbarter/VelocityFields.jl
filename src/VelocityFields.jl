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

function custom_quiver!(X::AbstractVector, Y::AbstractVector, U::AbstractVector, V::AbstractVector;
                          headwidth::Float64 = 3.0, headlength::Float64 = 5.0,
                          linewidth::Float64 = 1.0, color = :white)
    n = length(X)
    for i in 1:n
        # Compute the length of the arrow.
        L = sqrt(U[i]^2 + V[i]^2)
        if L == 0
            continue  # Skip arrows with zero magnitude.
        end

        # Normalize the arrow direction.
        ux = U[i] / L
        uy = V[i] / L

        # Define the arrowhead length in data units.
        # Here we interpret headlength as a multiplier on the base shaft width (linewidth)
        head_len = headlength * linewidth

        # Determine the end of the shaft where the arrowhead begins.
        x_base = X[i] + (L - head_len) * ux
        y_base = Y[i] + (L - head_len) * uy

        # Draw the arrow shaft as a line from the base (X[i], Y[i]) to (x_base, y_base).
        plot!([X[i], x_base], [Y[i], y_base], lw=linewidth, color=color)

        # Compute the arrow tip (end of the arrow).
        x_tip = X[i] + L * ux
        y_tip = Y[i] + L * uy

        # Compute a vector perpendicular to the arrow direction.
        perp_x = -uy
        perp_y = ux

        # Define the arrowhead width (again, as a multiplier on the base line width).
        hw = headwidth * linewidth

        # Calculate the left and right base corners of the arrowhead.
        x_left  = x_base + (hw/2) * perp_x
        y_left  = y_base + (hw/2) * perp_y
        x_right = x_base - (hw/2) * perp_x
        y_right = y_base - (hw/2) * perp_y

        # The arrowhead is drawn as a filled triangle defined by the vertices:
        # (x_tip, y_tip), (x_left, y_left), (x_right, y_right)
        arrow_x = [x_tip, x_left, x_right]
        arrow_y = [y_tip, y_left, y_right]
        plot!(arrow_x, arrow_y, seriestype = :shape, fillcolor = color, linecolor = color)
    end
    return nothing
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

    # # Calculate extended limits so that an extra bin margin appears on each side.
    # xlims = (x_origin - bin_size, x_origin + n_bins_x * bin_size + bin_size)
    # ylims = (y_origin - bin_size, y_origin + n_bins_y * bin_size + bin_size)

    # Compute the limits to line the edges up.
    xlims = (x_coords[1] - bin_size/2, x_coords[end] + bin_size/2)
    ylims = (y_coords[1] - bin_size/2, y_coords[end] + bin_size/2)

    # Create the heatmap colored by velocity magnitude.
    p = heatmap(x_coords, y_coords, mag',
        aspect_ratio = 1,
        colorbar_title = "Speed (m/s)",
        xlabel = "x′", ylabel = "y′",
        xlims = xlims,
        ylims = ylims)

    # Create meshgrid-like arrays for quiver overlay.
    X = repeat(x_coords, 1, n_bins_y)
    Y = repeat(y_coords', n_bins_x, 1)

    # Overlay arrows.
    custom_quiver!(
        vec(X), vec(Y),
        vec(arrow_u), vec(arrow_v);
        color=:white,
        linewidth=1,
        headwidth=arrow_length * 0.2,
        headlength=arrow_length * 0.2)

    # Display and save the plot.
    display(p)
    plot_name = "velocity_field.png"
    println("Saved plot $plot_name")
    flush(stdout)
    savefig(p, plot_name)

    return nothing
end

end # module VelocityFields
