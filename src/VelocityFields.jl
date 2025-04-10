# VelocityFields.jl

module VelocityFields

include("Field.jl")
include("dataset.jl")
include("geometry.jl")
include("plot.jl")

# Local imports
using .DataSetModule
using .Geometry
using .Plot
using .FieldModule

# External imports
using Packing3D # Default public API
using Packing3D: get_mesh_bounds # Custom function retrieval
using Distributed

export generate_field, plot_field, Plane, Cylinder, Field

# --- Helper function for per-file processing ---
# Processes an individual file to compute a dictionary of bin accumulations.
function process_file_helper(file::String, geom, bin_size_reciprocal::Float64)
    local_dict = Dict{Tuple{Int,Int}, Tuple{Vector{Float64}, Int}}()
    # Transform file data using our generic transform routine.
    data = Geometry.transform_file_data(file, geom)
    pts = data[:points]
    if !haskey(data[:point_data], :v)
        @warn "File $file does not contain velocity data under key :v. Skipping file."
        return local_dict
    end
    vels = data[:point_data][:v]
    num_particles = size(pts, 1)
    
    for i in 1:num_particles
        if geom isa Geometry.Plane
            # For a plane: use x' and y' (columns 1 and 2).
            x = pts[i, 1]
            y = pts[i, 2]
            # In-plane velocity is given by the first two components.
            vel = vels[i, 1:2]
        elseif geom isa Geometry.Cylinder
            # For a cylinder: after transformation, points are [r, θ, z].
            # Use r (column 1) and z (column 3) as the binned coordinates.
            x = pts[i, 1]  # radial coordinate
            y = pts[i, 3]  # axial coordinate
            # Average the velocity components corresponding to radial and axial directions.
            # (Assuming velocities were rotated similarly, so radial = first component and axial = third.)
            vel = [vels[i, 1], vels[i, 3]]
        else
            error("Unsupported geometry type in process_file_helper.")
        end
        # Compute provisional bin indices using floor division.
        i_bin = floor(Int, x * bin_size_reciprocal)
        j_bin = floor(Int, y * bin_size_reciprocal)
        
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
    generate_field(dataset_dir::String, geom; bin_size::Union{Float64,Nothing}=nothing, relative_to_mesh::Bool=false)

Processes all VTK files in the given directory (`dataset_dir`) to generate a Field instance.
For each file, it:
  1. Reads the file and transforms its data into the coordinate system defined by `geom` (either a Plane or a Cylinder).
  2. Computes provisional bin indices (using a provided or estimated bin_size) and accumulates the appropriate 2D velocity components:
       - For a Plane: averages the in-plane (x′, y′) velocity.
       - For a Cylinder: averages the radial and axial (r, z) velocity.
       
The function uses a parallel map–reduce strategy:
  - Each file is processed independently using `pmap` (via `process_file_helper`).
  - Local dictionaries are merged to form a global accumulator.
  - Global min/max provisional bin indices determine the true lower-left coordinate and grid dimensions.
  - A dense grid (initialized with NaNs for empty bins) is filled with averaged velocity values.

Returns a Field instance encapsulating:
  - `avg_field`: the averaged velocity field (size: [n_bins_x, n_bins_y, 2]),
  - `origin`: a tuple (x_min, y_min) representing the true lower-left coordinate of the field,
  - `bin_size`: the bin size used,
  - `geometry_type`: a Symbol indicating the analysis type (:plane or :cylindrical).
"""
function generate_field(dataset_dir::String, geom; bin_size::Union{Float64,Nothing}=nothing)
    # Load dataset info using our DataSet module.
    ds = DataSet(dataset_dir)
    
    # If no bin_size is provided, estimate one using the first file.
    if bin_size === nothing
        first_file = ds.files[1]
        data_first = Geometry.transform_file_data(first_file, geom)
        pts = data_first[:points]
        # For a Plane, use x' (column 1); for a Cylinder, use the radial coordinate (also column 1).
        xs = pts[:, 1]
        span = maximum(xs) - minimum(xs)
        bin_size = 0.05 * span   # 5% of the span; adjust as needed.
    end
    # Precompute reciprocal for efficiency.
    bin_size_reciprocal = 1 / bin_size
    
    # Process each file in parallel (each returns a dictionary of bin accumulations).
    local_dicts = pmap(file -> process_file_helper(file, geom, bin_size_reciprocal), ds.files)
    
    # Merge all local dictionaries into a single global dictionary.
    global_dict = reduce(merge_dicts, local_dicts)
    
    # Extract global min and max provisional bin indices.
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
    
    # Set the geometry type as a Symbol.
    geometry_type = (geom isa Geometry.Plane) ? :plane : (geom isa Geometry.Cylinder ? :cylindrical : :unknown)
    return Field(avg_field, (x_min, y_min), bin_size, geometry_type)
end



end # module VelocityFields
