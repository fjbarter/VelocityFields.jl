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

# CSV support
using CSV
using DataFrames

export generate_field, plot_field, Plane, Cylinder, Field, compute_vorticity, field_to_csv, csv_to_field, compute_curl

# --- Helper function for per-file processing ---
# Processes an individual file to compute a dictionary of bin accumulations.
function process_file_helper(
    file::String,
    geom,
    bin_size_reciprocal::Float64,
    data_1_ids,
    data_2_ids,
    split,
    vector_symbol
    )

    local_dict = Dict{Tuple{Int,Int}, Tuple{Vector{Float64}, Int}}()

    # Transform file data using our generic transform routine.
    data = Geometry.transform_file_data(file, geom, data_1_ids, data_2_ids, split, vector_symbol)
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
            # Get the rotated velocity (in [vₓ′, vᵧ′, v_z′] form).
            v_rot = vels[i, :]
            # Retrieve the polar angle from the transformed point (column 2) -- this is θ.
            θ = pts[i, 2]
            # Project the in-plane velocity [vₓ′, vᵧ′] onto the radial direction.
            v_r = cos(θ) * v_rot[1] + sin(θ) * v_rot[2]
            # Axial (z) velocity remains the same.
            v_z = v_rot[3]
            vel = [v_r, v_z]
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


function compute_file_index(file::String)::Int
    # Extract base filename without path and extension
    fname = split(split(file, r"[\\/]")[end], '.')
    base = fname[1]
    m = match(r".*_(\d+)$", base)
    @assert m !== nothing "Filename $file does not match expected pattern"
    return parse(Int, m.captures[1])
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
function generate_field(
    dataset_dir::String, geom;
    bin_size::Union{Float64,Nothing}=nothing,
    start_idx::Union{Nothing,Int}=nothing,
    end_idx::Union{Nothing,Int}=nothing,
    split_by::Union{Symbol, Nothing}=nothing,
    threshold::Union{<:Real, Nothing}=nothing,
    split::Union{Int64, Nothing}=nothing,
    long_average::Union{Bool, Nothing}=nothing,
    timestep::Union{<:Real, Nothing}=nothing,
    vector_type::Union{Symbol, Nothing}=nothing)

    if isnothing(vector_type)
        vector_symbol = :v
        quantity = :velocity
    else
        vector_symbol = vector_type
        quantity = vector_type
    end
    
    # Load dataset info
    ds = DataSet(dataset_dir; start_idx=start_idx, end_idx=end_idx)

    # Optional data splitting setup
    if !isnothing(split_by) && !isnothing(threshold) && !isnothing(split)
        initial_data = read_vtk_file(ds.files[1])
        data_1_ids, data_2_ids = split_data(initial_data; split_by=split_by, threshold=threshold)
    else
        data_1_ids = nothing
        data_2_ids = nothing
    end
    
    # Estimate bin size if not provided
    if isnothing(bin_size)
        data_first = Geometry.transform_file_data(ds.files[1], geom, data_1_ids, data_2_ids, split, vector_symbol)
        xs = data_first[:points][:, 1]
        span = maximum(xs) - minimum(xs)
        bin_size = 0.05 * span
    end
    bin_size_reciprocal = 1 / bin_size

    if long_average === true
        @assert !isnothing(timestep) "Timestep must be provided for long_average=true"
        @assert vector_symbol == :v "Vector type must be velocity for long_average=true"

        # Select start and end files
        file_start = ds.files[1]
        file_end   = ds.files[end]

        # Compute time difference
        idx_s = compute_file_index(file_start)
        idx_e = compute_file_index(file_end)
        dt = (idx_e - idx_s) * timestep

        # Transform both start and end files
        data_s = Geometry.transform_file_data(file_start, geom, data_1_ids, data_2_ids, split, vector_symbol)
        data_e = Geometry.transform_file_data(file_end,   geom, data_1_ids, data_2_ids, split, vector_symbol)

        pts_s = data_s[:points]
        pts_e = data_e[:points]

        # Accumulate displacement-based average velocity per particle
        global_dict = Dict{Tuple{Int,Int}, Tuple{Vector{Float64}, Int}}()
        num_particles = size(pts_s, 1)
        for i in 1:num_particles
            if geom isa Geometry.Plane
                # Use start X,Y for binning
                x = pts_s[i, 1]
                y = pts_s[i, 2]
                # Displacement in-plane
                disp = pts_e[i, 1:2] .- pts_s[i, 1:2]
                vel = disp ./ dt
            elseif geom isa Geometry.Cylinder
                # Start coordinates r and z for binning
                r_s = pts_s[i, 1]
                z_s = pts_s[i, 3]
                r_e = pts_e[i, 1]
                z_e = pts_e[i, 3]
                x = r_s
                y = z_s
                # Displacements
                disp_r = r_e - r_s
                disp_z = z_e - z_s
                vel = [disp_r / dt, disp_z / dt]
            else
                error("Unsupported geometry type in long-average mode.")
            end
            # Bin indices
            i_bin = floor(Int, x * bin_size_reciprocal)
            j_bin = floor(Int, y * bin_size_reciprocal)
            key = (i_bin, j_bin)
            if haskey(global_dict, key)
                sum_vel, count = global_dict[key]
                global_dict[key] = (sum_vel .+ vel, count + 1)
            else
                global_dict[key] = (copy(vel), 1)
            end
        end
    else
        # Standard parallel map-reduce
        local_dicts = pmap(file -> process_file_helper(file, geom, bin_size_reciprocal, data_1_ids, data_2_ids, split, vector_symbol), ds.files)
        global_dict = reduce(merge_dicts, local_dicts)
    end

    # Post-processing to form the Field
    all_i_bins = [k[1] for k in keys(global_dict)]
    all_j_bins = [k[2] for k in keys(global_dict)]
    global_min_i, global_max_i = minimum(all_i_bins), maximum(all_i_bins)
    global_min_j, global_max_j = minimum(all_j_bins), maximum(all_j_bins)
    n_bins_i = global_max_i - global_min_i + 1
    n_bins_j = global_max_j - global_min_j + 1
    x_min = global_min_i * bin_size
    y_min = global_min_j * bin_size
    avg_field = fill(NaN, n_bins_i, n_bins_j, 2)
    for ((i_bin, j_bin), (sum_vel, count)) in global_dict
        ii = i_bin - global_min_i + 1
        jj = j_bin - global_min_j + 1
        avg_field[ii, jj, :] .= sum_vel ./ count
    end
    geometry_type = (geom isa Geometry.Plane) ? :plane : :cylindrical
    return Field(avg_field, (x_min, y_min), bin_size, geometry_type, :vector, quantity)
end

"""
    field_to_csv(field::Field, filepath::String)

Write any Field (vector or scalar) to CSV. Metadata headers include field_type and quantity.  
Vector fields produce columns x,y,u,v. Scalar fields produce x,y,<quantity>.
"""
function field_to_csv(field::Field, filepath::String)
    avg = field.avg_field
    origin_x, origin_y = field.origin
    bin = field.bin_size
    n_i, n_j, n_comp = size(avg)
    ftype = field.field_type
    qty = string(field.quantity)

    open(filepath, "w") do io
        println(io, "# geometry_type=$(field.geometry_type)")
        println(io, "# origin_x=$origin_x")
        println(io, "# origin_y=$origin_y")
        println(io, "# bin_size=$bin")
        println(io, "# field_type=$(ftype)")
        println(io, "# quantity=$qty")

        if ftype == :vector
            println(io, "x,y,u,v")
        else
            println(io, "x,y,$qty")
        end

        for i in 1:n_i, j in 1:n_j
            x = origin_x + (i-1)*bin
            y = origin_y + (j-1)*bin
            if ftype == :vector
                u = avg[i,j,1]
                v = avg[i,j,2]
                println(io, "$x,$y,$u,$v")
            else
                val = avg[i,j,1]
                println(io, "$x,$y,$val")
            end
        end
    end
end

"""
    csv_to_field(filepath::String)

Read CSV with metadata and reconstruct a Field (vector or scalar).
"""
function csv_to_field(filepath::String)
    # Read raw lines and split metadata
    lines = readlines(filepath)
    metadata = Dict{String,String}()
    data_lines = String[]
    for line in lines
        if startswith(line, "#")
            meta = strip(lstrip(line, '#'))
            if occursin("=", meta)
                k, v = split(meta, "=", limit=2)
                metadata[strip(k)] = strip(v)
            end
        elseif isempty(strip(line))
            continue
        else
            push!(data_lines, line)
        end
    end

    # Parse metadata
    geometry_type = Symbol(metadata["geometry_type"])
    origin_x = parse(Float64, metadata["origin_x"])
    origin_y = parse(Float64, metadata["origin_y"])
    bin = parse(Float64, metadata["bin_size"])
    ftype = Symbol(metadata["field_type"])
    qty = metadata["quantity"]

    # The first non-meta line is the CSV header
    header = split(data_lines[1], ',')

    # Prepare column arrays
    xcol = Float64[]
    ycol = Float64[]
    if ftype == :vector
        @assert header == ["x","y","u","v"] "Unexpected header: $header"
        ucol = Float64[]
        vcol = Float64[]
        for row_text in data_lines[2:end]
            vals = split(row_text, ',')
            x = parse(Float64, vals[1])
            y = parse(Float64, vals[2])
            u = parse(Float64, vals[3])
            v = parse(Float64, vals[4])
            push!(xcol, x); push!(ycol, y)
            push!(ucol, u); push!(vcol, v)
        end
        df = DataFrame(:x=>xcol, :y=>ycol, :u=>ucol, :v=>vcol)
    else
        @assert header == ["x","y", qty] "Unexpected header: $header"
        valcol = Float64[]
        for row_text in data_lines[2:end]
            vals = split(row_text, ',')
            x = parse(Float64, vals[1])
            y = parse(Float64, vals[2])
            val = parse(Float64, vals[3])
            push!(xcol, x); push!(ycol, y)
            push!(valcol, val)
        end
        df = DataFrame(:x=>xcol, :y=>ycol, Symbol(qty)=>valcol)
    end

    # Reconstruct grid
    xs = sort(unique(df.x))
    ys = sort(unique(df.y))
    n_i, n_j = length(xs), length(ys)
    n_comp = ftype == :vector ? 2 : 1
    avg = fill(NaN, n_i, n_j, n_comp)
    
    for r in eachrow(df)
        i = Int(floor((r.x - origin_x) / bin)) + 1
        j = Int(floor((r.y - origin_y) / bin)) + 1
        if ftype == :vector
            avg[i, j, 1] = r.u
            avg[i, j, 2] = r.v
        else
            avg[i, j, 1] = r[Symbol(qty)]
        end
    end

    return Field(avg, (origin_x, origin_y), bin, geometry_type, ftype, Symbol(qty))
end

end # module VelocityFields
