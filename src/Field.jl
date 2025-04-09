# Field.jl

module FieldModule

export Field

# Define a Field struct to hold the computed averaged velocity field and associated metadata.
struct Field
    avg_field::Array{Float64,3}        # The averaged velocity field (size: [n_bins_x, n_bins_y, 2])
    origin::Tuple{Float64,Float64}       # True lower-left coordinate (x_min, y_min) of the field
    bin_size::Float64                  # The size of each bin, in the plane’s coordinates
end

end # module FieldModule