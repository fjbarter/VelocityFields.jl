# Field.jl

module FieldModule

using Statistics

export Field, compute_curl, compute_vorticity

# Define a Field struct to hold any computed field (vector or scalar) and metadata.
struct Field
    avg_field::Array{Float64,3}        # The averaged field (size: [n_bins_x, n_bins_y, n_comp])
    origin::Tuple{Float64,Float64}      # True lower-left coordinate (x_min, y_min) of the field
    bin_size::Float64                   # The size of each bin (in the geometry's coordinate system)
    geometry_type::Symbol               # Symbol indicating the coordinate system (:plane or :cylindrical)
    field_type::Symbol                  # :vector or :scalar
    quantity::Symbol                    # Name of the quantity, e.g., :velocity, :vorticity
end

"""
    compute_curl(field::Field) -> Field

Computes the 2D curl (vorticity) field from the vector `field` stored in the given `Field` instance.
Returns a new `Field` instance containing the scalar curl at each grid point.

Arguments:
- `field::Field`: A Field instance containing a vector field with:
    - `avg_field[:,:,1]` = U (x-component)
    - `avg_field[:,:,2]` = V (y-component)
    - `bin_size`: grid spacing (uniform in x and y)

Returns:
- `Field` instance where:
    - `avg_field[:,:,1]` holds ω(x,y) = ∂V/∂x - ∂U/∂y
    - `field_type` = :scalar, `quantity` = :vorticity
"""
function compute_curl(field::Field)
    # Check input field is vector
    n_bins_x, n_bins_y, n_comp = size(field.avg_field)
    if n_comp != 2
        error("compute_curl: Expected a 2-component vector field")
    end
    dx = field.bin_size
    U = field.avg_field[:,:,1]
    V = field.avg_field[:,:,2]

    # Initialize curl array
    ω = zeros(n_bins_x, n_bins_y)

    # Central differences for interior
    for i in 2:(n_bins_x-1), j in 2:(n_bins_y-1)
        dV_dx = (V[i+1, j] - V[i-1, j]) / (2*dx)
        dU_dy = (U[i, j+1] - U[i, j-1]) / (2*dx)
        ω[i, j] = dV_dx - dU_dy
    end

    # Package into a Field: treat as 1-component scalar field
    # Expand to 3D array [nx, ny, 1]
    curl_field = reshape(ω, n_bins_x, n_bins_y, 1)
    if field.quantity == :velocity
        quantity = :vorticity
    else
        quantity = :curl
    end
    return Field(curl_field, field.origin, dx, field.geometry_type, :scalar, quantity)
end

"""
    compute_vorticity(field::Field) -> Float64

Computes the RMS vorticity (root-mean-square of the curl) of the vector field.
Internally uses `compute_curl` to get the scalar curl Field.
Returns a single Float64 value.
"""
function compute_vorticity(field::Field)
    # Compute the curl field
    curl_field = compute_curl(field)
    ω = curl_field.avg_field[:,:,1]

    # Exclude edges
    interior = ω[2:end-1, 2:end-1]

    # RMS: sqrt(mean(ω^2))
    rms_vorticity = sqrt(mean(interior .^ 2))
    return rms_vorticity
end

end # module FieldModule
