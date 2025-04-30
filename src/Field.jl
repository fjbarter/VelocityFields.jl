# Field.jl

module FieldModule

using Statistics

export Field, compute_vorticity

# Define a Field struct to hold the computed averaged velocity field and associated metadata.
struct Field
    avg_field::Array{Float64,3}        # The averaged velocity field (size: [n_bins_x, n_bins_y, 2])
    origin::Tuple{Float64,Float64}       # True lower-left coordinate (x_min, y_min) of the field
    bin_size::Float64                  # The size of each bin (in the geometry's coordinate system)
    geometry_type::Symbol              # Symbol indicating the coordinate system (:plane or :cylindrical)
end


"""
    compute_vorticity(field::Field) -> Float64

Computes a single numerical measure of the vorticity (swirliness) of the velocity
field stored in the given `Field` instance. The 2D vorticity is defined as

    ω(x, y) = ∂V/∂x - ∂U/∂y

where U is the x-component and V is the y-component of the velocity field.

This function approximates the derivatives using central differences over the interior
of the grid (ignoring edge points) and returns the mean absolute vorticity, which can
serve as an objective measure of the rotational (convective) character of the field.

# Arguments
- `field::Field`: A Field instance containing:
    - `avg_field`: A 3D array of size `[n_bins_x, n_bins_y, 2]`, where `avg_field[:,:,1]` holds
      the U (x) component and `avg_field[:,:,2]` holds the V (y) component.
    - `bin_size`: The grid spacing (assumed uniform in x and y) in the plane’s coordinates.

# Returns
- A single `Float64` value representing the mean absolute vorticity (with units inverse to those
  of the grid spacing) computed from the interior of the grid.
"""

function compute_vorticity(field::Field)
    avg_field = field.avg_field
    dx = field.bin_size # use the bin size as the grid spacing

    n_bins_x, n_bins_y, comp = size(avg_field)
    if comp != 2
        error("Expected avg_field to have 2 components per grid cell")
    end

    # Create an array to hold the vorticity values.
    ω = zeros(n_bins_x, n_bins_y)

    # Use central difference approximations for the interior points.
    # For a 2D field:
    #   ∂V/∂x ≈ (V[i+1,j] - V[i-1,j]) / (2*dx)
    #   ∂U/∂y ≈ (U[i,j+1] - U[i,j-1]) / (2*dx)
    # And the vorticity is defined as ω = ∂V/∂x - ∂U/∂y.
    for i in 2:(n_bins_x-1), j in 2:(n_bins_y-1)
        dV_dx = (avg_field[i+1, j, 2] - avg_field[i-1, j, 2]) / (2 * dx)
        dU_dy = (avg_field[i, j+1, 1] - avg_field[i, j-1, 1]) / (2 * dx)
        ω[i, j] = dV_dx - dU_dy
    end

    # Compute the mean absolute vorticity over the interior grid cells.
    interior_vorticity = ω[2:end-1, 2:end-1]
    mean_abs_vorticity = mean(abs.(interior_vorticity))
    return mean_abs_vorticity

end


end # module FieldModule