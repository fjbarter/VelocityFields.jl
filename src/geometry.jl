# geometry.jl

module Geometry

using LinearAlgebra
using Packing3D

export Plane, transform_file_data

"""
    Plane(point::Vector{Float64}, normal::Vector{Float64})

Constructs a Plane defined by a point and a normal vector.

# Arguments
- `point`: A 3-element vector `[x₀, y₀, z₀]` representing a point on the plane.
- `normal`: A 3-element vector `[a, b, c]` representing the plane's normal.

# Fields
- `point`: The provided point.
- `normal`: The normalized normal vector.
- `u`: A unit vector lying in the plane.
- `v`: A second unit vector lying in the plane, such that (`u`, `v`, `normal`) forms an orthonormal basis.

# Details
- The inner constructor normalises the normal vector, then constructs the two in-plane basis vectors:
  1. It picks an arbitrary vector that is not nearly parallel to the normal.
  2. It uses the cross product to compute the first basis vector `u`.
  3. It computes the second basis vector `v` as the cross product of the normal and `u`.
"""
struct Plane
    point::Vector{Float64}
    normal::Vector{Float64}
    u::Vector{Float64}
    v::Vector{Float64}

    function Plane(point::Vector{Float64}, normal::Vector{Float64})
        # Validate inputs are 3-element vectors.
        if length(point) != 3 || length(normal) != 3
            error("Plane: Both point and normal must be 3-element vectors.")
        end

        # Normalise the normal vector.
        norm_val = sqrt(sum(normal.^2))
        if norm_val == 0
            error("Plane: The normal vector cannot be the zero vector.")
        end
        n = normal / norm_val

        # Choose an arbitrary vector that is not parallel to the normal.
        # Here, we select [0, 0, 1] unless the z-component of the normal is too close to 1 (in absolute value).
        # If this is the case, we choose [0, 1, 0]
        
        arbitrary = abs(n[3]) < 0.9 ? [0.0, 0.0, 1.0] : [0.0, 1.0, 0.0]

        # Compute the first in-plane basis vector u.
        u_init = cross(arbitrary, n)
        u_norm = sqrt(sum(u_init.^2))
        if u_norm == 0
            error("Plane: Failed to compute in-plane vector (u) from the arbitrary vector.")
        end
        u = u_init / u_norm

        # Compute the second in-plane basis vector v.
        v = cross(n, u)
        # v is a unit vector by definition since n and u are normalised and orthogonal.
        
        new(point, n, u, v)
    end
end


"""
    transform_file_data(file::String, plane::Plane) -> Dict

Reads the VTK file specified by `file` (using `read_vtk_file`), extracts
particle positions and (if available) velocities, and transforms these vectors
into a new coordinate system defined by the provided `plane`.

The transformation for points applies:
  - A translation by subtracting `plane.point`
  - A rotation using the matrix Tᵀ (where T = [plane.u  plane.v  plane.normal]).
  
Velocities are rotated (without translation) using the same rotation.

The returned dictionary has the same keys as that of `read_vtk_file`, but
with the `:points` (and, if present, `:point_data[:velocity]`) replaced with
the transformed values.
"""
function transform_file_data(file::String, plane::Plane)
    # Read data using Packing3D's VTK reader.
    data = read_vtk_file(file)

    # Extract original points.
    # Expected format: data[:points] is an N×3 array (each row is [x, y, z]).
    points = data[:points]
    N, nc = size(points)
    if nc != 3
        error("Expected points to be an N×3 array, got size ($(size(points)))")
    end

    # Build the transformation matrix.
    # T's columns are the new basis vectors: [plane.u  plane.v  plane.normal].
    T = hcat(plane.u, plane.v, plane.normal)
    T_transpose = T'

    # Transform all points:
    # For each point p, compute p' = Tᵀ * (p - plane.point).
    new_points = similar(points)
    for i in 1:N
        new_points[i, :] = T_transpose * (points[i, :] .- plane.point)
    end
    data[:points] = new_points

    # If a velocity field exists in point_data (assumed to be under key :velocity)
    # transform it by rotating (without translation).
    if haskey(data[:point_data], :v)
        velocities = data[:point_data][:v]
        nvel, ncomp = size(velocities)
        if ncomp != 3
            @warn("Expected velocity field to be an N×3 array, got size $(size(velocities)). Skipping transformation of velocity.")
        else
            new_velocities = similar(velocities)
            for i in 1:nvel
                new_velocities[i, :] = T_transpose * velocities[i, :]
            end
            data[:point_data][:v] = new_velocities
        end
    end

    return data
end


end # module Geometry
