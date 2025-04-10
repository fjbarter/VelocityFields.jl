# geometry.jl

module Geometry

using LinearAlgebra
using Packing3D

export Plane, Cylinder, transform_file_data

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
    Cylinder(base::Vector{Float64}, axis::Vector{Float64})

Constructs a Cylinder defined by a base point and an axis vector.

# Arguments
- `base`: A 3-element vector `[x₀, y₀, z₀]` representing the center of the cylinder's bottom.
- `axis`: A 3-element vector `[a, b, c]` representing the cylinder's axis. The length of this vector may represent the cylinder’s height.
  
# Fields
- `base`: The provided base point.
- `axis`: The normalized axis vector (used as the new z'-axis).
- `x`: The new x'-basis vector computed as the cross product of the normalized axis with the new y'-basis.
- `y`: The new y'-basis vector, computed by taking an arbitrary vector crossed with the normalized axis and then normalizing.

# Details
- The inner constructor normalizes the axis vector, then constructs the in-plane basis vectors as:
  1. Picks an arbitrary vector that is not nearly parallel to the axis.
  2. Computes the new y'-axis as the normalized cross product of that arbitrary vector with the cylinder's normalized axis.
  3. Computes the new x'-axis as the cross product of the normalized axis with the new y'-axis.
  
This ensures that `[x, y, axis]` forms a right-handed orthonormal basis.
"""
struct Cylinder
    base::Vector{Float64}
    axis::Vector{Float64}
    x::Vector{Float64}  # New x'-basis vector.
    y::Vector{Float64}  # New y'-basis vector.

    function Cylinder(base::Vector{Float64}, axis::Vector{Float64})
        # Validate inputs are 3-element vectors.
        if length(base) != 3 || length(axis) != 3
            error("Cylinder: Both base and axis must be 3-element vectors.")
        end

        # Normalize the axis vector.
        norm_val = sqrt(sum(axis.^2))
        if norm_val == 0
            error("Cylinder: The axis vector cannot be the zero vector.")
        end
        normalized_axis = axis / norm_val

        # Choose an arbitrary vector that is not parallel to the axis.
        # We select [0, 0, 1] unless the z-component of the axis is nearly 1 (in absolute value), 
        # in which case we choose [0, 1, 0].
        arbitrary = abs(normalized_axis[3]) < 0.9 ? [0.0, 0.0, 1.0] : [0.0, 1.0, 0.0]

        # Compute the new y'-basis vector.
        y = normalize(cross(arbitrary, normalized_axis))
        
        # Compute the new x'-basis vector.
        x = cross(normalized_axis, y)

        new(base, normalized_axis, x, y)
    end
end


"""
    transform_file_data(file::String, geom::Union{Plane, Cylinder}) -> Dict

Reads the VTK file specified by `file` (using `read_vtk_file`), extracts particle positions 
and (if available) velocities, and transforms these vectors into a new coordinate system 
defined by the provided geometry. For a Plane, the transformation applies a translation by subtracting 
`plane.point` and a rotation using the matrix Tᵀ (with T = [plane.u  plane.v  plane.normal]). 

For a Cylinder, the transformation:
  1. Translates points by subtracting `cyl.base`.
  2. Rotates points using the matrix Tᵀ (with T = [cyl.x, cyl.y, cyl.axis]) so that the cylinder’s 
     axis aligns with the z'-axis.
  3. Converts the in‑plane coordinates (x' and y') to cylindrical coordinates (r, θ) using 
     `convert_to_cylindrical` (which assumes a z‑aligned system with origin at [0, 0]). 
     
In the Cylinder case, the final coordinates become [r, θ, z] (with θ in [0, 2π]). For both types, 
if a velocity field exists it is rotated using the same transformation (without further conversion).
"""
function transform_file_data(file::String, geom::Union{Plane, Cylinder})
    # Read data using Packing3D's VTK reader.
    data = read_vtk_file(file)

    # Extract original points.
    points = data[:points]
    # Ensure points is a 2D Array (N x 3). If not, assume it is a vector of 3-element vectors and convert.
    if ndims(points) != 2
        # Convert a vector of vectors into an N×3 matrix.
        points = hcat(points...)'
    end
    N, nc = size(points)
    if nc != 3
        error("Expected points to be an N×3 array, got size $(size(points))")
    end

    # Declare variable T_transpose (will be set in either branch)
    T_transpose = nothing

    if geom isa Plane
        # Build the transformation matrix:
        # Columns are the new basis vectors: [geom.u, geom.v, geom.normal].
        T = hcat(geom.u, geom.v, geom.normal)
        T_transpose = T'
        # Transform each point: p' = Tᵀ * (p - geom.point).
        new_points = similar(points)
        for i in 1:N
            new_points[i, :] = T_transpose * (points[i, :] .- geom.point)
        end
        data[:points] = new_points

    elseif geom isa Cylinder
        # Build the transformation matrix with columns [geom.x, geom.y, geom.axis].
        T = hcat(geom.x, geom.y, geom.axis)
        T_transpose = T'
        # Transform each point: subtract the cylinder's base and then rotate.
        new_points = similar(points)
        for i in 1:N
            new_points[i, :] = T_transpose * (points[i, :] .- geom.base)
        end
        # Convert the in-plane (x' and y') Cartesian coordinates to cylindrical (r, θ)
        # (convert_to_cylindrical assumes the cylinder axis is z-aligned and the origin is at (0,0)).
        r, theta = convert_to_cylindrical(new_points[:, 1], new_points[:, 2])
        # Construct the final coordinate array: [r, θ, z].
        data[:points] = hcat(r, theta, new_points[:, 3])
    else
        error("Unsupported geometry type. Expected Plane or Cylinder.")
    end

    # Transform velocities if available.
    if haskey(data[:point_data], :v)
        velocities = data[:point_data][:v]
        # Ensure velocities is a 2D Array as well.
        if ndims(velocities) != 2
            velocities = hcat(velocities...)'
        end
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
