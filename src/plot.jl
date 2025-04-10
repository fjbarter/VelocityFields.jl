# plot.jl
# This module contains the plotting functions, as well as a custom quiver
# function for plotting a series of arrows

module Plot

using ..FieldModule

using LinearAlgebra
using Plots

export plot_field

"""
    custom_quiver!(X::AbstractVector, Y::AbstractVector, U::AbstractVector, V::AbstractVector;
                    headwidth::Float64 = 3.0, headlength::Float64 = 5.0, linewidth::Float64 = 1.0,
                    color = :white)

Draws a 2-D field of arrows on the current plot, similar to Plots.jl’s built-in `quiver!` function but with custom control over the arrowhead dimensions. In this routine, each arrow is centered at the coordinates given by `X` and `Y`. The arrow is constructed as follows:
  
- The total arrow length, `L`, is computed from the vector components (`U`, `V`).
- The arrow is centered such that its tail is at `(X - L/2 * u, Y - L/2 * v)` and its tip is at `(X + L/2 * u, Y + L/2 * v)`, where `(u,v)` is the normalized vector.
- A shaft is drawn from the tail to a point that is `headlength * linewidth` units from the tip.
- A filled triangular arrowhead is drawn at the tip with a width of `headwidth * linewidth`, using the perpendicular vector for proper offset.

# Arguments
- `X`: One-dimensional array of x-coordinates for the arrow centers.
- `Y`: One-dimensional array of y-coordinates for the arrow centers.
- `U`: One-dimensional array of x-components for the arrow vectors.
- `V`: One-dimensional array of y-components for the arrow vectors.

# Keyword Arguments
- `headwidth`: Factor to scale the arrowhead width relative to the base shaft width (default: 3.0).
- `headlength`: Factor to scale the arrowhead length relative to the base shaft width (default: 5.0).
- `linewidth`: The width of the arrow shaft (default: 1.0).
- `color`: The color to use for the arrow shaft and head (default: `:white`).

# Behavior
The function iterates over each element in the input vectors, computes and normalizes the arrow direction, then uses the specified multipliers to determine both the shaft and triangular head dimensions. The arrows are drawn using `plot!` calls with the legend disabled.

# Returns
Nothing; the function modifies the current plot in place.
"""
function custom_quiver!(X::AbstractVector, Y::AbstractVector, U::AbstractVector, V::AbstractVector;
                                  headwidth::Float64 = 3.0, headlength::Float64 = 5.0,
                                  linewidth::Float64 = 1.0, color = :white)
    n = length(X)
    for i in 1:n
        # Use (X[i], Y[i]) as the center of the arrow.
        center_x = X[i]
        center_y = Y[i]
        L = sqrt(U[i]^2 + V[i]^2)
        if L == 0
            continue  # Skip arrows with zero magnitude.
        end
        # Normalize the vector.
        ux = U[i] / L
        uy = V[i] / L

        # Compute tail and tip so that the arrow is centered at (center_x, center_y).
        tail_x = center_x - (L/2) * ux
        tail_y = center_y - (L/2) * uy
        tip_x  = center_x + (L/2) * ux
        tip_y  = center_y + (L/2) * uy

        # Define the arrowhead length in data units.
        head_len = headlength * linewidth
        if L < head_len
            # If the arrow is too short, plot a simple line.
            plot!([tail_x, tip_x], [tail_y, tip_y], lw=linewidth, color=color, legend=false)
            continue
        end

        # The shaft ends head_len away from the tip.
        shaft_end_x = tip_x - head_len * ux
        shaft_end_y = tip_y - head_len * uy

        # Draw the shaft.
        plot!([tail_x, shaft_end_x], [tail_y, shaft_end_y], lw=linewidth, color=color, legend=false)

        # Compute the perpendicular direction (for the arrowhead width).
        perp_x = -uy
        perp_y = ux

        # Arrowhead width: multiply shaft width by headwidth factor.
        hw = headwidth * linewidth

        # Compute the base corners of the arrowhead.
        left_x  = shaft_end_x + (hw/2) * perp_x
        left_y  = shaft_end_y + (hw/2) * perp_y
        right_x = shaft_end_x - (hw/2) * perp_x
        right_y = shaft_end_y - (hw/2) * perp_y

        # Arrowhead vertices: tip, left corner, right corner.
        arrow_x = [tip_x, left_x, right_x]
        arrow_y = [tip_y, left_y, right_y]

        # Draw the arrowhead as a filled shape.
        plot!(arrow_x, arrow_y, seriestype=:shape, fillcolor=color, linecolor=color, legend=false)
    end
    return nothing
end


function plot_field(field::Field; arrow_length::Union{Float64,Nothing}=nothing, 
                    figure_name::Union{String,Nothing}=nothing)
    avg_field = field.avg_field
    origin = field.origin
    bin_size = field.bin_size
    geom_type = field.geometry_type

    if !isdir("plots") mkdir("plots") end

    if isnothing(figure_name)
        figure_name = "velocity_field.png"
    end

    if isnothing(arrow_length)
        arrow_length = 0.7 * bin_size  # 70% of bin_size
    end

    # Determine x/y coordinates and possibly mirror the field for cylindrical fields.
    if geom_type == :cylindrical
        # For the cylindrical case, the first bin dimension is the radial coordinate (r)
        # and the second is the axial coordinate (z).
        n_bins_r = size(avg_field, 1)
        n_bins_z = size(avg_field, 2)

        # Positive radial bin centers.
        pos_r = [origin[1] + (i - 0.5) * bin_size for i in 1:n_bins_r]
        z_coords = [origin[2] + (j - 0.5) * bin_size for j in 1:n_bins_z]

        # Mirror the radial bins: negative r are a mirror of the positive ones.
        mirrored_r = vcat([-r for r in reverse(pos_r)], pos_r)
        n_bins_r_full = length(mirrored_r)

        # Create a new field array with dimensions [2*n_bins_r, n_bins_z, 2].
        mirrored_field = Array{Float64}(undef, n_bins_r_full, n_bins_z, 2)
        # Fill negative-r half by mirroring the field (flip the sign of the radial velocity component).
        for i in 1:n_bins_r
            src_idx = n_bins_r - i + 1
            for j in 1:n_bins_z
                v = avg_field[src_idx, j, :]
                mirrored_field[i, j, 1] = -v[1]  # flip radial component
                mirrored_field[i, j, 2] = v[2]
            end
        end
        # Copy the original field into the positive-r half.
        for i in 1:n_bins_r, j in 1:n_bins_z
            mirrored_field[n_bins_r + i, j, :] = avg_field[i, j, :]
        end

        # Use the mirrored field and set coordinate labels.
        avg_field_to_plot = mirrored_field
        x_coords = mirrored_r
        y_coords = z_coords
        xlabel = "r"
        ylabel = "z"
    else
        # For a plane, simply compute the bin centers.
        n_bins_x = size(avg_field, 1)
        n_bins_y = size(avg_field, 2)
        x_coords = [origin[1] + (i - 0.5) * bin_size for i in 1:n_bins_x]
        y_coords = [origin[2] + (j - 0.5) * bin_size for j in 1:n_bins_y]
        avg_field_to_plot = avg_field
        xlabel = "x′"
        ylabel = "y′"
    end

    # Compute the field magnitude and fixed-length arrow components.
    n_x, n_y, two = size(avg_field_to_plot)
    mag = zeros(n_x, n_y)
    arrow_u = zeros(n_x, n_y)
    arrow_v = zeros(n_x, n_y)
    for i in 1:n_x, j in 1:n_y
        v = avg_field_to_plot[i, j, :]
        mag[i, j] = norm(v)
        θ = mag[i, j] > 0 ? atan(v[2], v[1]) : 0.0
        arrow_u[i, j] = arrow_length * cos(θ)
        arrow_v[i, j] = arrow_length * sin(θ)
    end

    # Compute plot limits.
    xlims = (x_coords[1] - bin_size/2, x_coords[end] + bin_size/2)
    ylims = (y_coords[1] - bin_size/2, y_coords[end] + bin_size/2)

    # Create the heatmap using velocity magnitude.
    p = heatmap(x_coords, y_coords, mag',
        aspect_ratio = 1,
        colorbar_title = "Speed (m/s)",
        xlabel = xlabel, ylabel = ylabel,
        xlims = xlims,
        ylims = ylims)

    # Build meshgrid-like arrays for the quiver overlay.
    X = repeat(x_coords, 1, n_y)
    Y = repeat(y_coords', n_x, 1)

    # Overlay constant-length arrows.
    custom_quiver!(vec(X), vec(Y), vec(arrow_u), vec(arrow_v);
        color = :white,
        linewidth = 1.0,
        headwidth = arrow_length * 0.2,
        headlength = arrow_length * 0.2)

    display(p)
    println("Saved plot $figure_name")
    flush(stdout)
    savefig(p, joinpath("plots", figure_name))
    return nothing
end


end # module Plot