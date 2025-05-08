# example.jl

using Pkg
# Pkg.develop(url="https://github.com/fjbarter/VelocityFields.jl")
# Pkg.develop(url="https://github.com/fjbarter/Packing3D.jl")

using VelocityFields

directory = "post"

# ref_point = [0.0, 0.0, 0.0]
# normal_vector = [1.0, 0.0, 0.0]

# plane = Plane(ref_point, normal_vector)

# # Generate the Field instance.
# @time field = generate_field(directory, plane; bin_size=0.005)

origin = [0.0, 0.0, 0.0]
axis_vector = [0.0, 0.0, 0.08]

cylinder = Cylinder(origin, axis_vector)

# @time field = generate_field(directory, cylinder; bin_size=0.005, long_average=true, timestep=1e-5)

@time field_small = generate_field(directory, cylinder; bin_size=0.0043, split_by=:radius, threshold=0.0007, split=1, long_average=true, timestep=1e-5)
@time field_large = generate_field(directory, cylinder; bin_size=0.0043, split_by=:radius, threshold=0.0007, split=2, long_average=true, timestep=1e-5)

max_speed = 0.000083

# Generate plots
plot_field(field_small; figure_name="small_cyl.png", cbar_max=max_speed)
plot_field(field_large; figure_name="large_cyl.png", cbar_max=max_speed)

# vorticity = compute_vorticity(field)

# Plot the field.
# plot_field(field)

# println(vorticity)