# example.jl

using Pkg
# Pkg.develop(url="https://github.com/fjbarter/VelocityFields.jl")
Pkg.develop(url="https://github.com/fjbarter/Packing3D.jl")

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

@time field = generate_field(directory, cylinder; bin_size=0.005)

vorticity = compute_vorticity(field)

# Plot the field.
plot_field(field)

println(vorticity)