# example.jl

using Pkg
# Pkg.develop(url="https://github.com/fjbarter/VelocityFields.jl")
# Pkg.develop(url="https://github.com/fjbarter/Packing3D.jl")

using VelocityFields

directory = "post"

base = [0.0, 0.0, 0.0]
axis_vector = [0.0, 0.0, 1.0]

cylinder = Cylinder(base, axis_vector)

field = generate_field(directory, cylinder; bin_size=0.005)

plot_field(field; figure_name="field.png")

# origin = [0.0, 0.0, 0.0]
# normal_vector = [0.0, 1.0, 0.0]

# plane = Plane(origin, normal_vector)

# field = generate_field(directory, plane; bin_size=0.005)

# plot_field(field; figure_name="x_z_field.png")