# example.jl

# include(joinpath("..", "src", "VelocityFields.jl"))

# using Pkg
# Pkg.develop(url="https://github.com/fjbarter/VelocityFields.jl")

using Distributed
addprocs(4)
@everywhere using VelocityFields

directory = joinpath("Y:", "RAM_fields_test", "study", "RAM_80", "post")

ref_point = [0.0, 0.0, 0.0]
normal_vector = [0.0, 1.0, 0.0]

plane = Plane(ref_point, normal_vector)

# Generate the Field instance.
field = generate_field(directory, plane; bin_size=0.005)

# Plot the field.
plot_field(field)