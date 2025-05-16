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
# @time raw_field = generate_field(directory, plane; bin_size=0.005)

# base = [0.0, 0.0, 0.0]
# axis_vector = [0.0, 0.0, 1.0]

# cylinder = Cylinder(base, axis_vector)

# @time raw_field = generate_field(directory, cylinder; bin_size=0.005)

@time raw_field = csv_to_field("field_csvs/fillheight_63_g_100_y_z.csv")

plot_field(raw_field; figure_name="field.png")

# field_to_csv(raw_field, "raw_field.csv")

# field = csv_to_field("raw_field.csv")

# plot_field(field; figure_name="field.png")

# raw_curl_field = compute_curl(field)

# field_to_csv(raw_curl_field, "raw_curl_field.csv")

# curl_field = csv_to_field("raw_curl_field.csv")

# plot_field(curl_field; figure_name="curl_field.png")