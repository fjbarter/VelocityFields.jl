using Test
using VelocityFields
using VelocityFields: FieldModule, Geometry, DataSetModule
using LinearAlgebra

###########################################################################
# FieldModule Tests
###########################################################################
@testset "FieldModule Tests" begin
    # Create a simple 5×5 field where V = x and U = 0, bin_size = 1.0
    n = 5
    avg = fill(0.0, n, n, 2)
    for i in 1:n, j in 1:n
        avg[i, j, 2] = i  # V-component = i
    end
    field = Field(avg, (0.0, 0.0), 1.0, :plane, :vector, :velocity)

    ω = compute_vorticity(field)
    # For this field, ∂V/∂x = 1 everywhere on interior, so mean_abs_vorticity ≈ 1.0
    @test isapprox(ω, 1.0; atol=1e-8)
end

###########################################################################
# CSV I/O Tests
###########################################################################
@testset "CSV I/O Tests" begin
    # Construct a tiny 2×2 field
    avg = zeros(2,2,2)
    avg[1,1,1] = 0.1; avg[1,1,2] = 0.2
    avg[2,2,1] = 0.3; avg[2,2,2] = 0.4
    origin = (1.5, -0.5)
    bin_size = 0.25
    geom = :cylindrical
    field = Field(avg, origin, bin_size, geom, :vector, :velocity)

    # Write to a temp CSV, then read back
    dir = mktempdir()
    csvfile = joinpath(dir, "test_field.csv")
    try
        field_to_csv(field, csvfile)
        field2 = csv_to_field(csvfile)

        @test field2.origin == field.origin
        @test isapprox(field2.bin_size, field.bin_size; atol=1e-12)
        @test field2.geometry_type == field.geometry_type
        @test size(field2.avg_field) == size(field.avg_field)
        @test all(isapprox.(field2.avg_field, field.avg_field; atol=1e-12))
    finally
        rm(dir; force=true, recursive=true)
    end
end

###########################################################################
# Geometry Tests
###########################################################################
@testset "Geometry Tests" begin
    # Plane basis orthonormality
    p = Geometry.Plane([0.0,0.0,0.0], [0.0,0.0,1.0])
    @test isapprox(norm(p.normal), 1.0; atol=1e-8)
    @test isapprox(norm(p.u), 1.0; atol=1e-8)
    @test isapprox(norm(p.v), 1.0; atol=1e-8)
    @test isapprox(dot(p.normal, p.u), 0.0; atol=1e-8)
    @test isapprox(dot(p.u, p.v), 0.0; atol=1e-8)

    # Cylinder basis orthonormality
    c = Geometry.Cylinder([0.0,0.0,0.0], [0.0,1.0,0.0])
    @test isapprox(norm(c.axis), 1.0; atol=1e-8)
    @test isapprox(norm(c.x), 1.0; atol=1e-8)
    @test isapprox(norm(c.y), 1.0; atol=1e-8)
    @test isapprox(dot(c.axis, c.x), 0.0; atol=1e-8)
    @test isapprox(dot(c.x, c.y), 0.0; atol=1e-8)
end

###########################################################################
# DataSetModule Tests
###########################################################################
@testset "DataSetModule Tests" begin
    # Create a temp directory with matching and non-matching files
    dir = mktempdir()
    try
        # create some files
        touch(joinpath(dir, "particles_1.vtk"))
        touch(joinpath(dir, "particles_2.vtk"))
        touch(joinpath(dir, "ignore_me.txt"))

        # find_files should only list the two .vtk files sorted
        files = DataSetModule.find_files(dir)
        @test length(files) == 2
        @test endswith(files[1], "particles_1.vtk")
        @test endswith(files[2], "particles_2.vtk")

        # DataSet struct slicing
        ds_all = DataSetModule.DataSet(dir)
        @test length(ds_all.files) == 2

        ds_one = DataSetModule.DataSet(dir; start_idx=1, end_idx=1)
        @test length(ds_one.files) == 1
        @test endswith(ds_one.files[1], "particles_1.vtk")

        # out-of-bounds slicing throws
        @test_throws ErrorException DataSetModule.DataSet(dir; start_idx=3)
    finally
        rm(dir; force=true, recursive=true)
    end
end
