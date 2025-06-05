module DataSetModule

using Packing3D

export DataSet, process_file

"""
    find_files(dir; pattern=nothing)

Scan `dir` for files matching `pattern`)
and return a sorted vector of full paths.
"""
function find_files(dir::String; pattern::Union{Regex,Nothing}=nothing)::Vector{String}
    if !isdir(dir)
        error("DataSet: Provided directory '$dir' does not exist!")
    end
    pattern === nothing && (pattern = r"^particles_\d+\.vtk$")
    files = [ joinpath(dir,f) for f in readdir(dir) if occursin(pattern, f) ]
    isempty(files) && error("DataSet: No VTK files matching 'particles_*.vtk' found in '$dir'")
    return sort(files)
end

"""
    process_file(file)

Load and return the contents of a single VTK file (via `Packing3D.read_vtk_file`),
or return `nothing` (with a warning) on error.
"""
function process_file(file::String)
    try
        return read_vtk_file(file)
    catch err
        @warn "DataSet: Failed to process file '$file'" exception=err
        return nothing
    end
end

"""
    DataSet(dir; start_idx=nothing, end_idx=nothing)

Construct a dataset by scanning `dir` for `particles_*.vtk` files,
then slicing that list:

- If neither `start_idx` nor `end_idx` are provided, includes *all* files.
- If only `start_idx` is given, includes from `start_idx` through the end.
- If only `end_idx` is given, includes from 1 through `end_idx`.
- If both are given, includes `start_idx:end_idx`.

Throws an error if the resulting slice is invalid (out of bounds or empty).
"""
struct DataSet
    dir::String
    files::Vector{String}
    pattern::Union{Nothing, Regex}

    function DataSet(dir::String;
                     start_idx::Union{Nothing,Int}=nothing,
                     end_idx::Union{Nothing,Int}=nothing,
                     pattern::Union{Regex,Nothing}=nothing)
                     
        # validate directory
        if !isdir(dir)
            error("DataSet: Provided directory '$dir' does not exist!")
        end

        # get the full, sorted list
        all_files = find_files(dir; pattern=pattern)

        # determine slice bounds
        i1 = isnothing(start_idx) ? 1              : start_idx
        i2 = isnothing(end_idx)   ? length(all_files) : end_idx

        # validate bounds
        if i1 < 1 || i2 > length(all_files) || i1 > i2
            error("DataSet: Invalid slice indices (start=$i1, end=$i2) for $(length(all_files)) files")
        end

        # slice and construct
        files = all_files[i1:i2]
        new(dir, files)
    end
end

end # module DataSetModule
