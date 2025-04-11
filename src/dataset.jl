# DataSet.jl
# This file defines the struct used for processing a particle dataset
# stored as multiple VTK files in a directory, without loading all data into memory.

module DataSetModule

using Packing3D

export DataSet, process_directory, process_file

# ---
# find_files
#
# Scans the provided directory for files matching the pattern
# "particles_*.vtk" (standard LIGGGHTS output) and returns a vector of
# full file paths.
# ---
function find_files(dir::String; pattern::Union{Regex, Nothing}=nothing)::Vector{String}
    # Check that the directory exists.
    if !isdir(dir)
        error("DataSet: Provided directory '$dir' does not exist!")
    end

    if isnothing(pattern)
        # Regex for particles_*.vtk (standard LIGGGHTS output)
        pattern = r"^particles_\d+\.vtk$"
    end

    found_files = String[]

    for file in readdir(dir)
        if occursin(pattern, file)
            push!(found_files, joinpath(dir, file))
        end
    end

    # Gracefully check if no matching files were found.
    if isempty(found_files)
        error("DataSet: No VTK files matching 'particles_*.vtk' found in directory '$dir'")
    end

    # return found_files
    return sort(found_files)
end

# ---
# process_file
#
# Loads and processes a single VTK file from the dataset.
# This function demonstrates how you might load the data, perform
# analysis, and then let the data go out of scope to free memory.
#
# For now, it simply loads the file using read_vtk_file from Packing3D,
# but you can expand it to include your analysis workflow.
# ---
function process_file(file::String)
    try
        data = read_vtk_file(file)
        # Add your processing logic here; for example:
        # extract coordinates, compute averages, etc.
        # For demonstration, we'll just return the loaded data.
        return data
    catch err
        @warn "DataSet: Failed to process file '$file'" exception=err
        return nothing
    end
end

# ---
# DataSet Struct
#
# A minimal dataset struct that stores the directory and the list of
# matching VTK file paths (according to the standard LIGGGHTS naming).
#
# The inner constructor processes the directory and stores the file paths.
# ---
struct DataSet
    dir::String           # Provided directory path.
    files::Vector{String} # List of VTK file paths that match the pattern.

    # Inner constructor for enhanced control and validation.
    function DataSet(dir::String)
        if !isdir(dir)
            error("DataSet: Provided directory '$dir' does not exist!")
        end
        files = find_files(dir)
        new(dir, files)
    end
end

end # module DataSetModule
