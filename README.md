# VelocityFields.jl #

VelocityFields.jl is a Julia package for generating and analysing averaged velocity fields from particle datasets. It provides high-level routines to transform raw 3D point and velocity data into either projected planar or cylindrical coordinates, bin the data points and generate a 2D field. The resulting `Field` object can be exported to and reconstructed from CSV, and its vorticity automatically calculated. For visualisation, it offers a custom built-in quiver-overlaid heatmap generator with fine control over arrowheads and colorbar ticks. [Packing3D.jl](https://github.com/fjbarter/Packing3D.jl) is used throughout for mesh bounds, data splitting and I/O.

![Example Image1](https://github.com/fjbarter/VelocityFields.jl/blob/main/source/converging_densification_segregation_field.png?raw=true)
Above image: Time- and azimuthally averaged velocity field for the net motion of 50000 bidisperse particles throughout 1000 horizontal taps. The process shown exhibits strong segregating densification. Fields for the small (a) and large (b) particles are shown separately. Velocity was computed from the net displacement of each particle. Field is mirrored about 𝑟 = 0 to better visually represent the vessel, as the field is azimuthally averaged about 𝜃 ∈ [0,2𝜋].

---



**Public API Overview:**  
The package exposes a clean, high-level interface designed for users (e.g., Master’s students) who may not be familiar with the lower-level Julia intricacies. All functions automatically dispatch to the appropriate routines based on the provided geometry type.

The key public functions and types are:

- **Field Generation & Analysis:**  
  - `generate_field`  
  - `compute_vorticity`

- **I/O:**  
  - `field_to_csv`  
  - `csv_to_field`

- **Visualisation:**  
  - `plot_field`

- **Geometry Types:**  
  - `Plane`  
  - `Cylinder`

- **Core Type:**  
  - `Field`

> **Note:** Some internal docstrings may be outdated. The documentation below reflects the current intended usage.

## Table of Contents

- [Public API Overview](#public-api-overview)  
- [Key Functions](#key-functions)  
  - [`generate_field`](#generate_field)  
  - [`compute_vorticity`](#compute_vorticity)  
  - [`plot_field`](#plot_field)  
  - [`field_to_csv`](#field_to_csv)  
  - [`csv_to_field`](#csv_to_field)  
- [Data Structures](#data-structures)  
  - [`Field`](#field)  
  - [`Plane` & `Cylinder`](#plane--cylinder)  
- [How It Works](#how-it-works)  
- [Requirements](#requirements)  
- [Installation](#installation)  
- [Examples](#examples)  
- [Limitations](#limitations)  
- [Planned Features](#planned-features)  
- [License](#license)  
- [Contact](#contact)  

## Public API Overview

VelocityFields.jl is built around a handful of high-level routines that abstract away the details of file I/O, coordinate transformations, parallel processing, and plotting. Whether you need a planar projection or a cylindrical average, a single dispatch call is all that’s required.

## Key Functions


### `generate_field`

**Description:**  
Processes all VTK files in a directory to produce a 2D averaged velocity field. Supports planar or cylindrical geometries, optional data splitting, long-average (displacement-based) mode, and automatic bin-size estimation.

**Signature:**  
```julia
generate_field(
    dataset_dir::String,
    geom; 
    bin_size::Union{Float64,Nothing}=nothing,
    start_idx::Union{Nothing,Int}=nothing,
    end_idx::Union{Nothing,Int}=nothing,
    split_by::Union{Symbol,Nothing}=nothing,
    threshold::Union{<:Real,Nothing}=nothing,
    split::Union{Int,Nothing}=nothing,
    long_average::Union{Bool,Nothing}=nothing,
    timestep::Union{<:Real,Nothing}=nothing,
    vector_type::Union{Symbol,Nothing}=nothing
) -> Field
```

**Arguments:**

- `dataset_dir` (positional): Path to folder with `particles_*.vtk` files.  
- `geom` (positional): Either `Plane(...)` or `Cylinder(...)`.  
- `bin_size` (kw): Size of each bin; if `nothing`, determined from data span.  
- `start_idx`, `end_idx` (kw): Slice indices for file selection.  
- `split_by`, `threshold`, `split` (kw): Criteria for partitioning points.  
- `long_average`, `timestep` (kw): Enable displacement-based averaging.  
- `vector_type` (kw): Symbol for velocity field (default `:v`).

**Returns:**  
A `Field` instance containing the averaged field, origin, bin size, and geometry type.

---

### `plot_field`

**Description:**  
Generates a heatmap of speed with a quiver overlay. Supports fine control of arrow length, colorbar ticks, and output filename.

**Signature:**  
```julia
plot_field(
    field::Field;
    arrow_length::Union{Float64,Nothing}=nothing,
    figure_name::Union{String,Nothing}=nothing,
    cbar_max::Union{<:Real,Nothing}=nothing
) -> nothing
```

**Arguments:**

- `field` (positional): A `Field` instance.  
- `arrow_length` (kw): Fixed arrow length (default 0.7×bin_size).  
- `figure_name` (kw): Output PNG filename (default `"velocity_field.png"`).  
- `cbar_max` (kw): Max colorbar value (default = data max).
---

### `compute_vorticity`

**Description:**  
Computes the mean absolute 2D vorticity of a `Field` using central differences on the interior grid.

**Signature:**  
```julia
compute_vorticity(field::Field) -> Float64
```

**Arguments:**

- `field` (positional): A `Field` instance from `generate_field` or `csv_to_field`.

**Returns:**  
Mean absolute vorticity (units $s^{-1}$).

---

### `field_to_csv`

**Description:**  
Exports a `Field` to CSV with metadata headers. The CSV includes columns `x,y,u,v`.

**Signature:**  
```julia
field_to_csv(field::Field, filepath::String) -> nothing
```

**Arguments:**

- `field` (positional): A `Field` instance.  
- `filepath` (positional): Path to write CSV.

---

### `csv_to_field`

**Description:**  
Reconstructs a `Field` from a CSV produced by `field_to_csv`, reading metadata and rebuilding the grid.

**Signature:**  
```julia
csv_to_field(filepath::String) -> Field
```

**Arguments:**

- `filepath` (positional): Path to a CSV with metadata and `x,y,u,v`.

---

## Data Structures

### `Field`

A struct holding

- `avg_field::Array{Float64,3}` — Averaged velocity array `[nₓ, nᵧ, 2]`.  
- `origin::Tuple{Float64,Float64}` — Lower-left `(x_min,y_min)`.  
- `bin_size::Float64` — Grid spacing.  
- `geometry_type::Symbol` — `:plane` or `:cylindrical`.

### `Plane` & `Cylinder`

Construct coordinate transforms:

- **`Plane(point::Vector{3}, normal::Vector{3})`**: defines a local 2D basis.  
- **`Cylinder(base::Vector{3}, axis::Vector{3})`**: defines radial/axial coordinates.

## How It Works

1. **File Discovery:** Scan for `particles_*.vtk` via `DataSetModule`.  
2. **Transformation:** Use `Geometry.transform_file_data` to rotate & project points and velocities.  
3. **Binning:** Compute provisional bins via `process_file_helper` in parallel, then merge.  
4. **Post-Processing:** Build dense `avg_field`, compute origin & dimensions.  
5. **Analysis & I/O:** Compute vorticity or export/reimport via CSV.  
6. **Visualisation:** Heatmap + custom quiver with `plot_field`.

## Requirements

- Julia 1.6 or later  
- [Packing3D.jl](https://github.com/fjbarter/Packing3D.jl) for mesh bounds & I/O  
- CSV, DataFrames, Distributed

## Installation

Not yet registered. Install via:

```julia
using Pkg
Pkg.develop(url="https://github.com/fjbarter/VelocityFields.jl")
```

## Examples

See the `examples/` directory for:

- Planar velocity field generation & plotting  
- Cylindrical field generation with mirrored r  
- CSV export/import and vorticity calculation

## Limitations

- **Data Format:** Only legacy ASCII VTK via Packing3D.  
- **Geometry:** Only planar & cylindrical supported.  
- **Performance:** Depends on `pmap` scalability and file I/O.

## Planned Features

- Support for divergence computation.  
- Extended geometries (spherical, arbitrary surfaces).  
- Interactive plotting backend integration.

## License

MIT License. See [LICENSE](LICENSE).

## Contact

**Your Name**  
Email: [fjbarter@outlook.com](mailto:fjbarter@outlook.com)  
GitHub: [fjbarter/VelocityFields.jl](https://github.com/fjbarter/VelocityFields.jl)