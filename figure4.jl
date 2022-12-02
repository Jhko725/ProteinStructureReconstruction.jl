##
using Revise
using GLMakie
import Pkg
Pkg.activate(".")

import HDF5
using FixedPointNumbers
using Images
using ImageBinarization
import Statistics: mean
using Interpolations
using MarchingCubes
using Meshes
using MeshViz

include("utils.jl")
include("transforms.jl")
using .Transforms

struct ZStack{T<:Number}
    data::AbstractArray{T,4}
    channels
    pixel_sizes
    dim_order::String
end

function to_zstack(file::String)
    @assert ispath(file) "Given filepath does not exist, or is invalid"
    ext = splitext(file)
    if ext[end] == ".h5"
        return HDF5.h5open(file, "r") |> to_zstack
    end
    return nothing
end

function to_zstack(file::HDF5.File)
    group = file["Data"]
    data = read(group) |> invertdims |> to_float32
    pixel_sizes = HDF5.read_attribute(group, "pixel_sizes")
    channels = HDF5.read_attribute(group, "channels")
    dim_order = HDF5.read_attribute(group, "dim_order")

    return ZStack(data, channels, pixel_sizes, dim_order)
end

function to_float32(array::AbstractArray{T}) where {T<:Unsigned}
    fixedptarray = reinterpret(Normed{T,8 * sizeof(T)}, array)
    return float32.(fixedptarray)
end

function to_float32(array::AbstractArray{T}) where {T<:AbstractFloat}
    return float32.(array)
end

function to_float32(array::AbstractArray{Float32})
    return array
end

function interpolate_zstack(zstack, scale)
    scale_z, scale_y, scale_x = scale / minimum(scale)
    nz, ny, nx = size(zstack)
    grids = ((0:nz-1) * scale_z, (0:ny-1) * scale_y, (0:nx-1) * scale_x)
    algorithm = BSpline(Linear())
    itp = interpolate(zstack, algorithm)
    sitp = Interpolations.scale(itp, grids...)
    return sitp(0:(nz-1)*scale_z, 0:(ny-1)*scale_y, 0:(nx-1)*scale_x)
end

const filepath = "./Data/SIM2/Image2_SIM2.h5"
const fid = HDF5.h5open(filepath, "r")
const sim_actinin = to_zstack("./Data/SIM2/Image2_SIM2.h5")
##
let
    function filter_components(binarized_volume::AbstractArray{T}, threshold::Real) where {T}
        labels = label_components(binarized_volume)
        sizes = component_lengths(labels)
        inds = component_indices(labels)
        for (size, ind) in zip(sizes, inds)
            if size < threshold
                binarized_volume[ind] .= zero(T)
            end
        end
        return binarized_volume
    end
    roi = sim_actinin.data[:, 1:7, 100:500, 1500:1900]

    desmin_interp = interpolate_zstack(roi[1, :, :, :], sim_actinin.pixel_sizes)
    actinin_interp = interpolate_zstack(roi[2, :, :, :], sim_actinin.pixel_sizes)

    desmin_bin = mapslices(slice -> binarize(slice, Sauvola()), desmin_interp, dims=[2, 3])
    actinin_bin = mapslices(slice -> binarize(slice, Sauvola()), actinin_interp, dims=[2, 3])

    desmin_mc = MC(permutedims(filter_components(desmin_bin, 2000), [3, 2, 1]))
    actinin_mc = MC(permutedims(filter_components(actinin_bin, 2000), [3, 2, 1]))

    march(desmin_mc, 0.5)
    mesh_desmin = MarchingCubes.makemesh(Meshes, desmin_mc)

    march(actinin_mc, 0.5)
    mesh_actinin = MarchingCubes.makemesh(Meshes, actinin_mc)

    fig = Figure(resolution=(1300, 500))
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    gl3 = fig[1, 3] = GridLayout()

    axis3_kwargs = Dict(:aspect => reverse(size(actinin_interp)), :elevation => 0.25π, :azimuth => -0.35π, :xlabel => "x (nm)", :ylabel => "y (nm)", :zlabel => "z (nm)")
    ax1 = Axis3(gl1[1, 1]; axis3_kwargs...)
    viz!(mesh_desmin, color=1:nvertices(mesh_desmin), colorscheme=:Reds)
    fig
    ax2 = Axis3(gl2[1, 1]; axis3_kwargs...)
    viz!(mesh_actinin, color=1:nvertices(mesh_actinin), colorscheme=:Greens)
    ax3 = Axis3(gl3[1, 1]; axis3_kwargs...)
    viz!(mesh_desmin, color=1:nvertices(mesh_desmin), colorscheme=:Reds)
    viz!(mesh_actinin, color=1:nvertices(mesh_actinin), colorscheme=:Greens)

    for ax_ in [ax1, ax2, ax3]
        ax_.zticks = 0:30:reverse(size(actinin_interp))[end]
    end
    fig
    #save("./Figures/Figure 4c.png", fig)
end