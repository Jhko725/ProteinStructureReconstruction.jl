##
using Revise
using GLMakie
import Pkg
Pkg.activate(".")
##
import HDF5
include("utils.jl")

filepath = "./Data/SIM2/Image2_SIM2.h5"
fid = HDF5.h5open(filepath, "r")
##
read(fid["Data"]) |> typeof
##
HDF5.read_attribute(fid["Data"], "pixel_sizes")
##
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
##
using FixedPointNumbers
using Images
out = to_zstack("./Data/SIM2/Image2_SIM2.h5")

##
let
    fig = Figure(resolution=(800, 800))
    ax = GLMakie.Axis(fig[1, 1], yscale=Makie.pseudolog10, yminorticksvisible=true, yminorgridvisible=true,
        yminorticks=IntervalsBetween(8))
    hist!(reshape(to_float32(out.data)[1, :, :, :], :), bins=100)
    fig
end
#hist(reshape(to_float32(out.data)[:, :, :, 1], :), bins = 100)
##
let
    import Statistics: mean
    overlay = mean(out.data, dims=2)
    fig = Figure(resolution=(800, 800))
    ax = GLMakie.Axis(fig[1, 1])
    image!(ax, overlay[1, 1, :, :], colormap=:Reds, tranparency=true)
    image!(ax, overlay[2, 1, :, :], colormap=:Greens, tranparency=true)
    fig
end
##
