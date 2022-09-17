##
using Revise
using GLMakie
import Pkg
Pkg.activate(".")

import HDF5
include("utils.jl")

filepath = "./Data/SIM2/Image2_SIM2.h5"
fid = HDF5.h5open(filepath, "r")
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
    hist!(reshape(to_float32(out.data)[1, 1:20, 100:500, 1500:1900], :), bins=100)
    fig
end
#hist(reshape(to_float32(out.data)[:, :, :, 1], :), bins = 100)
##
let
    import Statistics: mean
    using Images
    overlay = mean(out.data, dims=2)
    fig = Figure(resolution=(800, 800))
    ax = GLMakie.Axis(fig[1, 1])
    scaler = scaleminmax(0.0f0, 1.0f0)
    image!(ax, scaler.(overlay[1, 1, :, :]), colormap=cgrad(:Reds, scale=:exp, alpha=0.5))
    image!(ax, scaler.(overlay[2, 1, :, :]), colormap=cgrad(:Greens, scale=:exp, alpha=0.5))
    fig
end
##
let
    let
        import Statistics: mean
        using Images
        overlay = mean(out.data, dims=2)
        fig = Figure(resolution=(1600, 800))
        ax1, img1 = heatmap(fig[1, 1][1, 1], overlay[1, 1, 100:500, 1500:1900], colormap=cgrad(:Reds, scale=x -> x^2), colorrange=(0.001, 0.15), lowclip=:white, highclip=:red)
        Colorbar(fig[1, 1][1, 2], img1)
        ax2, img2 = heatmap(fig[1, 2][1, 1], overlay[2, 1, 100:500, 1500:1900], colormap=cgrad(:Greens, scale=x -> x^2), colorrange=(0.001, 0.15), lowclip=:white, highclip=:green)
        Colorbar(fig[1, 2][1, 2], img2)
        fig
    end
end
##
let
    using Images
    fig = Figure(resolution=(1600, 800))
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    indices = range(1, 45, length=25) |> collect |> x -> round.(Integer, x)
    for i in 1:5
        for j in 1:5
            ax1 = GLMakie.Axis(gl1[i, j], aspect=1)
            ax2 = GLMakie.Axis(gl2[i, j], aspect=1)
            foreach(hidedecorations!, (ax1, ax2))
            #foreach(hidespines!, (ax1, ax2))

            ind = indices[5(i-1)+j]
            heatmap!(ax1, out.data[1, ind, 100:500, 1500:1900], colormap=cgrad(:Reds, scale=x -> x^2), colorrange=(0.001, 0.15), lowclip=:white, highclip=:red)
            heatmap!(ax2, out.data[2, ind, 100:500, 1500:1900], colormap=cgrad(:Greens, scale=x -> x^2), colorrange=(0.001, 0.15), lowclip=:white, highclip=:green)

        end
    end
    fig
end
##
let
    include("transforms.jl")
    using .Transforms
    import Statistics: mean
    overlay = mean(out.data, dims=2)
    fig = Figure(resolution=(1600, 800))
    cmaps = [:Reds, :Greens]
    for (i, cmap) in enumerate(cmaps)
        intensity, orientation = LineFilterTransform(overlay[i, 1, 100:500, 1500:1900], 10, 5, 20)
        ax, img = heatmap(fig[1, i][1, 1], intensity, colorrange=(0.0, maximum(intensity)), colormap=cmap)
        Colorbar(fig[1, i][1, 2], img)
    end

    fig
end
##
let
    include("transforms.jl")
    using .Transforms
    import Statistics: mean
    overlay = mean(out.data, dims=2)
    fig = Figure(resolution=(1600, 800))
    cmaps = [:Reds, :Greens]
    for (i, cmap) in enumerate(cmaps)
        intensity, orientation = LineFilterTransform(overlay[i, 1, 100:500, 1500:1900], 20, 5, 40)
        oft = Transforms.OrientationFilterTransform(intensity, orientation, 20, 5, 40)
        ax, img = heatmap(fig[1, i][1, 1], oft, colorrange=(0.0, maximum(oft)), colormap=cmap)
        Colorbar(fig[1, i][1, 2], img)
    end

    fig
end
##
let
    using Images
    fig = Figure(resolution=(1600, 800))
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    indices = range(1, 45, length=25) |> collect |> x -> round.(Integer, x)
    cmaps = [:Reds, :Greens]
    for i in 1:5
        for j in 1:5
            ax1 = GLMakie.Axis(gl1[i, j], aspect=1)
            ax2 = GLMakie.Axis(gl2[i, j], aspect=1)
            foreach(hidedecorations!, (ax1, ax2))
            #foreach(hidespines!, (ax1, ax2))

            ind = indices[5(i-1)+j]
            lft1_i, lft1_o = LineFilterTransform(out.data[1, ind, 100:500, 1500:1900], 20, 15, 40)
            oft1 = Transforms.OrientationFilterTransform(lft1_i, lft1_o, 20, 5, 40)
            lft2_i, lft2_o = LineFilterTransform(out.data[2, ind, 100:500, 1500:1900], 20, 15, 40)
            oft2 = Transforms.OrientationFilterTransform(lft2_i, lft2_o, 20, 5, 40)
            heatmap!(ax1, oft1, colormap=:Reds, colorrange=(0.0, 0.8 * maximum(oft1)))
            heatmap!(ax2, oft2, colormap=:Greens, colorrange=(0.0, 0.8 * maximum(oft2)))
        end
    end
    fig
end
##
using Interpolations

function interpolate_zstack(zstack, scale)
    scale_z, scale_y, scale_x = scale / minimum(scale)
    nz, ny, nx = size(zstack)
    grids = ((0:nz-1) * scale_z, (0:ny-1) * scale_y, (0:nx-1) * scale_x)
    algorithm = BSpline(Linear())
    itp = interpolate(zstack, algorithm)
    sitp = Interpolations.scale(itp, grids...)
    return sitp(0:(nz-1)*scale_z, 0:(ny-1)*scale_y, 0:(nx-1)*scale_x)
end


##
let
    fig = Figure(resolution=(800, 800))
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1))
    interped = interpolate_zstack(out.data[2, 1:10, 250:750, 1000:1500], out.pixel_sizes)
    volume!(interped, algorithm=:mip)
    fig
end
##
using MarchingCubes
using Meshes
using MeshViz
let
    interped = interpolate_zstack(out.data[1, 1:20, 100:500, 1500:1900], out.pixel_sizes)
    mc = MC(interped)
    march(mc, 0.001)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    screen = viz(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    march(mc, 0.05)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    viz!(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    display(screen)
end
##
include("transforms.jl")
using .Transforms
interped = interpolate_zstack(out.data[1, 1:20, 100:500, 1500:1900], out.pixel_sizes)
lft1_i, lft1_o = Transforms.LineFilterTransform(interped, 20, 15, 40)
##
begin
    fig = Figure(resolution=(800, 800))
    ax = Axis3(fig[1, 1], aspect=(1, 1, 1))

    volume!(lft1_i, algorithm=:mip)
    fig
end
##
begin
    mc = MC(lft1_i)
    march(mc, 1.0)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    screen = viz(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    march(mc, 2.0)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    viz!(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    display(screen)
end
##
oft1 = Transforms.OrientationFilterTransform(lft1_i, lft1_o, 20, 5, 40)
##
begin
    mc = MC(oft1)
    march(mc, 5.0)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    screen = viz(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    march(mc, 7.0)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    viz!(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    display(screen)
end