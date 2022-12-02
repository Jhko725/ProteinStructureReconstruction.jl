##
using Revise
using WGLMakie
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

out = to_zstack("./Data/SIM2/Image2_SIM2.h5")
size(out.data)[2:4] .* out.pixel_sizes
roi = (1:1000, 1001:2000)
##
let
    fig = Figure(resolution=(800, 800))
    ax = WGLMakie.Axis(fig[1, 1], yscale=Makie.pseudolog10, yminorticksvisible=true, yminorgridvisible=true,
        yminorticks=IntervalsBetween(8))
    hist!(reshape(to_float32(out.data)[1, 1:20, 100:500, 1500:1900], :), bins=100)
    fig
end
#hist(reshape(to_float32(out.data)[:, :, :, 1], :), bins = 100)
##
let
    overlay = mean(out.data, dims=2)
    fig = Figure(resolution=(600, 600))
    ax = WGLMakie.Axis(fig[1, 1])
    scaler = scaleminmax(0.0f0, 1.0f0)
    image!(ax, scaler.(overlay[1, 1, :, :]), colormap=cgrad(:Reds, scale=:exp, alpha=0.7))
    image!(ax, scaler.(overlay[2, 1, :, :]), colormap=cgrad(:Greens, scale=:exp, alpha=0.7))
    fig
end
##
#100:500, 1500:1900

let
    let
        overlay = mean(out.data, dims=2)
        fig = Figure(resolution=(800, 400))
        ax1, img1 = heatmap(fig[1, 1][1, 1], overlay[1, 1, roi...], colormap=cgrad(:Reds, scale=x -> x^2), colorrange=(0.001, 0.15), lowclip=:white, highclip=:red, aspect=1)
        Colorbar(fig[1, 1][1, 2], img1)
        ax2, img2 = heatmap(fig[1, 2][1, 1], overlay[2, 1, roi...], colormap=cgrad(:Greens, scale=x -> x^2), colorrange=(0.001, 0.15), lowclip=:white, highclip=:green, aspect=1)
        Colorbar(fig[1, 2][1, 2], img2)
        fig
    end
end
##
let
    fig = Figure(resolution=(800, 400))
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    indices = range(1, 45, length=25) |> collect |> x -> round.(Integer, x)
    for i in 1:5
        for j in 1:5
            ax1 = WGLMakie.Axis(gl1[i, j], aspect=1)
            ax2 = WGLMakie.Axis(gl2[i, j], aspect=1)
            foreach(hidedecorations!, (ax1, ax2))
            #foreach(hidespines!, (ax1, ax2))

            ind = indices[5(i-1)+j]
            heatmap!(ax1, out.data[1, ind, roi...], colormap=cgrad(:Reds, scale=x -> x^2), colorrange=(0.001, 0.15), lowclip=:white, highclip=:red)
            heatmap!(ax2, out.data[2, ind, roi...], colormap=cgrad(:Greens, scale=x -> x^2), colorrange=(0.001, 0.15), lowclip=:white, highclip=:green)

        end
    end
    fig
end
##
let
    overlay = mean(out.data, dims=2)
    fig = Figure(resolution=(800, 400))
    cmaps = [:Reds, :Greens]
    for (i, cmap) in enumerate(cmaps)
        intensity, orientation = LineFilterTransform(overlay[i, 1, roi...], 10, 5, 20)
        ax, img = heatmap(fig[1, i][1, 1], intensity, colorrange=(0.0, maximum(intensity)), colormap=cmap)
        Colorbar(fig[1, i][1, 2], img)
    end

    fig
end
##
let
    overlay = mean(out.data, dims=2)
    fig = Figure(resolution=(1600, 800))
    cmaps = [:Reds, :Greens]
    for (i, cmap) in enumerate(cmaps)
        intensity, orientation = LineFilterTransform(overlay[i, 1, 100:500, 1500:1900], 10, 5, 20)
        oft = Transforms.OrientationFilterTransform(intensity, orientation, 10, 5, 20)
        ax, img = heatmap(fig[1, i][1, 1], oft, colorrange=(0.0, maximum(oft)), colormap=cmap)
        Colorbar(fig[1, i][1, 2], img)
    end

    fig
end
##
let
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
            heatmap!(ax1, out.data[1, ind, 100:500, 1500:1900], colormap=:Reds, colorrange=(0.0, 0.8 * maximum(out.data[1, ind, 100:500, 1500:1900])))
            heatmap!(ax2, out.data[2, ind, 100:500, 1500:1900], colormap=:Greens, colorrange=(0.0, 0.8 * maximum(out.data[2, ind, 100:500, 1500:1900])))

        end
    end
    fig
end
##
let
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
            heatmap!(ax1, binarize(out.data[1, ind, 100:500, 1500:1900], Sauvola()), colormap=:Reds)
            heatmap!(ax2, binarize(out.data[2, ind, 100:500, 1500:1900], Sauvola()), colormap=:Greens)

        end
    end
    fig
end
##
let
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
            lft1_i, lft1_o = LineFilterTransform(out.data[1, ind, 100:500, 1500:1900], 10, 10, 40, interp_scheme=BSpline(Linear()))
            lft2_i, lft2_o = LineFilterTransform(out.data[2, ind, 100:500, 1500:1900], 10, 10, 40, interp_scheme=BSpline(Linear()))
            heatmap!(ax1, lft1_i, colormap=:Reds, colorrange=(0.0, 0.8 * maximum(lft1_i)))
            heatmap!(ax2, lft2_i, colormap=:Greens, colorrange=(0.0, 0.8 * maximum(lft2_i)))
        end
    end
    fig
end
##
let
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
            lft1_i, lft1_o = LineFilterTransform(out.data[1, ind, 100:500, 1500:1900], 20, 5, 20, interp_scheme=BSpline(Linear()))
            oft1 = Transforms.OrientationFilterTransform(lft1_i, lft1_o, 20, 5, 20, BSpline(Linear()))
            lft2_i, lft2_o = LineFilterTransform(out.data[2, ind, 100:500, 1500:1900], 20, 5, 20, interp_scheme=BSpline(Linear()))
            oft2 = Transforms.OrientationFilterTransform(lft2_i, lft2_o, 20, 5, 20, BSpline(Linear()))
            heatmap!(ax1, oft1, colormap=:Reds)
            heatmap!(ax2, oft2, colormap=:Greens)
        end
    end
    fig
end
##
let
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
            lft1_i, lft1_o = LineFilterTransform(out.data[1, ind, 100:500, 1500:1900], 20, 20, 30, interp_scheme=BSpline(Linear()))
            oft1 = Transforms.OrientationFilterTransform(lft1_i, lft1_o, 20, 20, 30, BSpline(Linear()))
            lft2_i, lft2_o = LineFilterTransform(out.data[2, ind, 100:500, 1500:1900], 20, 20, 30, interp_scheme=BSpline(Linear()))
            oft2 = Transforms.OrientationFilterTransform(lft2_i, lft2_o, 20, 20, 30, BSpline(Linear()))
            heatmap!(ax1, binarize(oft1, Otsu()), colormap=:Reds)
            heatmap!(ax2, binarize(oft2, Otsu()), colormap=:Greens)
        end
    end
    fig
end
##

let
    fig = Figure(resolution=(600, 600))
    ax = Axis3(fig[1, 1], aspect=(50, 500, 500))
    interped = interpolate_zstack(out.data[1, :, roi...], out.pixel_sizes)
    volume!(interped, algorithm=:mip)
    fig
end
##
let
    fig = Figure(resolution=(800, 800))
    ax = Axis3(fig[1, 1], aspect=(50, 500, 500))
    interped = interpolate_zstack(out.data[1, 1:5, 250:750, 1000:1500], out.pixel_sizes)
    volume!(interped, algorithm=:mip)
    fig
end
##
let
    fig = Figure(resolution=(800, 800))
    ax = GLMakie.Axis(fig[1, 1], aspect=1)
    interped = interpolate_zstack(out.data[1, 1:5, 250:750, 1000:1500], out.pixel_sizes)
    heatmap!(interped[5, :, :])
    fig
end
##

let
    interped = interpolate_zstack(out.data[1, 1:20, 100:500, 1500:1900], out.pixel_sizes)
    mc = MC(interped)
    march(mc, 0.001)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    screen = viz(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    march(mc, 0.05)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    print(mesh)
    #viz!(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    #display(screen)
end
##
let
    function process_slice(slice, N::Integer, r::Integer, Nr::Integer)
        #lft_i, lft_o = LineFilterTransform(slice, N, r, Nr, interp_scheme=BSpline(Linear()))
        #oft = OrientationFilterTransform(lft_i, lft_o, N, r, Nr, BSpline(Linear()))
        #binarized = binarize(oft, Otsu())
        binarized = binarize(slice, Sauvola())
        return binarized
    end

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

    desmin_interp = interpolate_zstack(out.data[1, 1:7, 100:500, 1500:1900], out.pixel_sizes)
    actinin_interp = interpolate_zstack(out.data[2, 1:7, 100:500, 1500:1900], out.pixel_sizes)

    desmin_bin = mapslices(x -> process_slice(x, 20, 20, 40), desmin_interp, dims=[2, 3])
    actinin_bin = mapslices(x -> process_slice(x, 20, 20, 40), actinin_interp, dims=[2, 3])

    labels_desmin = label_components(filter_components(desmin_bin, 50))
    labels_actinin = label_components(filter_components(actinin_bin, 50))

    desmin_sizes = component_lengths(labels_desmin)
    actinin_sizes = component_lengths(labels_actinin)

    fig = Figure(resolution=(1400, 700))
    ax = GLMakie.Axis(fig[1, 1], yscale=Makie.pseudolog10, yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(8))
    ax2 = GLMakie.Axis(fig[1, 2], yscale=Makie.pseudolog10, yminorticksvisible=true, yminorgridvisible=true, yminorticks=IntervalsBetween(8))
    hist!(ax, log10.(desmin_sizes[2:end]), bins=100)
    hist!(ax2, log10.(actinin_sizes[2:end]), bins=100)
    fig
end
##
function make_mesh(zstack, pixel_sizes, isovalue)
    interped = interpolate_zstack(zstack, pixel_sizes)
    mc = MC(permutedims(interped, [3, 2, 1]))
    march(mc, isovalue)
    return MarchingCubes.makemesh(Meshes, mc)
end

let
    function process_slice(slice, N::Integer, r::Integer, Nr::Integer)
        #lft_i, lft_o = LineFilterTransform(slice, N, r, Nr, interp_scheme=BSpline(Linear()))
        #oft = OrientationFilterTransform(lft_i, lft_o, N, r, Nr, BSpline(Linear()))
        #binarized = binarize(oft, Otsu())
        binarized = binarize(slice, Sauvola())
        return binarized
    end

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

    desmin_interp = interpolate_zstack(out.data[1, 1:7, 100:500, 1500:1900], out.pixel_sizes)
    actinin_interp = interpolate_zstack(out.data[2, 1:7, 100:500, 1500:1900], out.pixel_sizes)

    desmin_bin = mapslices(x -> process_slice(x, 20, 20, 40), desmin_interp, dims=[2, 3])
    actinin_bin = mapslices(x -> process_slice(x, 20, 20, 40), actinin_interp, dims=[2, 3])

    desmin_mc = MC(permutedims(filter_components(desmin_bin, 200), [3, 2, 1]))
    actinin_mc = MC(permutedims(filter_components(actinin_bin, 200), [3, 2, 1]))

    march(desmin_mc, 0.5)
    mesh_desmin = MarchingCubes.makemesh(Meshes, desmin_mc)

    march(actinin_mc, 0.5)
    mesh_actinin = MarchingCubes.makemesh(Meshes, actinin_mc)

    fig = Figure(resolution=(1400, 700))
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    gl3 = fig[1, 3] = GridLayout()

    axis3_kwargs = Dict(:aspect => reverse(size(actinin_interp)), :elevation => 0.25π, :azimuth => -0.35π)
    ax1 = Axis3(gl1[1, 1]; axis3_kwargs...)
    viz!(mesh_desmin, color=1:nvertices(mesh_desmin), colorscheme=:Reds)
    fig
    ax2 = Axis3(gl2[1, 1]; axis3_kwargs...)
    viz!(mesh_actinin, color=1:nvertices(mesh_actinin), colorscheme=:Greens)
    ax3 = Axis3(gl3[1, 1]; axis3_kwargs...)
    viz!(mesh_desmin, color=1:nvertices(mesh_desmin), colorscheme=:Reds)
    viz!(mesh_actinin, color=1:nvertices(mesh_actinin), colorscheme=:Greens)

    for ax_ in [ax1, ax2, ax3]
        ax_.zticks = 0:20:reverse(size(actinin_interp))[end]
    end
    fig
end
##
let
    desmin_interp = interpolate_zstack(out.data[1, 5:15, 100:500, 1500:1900], out.pixel_sizes)
    desmin_bin = mapslices(x -> binarize(x, Sauvola()), desmin_interp, dims=[2, 3])
    desmin_mc = MC(permutedims(desmin_bin, [3, 2, 1]))
    march(desmin_mc, 1.0)
    mesh_desmin = MarchingCubes.makemesh(Meshes, desmin_mc)

    actinin_interp = interpolate_zstack(out.data[2, 5:15, 100:500, 1500:1900], out.pixel_sizes)
    actinin_bin = mapslices(x -> binarize(x, Sauvola()), actinin_interp, dims=[2, 3])
    actinin_mc = MC(permutedims(actinin_bin, [3, 2, 1]))
    march(actinin_mc, 1.0)
    mesh_actinin = MarchingCubes.makemesh(Meshes, actinin_mc)


    fig = Figure(resolution=(1400, 700))
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    gl3 = fig[1, 3] = GridLayout()

    axis3_kwargs = Dict(:aspect => reverse(size(actinin_interp)), :elevation => 0.25π, :azimuth => -0.35π)
    ax1 = Axis3(gl1[1, 1]; axis3_kwargs...)
    viz!(mesh_desmin, color=1:nvertices(mesh_desmin), colorscheme=:Reds)
    fig
    ax2 = Axis3(gl2[1, 1]; axis3_kwargs...)
    viz!(mesh_actinin, color=1:nvertices(mesh_actinin), colorscheme=:Greens)
    ax3 = Axis3(gl3[1, 1]; axis3_kwargs...)
    viz!(mesh_desmin, color=1:nvertices(mesh_desmin), colorscheme=:Reds)
    viz!(mesh_actinin, color=1:nvertices(mesh_actinin), colorscheme=:Greens)

    for ax_ in [ax1, ax2, ax3]
        ax_.zticks = 0:20:reverse(size(actinin_interp))[end]
    end
    fig
end
##
let

    fig = Figure(resolution=(1600, 800))
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    interped1 = interpolate_zstack(out.data[1, 1:10, 250:750, 1000:1500], out.pixel_sizes)
    interped2 = interpolate_zstack(out.data[2, 1:5, 250:750, 1000:1500], out.pixel_sizes)
    indices = range(1, 33, length=25) |> collect |> x -> round.(Integer, x)
    for i in 1:5
        for j in 1:5
            ax1 = GLMakie.Axis(gl1[i, j], aspect=1)
            ax2 = GLMakie.Axis(gl2[i, j], aspect=1)
            #foreach(hidedecorations!, (ax1, ax2))
            #foreach(hidespines!, (ax1, ax2))

            ind = indices[5(i-1)+j]
            heatmap!(ax1, interped1[ind, :, :], colormap=:Reds)
            heatmap!(ax2, interped2[ind, :, :], colormap=:Greens)

        end
    end
    fig
end
##
interped = interpolate_zstack(out.data[1, 5:10, 100:500, 1500:1900], out.pixel_sizes)
lft1_i, lft1_o = Transforms.LineFilterTransform(interped, 20, 10, 20, interp_scheme=BSpline(Linear()))
##
begin
    fig = Figure(resolution=(800, 800))
    ax = Axis3(fig[1, 1], aspect=(50, 400, 400))

    volume!(lft1_i, algorithm=:mip)
    fig
end
##
begin
    mc = MC(lft1_i)
    march(mc, 0.1)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    screen = viz(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    march(mc, 1.0)
    mesh = MarchingCubes.makemesh(Meshes, mc)
    viz!(mesh, color=1:nvertices(mesh), colorscheme=:Spectral)
    display(screen)
end
##
oft1 = Transforms.OrientationFilterTransform(lft1_i, lft1_o, 20, 10, 20, BSpline(Linear()))
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