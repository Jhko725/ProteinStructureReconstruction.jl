##
using Revise
using GLMakie
using Pkg
Pkg.activate(".")
using ProteinStructureReconstruction
using HDF5
using Images
import Statistics: mean, median
import GeometryBasics: Point2f0
##
filepath = "./Data/Fed_X63_Z3_SIM.h5"
h5file = h5open(filepath, "r")
##
# TODO: don't load everything in memory at once -  just draw overlay first, then 
##
x_range, y_range = 2000:2300, 1600:1900
img_data = h5file["Data"][x_range, y_range, :, :] |> x -> reinterpret(N0f16, x)
##
function make_overlay(zstack_dataset::AbstractArray; dims = 0)
    overlay = zstack_dataset |> x -> reinterpret(N0f16, x) |> x ->  mean(x, dims = dims) |> x -> dropdims(x, dims = dims)
    return overlay
end
##
overlay = make_overlay(read(h5file["Data"]), dims = 3)
##

let
    x1, x2 = first(x_range), last(x_range)
    y1, y2 = first(y_range), last(y_range)
    rect = Point2f0[(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
    fig = Figure(resolution = (1500, 800))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Desmin overlay", titlesize = 20.0f0)
    ax2 = GLMakie.Axis(fig[1, 2], title = "Actin overlay", titlesize = 20.0f0)
    axes = [ax1, ax2]
    for (i, ax) in enumerate(axes)
        image!(ax, overlay[:, :, i])
        poly!(ax, rect, color = nothing, strokecolor = :red, strokewidth = 2.0)
    end
    fig
end
##
let
    fig = Figure(resolution = (1500, 800))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Desmin overlay", titlesize = 20.0f0)
    ax2 = GLMakie.Axis(fig[1, 2], title = "Actin overlay", titlesize = 20.0f0)
    axes = [ax1, ax2]
    for (i, ax) in enumerate(axes)
        image!(ax, overlay[x_range, y_range, i])
    end
    fig
end
##
size(h5file["Data"])
##
function plot_image_histogram(image:AbstractArray)
    flattened = vec(image)

end
##
image(overlay[1500:2000, 2000:2500, 2])
##
import StatsBase: fit, Histogram
let 
    fig = Figure()
    axes = [GLMakie.Axis(fig[1, 1]), GLMakie.Axis(fig[1, 2])]
    Δz = 7
    for i in 1:Δz:size(h5file["Data"])[3], j in 1:size(h5file["Data"])[4]
        print(i)
        img = h5file["Data"][:, :, i, j]
        flat_img = vec(img)
        density!(axes[j], flat_img[flat_img .> 0], label = "z slice #$(i)")
    end
    
    fig
end
##
let 
    fig = Figure()
    means = Array{Float32}(undef, size(h5file["Data"])[3:4]...)
    for i in 1:size(h5file["Data"])[3], j in 1:size(h5file["Data"])[4]
        img = h5file["Data"][:, :, i, j]
        flat_img = vec(img)
        means[i, j] = median(flat_img[flat_img .> 0])
    end
    
    axes = [GLMakie.Axis(fig[1, 1]), GLMakie.Axis(fig[1, 2])]
    for (i, ax) in enumerate(axes)
        plot!(ax, means[:, i])
    end
    fig
end
##

##
img_hist = fit(Histogram, vec(overlay[:, :, 2]), nbins = 30)
barplot(img_hist)
##
let 
    fig = Figure(resolution = (1200, 800))
    axes = [Axis3(fig[1, 1], title = "Actin (Raw)"), Axis3(fig[1, 2], title = "Desmin (Raw)")]
    cmaps = [:Greens_3, :Reds_3]
    for (i, ax) in enumerate(axes)
        contour!(ax, img_data[:, :, :, i], levels = 4, alpha = 0.4, colormap = cmaps[i])
    end
    
    fig
end
##
function equalize_zstack(zstack::AbstractArray)
    equalizer = Equalization(nbins = 256, minval = 0.0, maxval = 0.8)
    out = mapslices(x -> adjust_histogram(x, equalizer), zstack, dims = [1, 2])
    return out
end
##
read(h5file["Data"]["channels"])
##
img_eq = equalize_zstack(img_data)