##
using Revise
using GLMakie
import Pkg
Pkg.activate(".")
##
using Images
using FileIO
using StaticArrays
import Statistics
using NearestNeighbors
using ProgressMeter
##
function preprocess_image(img_path)
    img = load(img_path)
    return @. Float32(Images.Gray(img))
end
##
desmin, actin = map(preprocess_image, ["./Data/SR2_desmin_image.jpg", "./Data/SR3_actin_image.jpg"])

function pointcloud_from_image(image::AbstractArray{T, N}) where {T, N}
    coords = map(findall(x -> x > zero(T), image)) do ind
        ind |> Tuple |> SVector{N, T}
    end
    return coords
end
##
import ImageBinarization: binarize, Otsu
algo = Otsu()
threshold(img) = binarize(img, algo)
##
let 
    fig = Figure(resolution = (1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Desmin")
    ax2 = GLMakie.Axis(fig[1, 2], title = "Thresholded (Otsu)")
    image!(ax1, desmin, colormap = :Reds_9)
    image!(ax2, threshold(desmin))
    fig
end
##
let 
    fig = Figure(resolution = (1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Actin")
    ax2 = GLMakie.Axis(fig[1, 2], title = "Thresholded (Otsu)")
    image!(ax1, actin, colormap = :Greens_9)
    image!(ax2, threshold(actin))
    fig
end
##
let 
    xrange, yrange = 1900:2900, 1000:3000
    fig = Figure(resolution = (1500, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Desmin (Thresholded)")
    ax2 = GLMakie.Axis(fig[1, 2], title = "Actin (Thresholded)")
    image!(ax1, threshold(desmin)[xrange, yrange])
    image!(ax2, threshold(actin)[xrange, yrange])
    fig
end

##
xrange, yrange = 1900:2900, 1000:3000
desmin_coord = threshold(desmin)[xrange, yrange] |> pointcloud_from_image;
actin_coord = threshold(actin)[xrange, yrange] |> pointcloud_from_image;
##
size(actin_coord)
##
import ThreadTools: tmap1
function RipleyK(r::T, tree::NNTree, area) where {T}
    counts = Statistics.mean([length(inrange(tree, p, r)) for p in tree.data])
    N = length(tree.data)
    return counts*area/N
end

function RipleyK(r, T, area, p::ProgressMeter.AbstractProgress)
    K = RipleyK(r, T, area)
    next!(p)
    return K
end

function RipleyK(r::AbstractVector, tree::NNTree, area)
    p = Progress(length(r))
    return tmap1(x -> RipleyK(x, tree, area, p), r)
end 

RipleyK(r, points, area) = RipleyK(r, KDTree(points), area)

##
r = 0:5:100
trees = map(KDTree, [desmin_coord, actin_coord])
##
K_desmin = RipleyK(r, trees[1], 1000*2000)
##
K_actin = RipleyK(r, trees[2], 1000*2000)
##
H(r, K) = sqrt.(K/π) - r
##
H_desmin, H_actin = H(r, K_desmin), H(r, K_actin)
##
let 
    fig = Figure(resolution = (1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Ripley K function")
    l1 = lines!(r, K_desmin)
    l2 = lines!(r, K_actin)
    Legend(fig[1, 2], [l1, l2], ["Desmin", "Actin"])

    ax1 = GLMakie.Axis(fig[1, 3], title = "Ripley H function")
    l3 = lines!(r, H_desmin)
    l4 = lines!(r, H_actin)
    Legend(fig[1, 4], [l3, l4], ["Desmin", "Actin"])

    fig
end
##
lines(r, H)
##

relative_angle(coord1::SVector{2, T}, coord2::SVector{2, T}) where {T} = atan(coord2[2]-coord1[2], coord2[1]-coord1[1])

function get_inrange_angles(r::T, tree, coord::SVector{2, T}) where {T}
    idx = inrange(tree, coord, r)
    angles = [relative_angle(coord, tree.data[i]) for i in idx]
    return angles
end

import StatsBase: Histogram, fit
function get_angle_bin_counts(angles, angles_bins)
    h = fit(Histogram, angles, angle_bins, closed = :left)
    return h.weights
end

r = 20.0f0
Δα = 5π/180
A = 1000*2000
angle_bins = float32.(collect(-π:Δα:π))
##
angle_counts_desmin = (2π/Δα)*(A/length(desmin_coord)^2)*sum(tmap1(c -> get_angle_bin_counts(get_inrange_angles(r, trees[1], c), angle_bins), desmin_coord))
angle_counts_actin = (2π/Δα)*(A/length(actin_coord)^2)*sum(tmap1(c -> get_angle_bin_counts(get_inrange_angles(r, trees[2], c), angle_bins), actin_coord))

##
extrema(angle_counts_desmin)
##


##
angle_bins
##
let 
    fig = Figure()
    xticks = ([-π, -π/2, 0,  π/2, π], ["-π", "-π/2", "0",  "π/2", "π"])
    ax = GLMakie.Axis(fig[1, 1], xticks = xticks, xlabel = "Angle (rad)", title = "Angular Ripley's function K(α; r = $(r))")
    A = 1000*2000
    l1 = lines!(ax, angle_bins[1:end-1], sqrt.(angle_counts_desmin/π).-r)
    l2 = lines!(ax, angle_bins[1:end-1], sqrt.(angle_counts_actin/π).-r)

    Legend(fig[1, 2], [l1, l2], ["Desmin", "Actin"])
    fig
end
##
