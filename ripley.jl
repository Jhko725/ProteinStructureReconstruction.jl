module Ripley

using NearestNeighbors
using StaticArrays
import StatsBase: Histogram, fit

export K, AngularRipleyEstimator, SimpleROI, erode, GetisFranklin_L, K_angular, H_angular

## Some utility functions 

include("roi.jl")
function inrangecount(tree::NNTree, points::Vector{T}, r::Number) where {T<:AbstractVector}
    return length.(inrange(tree, points, r, false))
end


##


# TODO: add field & type hierarchy for boundary condition
struct AngularRipleyEstimator{T<:AbstractFloat}
    kdtree::KDTree
    area::T

    AngularRipleyEstimator(points::Vector{SVector}, area) = AngularRipleyEstimator(KDTree(points), area)
end

function K_angular(point_cloud, r::T, Δα::T, xlims, ylims) where {T<:AbstractFloat}
    π_ = T(π)
    tree = KDTree(point_cloud)
    α_bin_edges = collect(zero(T):Δα:2π_)
    counts = zeros(T, length(α_bin_edges) - 1)

    for point in point_cloud
        inrange_inds = inrange(tree, point, r, false)
        angles = [relative_angle(point, tree.data[i]) for i in inrange_inds]

        counts .+= get_angle_bin_counts(angles, α_bin_edges)
    end

    A = length(SimpleROI(xlims, ylims))
    N = length(point_cloud)
    return α_bin_edges[1:end-1], (2π_ / Δα) * A / N^2 * counts
end

function K(Δα::T, r::T, tree::KDTree, area) where {T<:AbstractFloat}
    π_ = T(π)
    α_bin_edges = collect(zero(T):Δα:2π_)
    counts = zeros(T, length(α_bin_edges) - 1)

    for point in tree.data
        inrange_inds = inrange(tree, point, r)

        angles = [relative_angle(point, tree.data[i]) for i in inrange_inds]

        counts += get_angle_bin_counts(angles, α_bin_edges)
    end

    return α_bin_edges[1:end-1], counts
end

relative_angle(coord1::SVector{2,T}, coord2::SVector{2,T}) where {T<:AbstractFloat} = atan(coord2[2] - coord1[2], coord2[1] - coord1[1]) + T(π)


function H_angular(point_cloud, r::T, Δα::T, xlims, ylims) where {T<:AbstractFloat}
    angles, K = K_angular(point_cloud, r, Δα, xlims, ylims)
    L = sqrt.(K / T(π))
    return angles, L .- r
end


function get_angle_bin_counts(angles, bin_edges)
    h = fit(Histogram, angles, bin_edges, closed=:left)
    return h.weights
end


function filter_points(points, radius, threshold_count)
    tree = KDTree(points)
    neighborscount = inrangecount(tree, points, radius)
    is_valid = neighborscount .> threshold_count
    accepted = points[is_valid]
    rejected = points[.!is_valid]
    return accepted, rejected
end

function GetisFranklin_L(point_cloud, r, xlims, ylims, zlims)
    return GetisFranklin_L(point_cloud, point_cloud, r, xlims, ylims, zlims)
end

function GetisFranklin_L(point_cloud1, point_cloud2, r, xlims, ylims, zlims)
    roi_inner = SimpleROI(xlims, ylims, zlims) |> x -> erode(x, r)
    points1_inner = filter(pt -> pt ∈ roi_inner, point_cloud1)
    tree2 = KDTree(point_cloud2)
    counts12 = inrangecount(tree2, points1_inner, r)
    K12 = length(roi_inner) * counts12 / sum(counts12)
    L12 = cbrt.(K12 / (4π / 3))
    return points1_inner, L12
end

end