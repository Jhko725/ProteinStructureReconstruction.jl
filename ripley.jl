module Ripley

using NearestNeighbors
using StaticArrays
import StatsBase: Histogram, fit

export K, AngularRipleyEstimator

# TODO: add field & type hierarchy for boundary condition
struct AngularRipleyEstimator{T<:AbstractFloat}
    kdtree::KDTree
    area::T

    AngularRipleyEstimator(points::Vector{SVector}, area) = AngularRipleyEstimator(KDTree(points), area)
end

function K(Δα::T, r::T, tree::KDTree, area) where {T<:AbstractFloat}
    π_ = T(π)
    α_bin_edges = collect(-π_:Δα:π_)
    counts = zeros(T, length(α_bin_edges)-1)

    for point in tree.data
        inrange_inds = inrange(tree, point, r)

        angles = [relative_angle(point, tree.data[i]) for i in inrange_inds]

        counts += get_angle_bin_counts(angles, α_bin_edges)
    end

    return α_bin_edges[1:end-1], counts
end

relative_angle(coord1::SVector{2, T}, coord2::SVector{2, T}) where {T<:AbstractFloat} = atan(coord2[2]-coord1[2], coord2[1]-coord1[1])

function get_angle_bin_counts(angles, bin_edges)
    h = fit(Histogram, angles, bin_edges, closed = :left)
    return h.weights
end

end