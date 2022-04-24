module HDBSCAN

using StaticArrays
import PyCall
const _hdbscan = PyCall.pyimport("hdbscan")

include("pointcloud.jl")
import .Pointcloud: to_abstractmatrix

export hdbscan, get_cluster

function hdbscan(points::Vector{T}; min_cluster_size, min_samples, hdbscan_kwargs...) where {T<:SVector}
    hdbscan = PyCall.pyimport("hdbscan")
    clusterer = _hdbscan.HDBSCAN(min_cluster_size = min_cluster_size, min_samples = min_samples, hdbscan_kwargs...)
    output = points |> to_abstractmatrix |> transpose |> clusterer.fit
    return HDBSCANResult(output)
end

struct HDBSCANResult
    clusterer
end

cluster_labels(result::HDBSCANResult) = result.clusterer.labels_ .+  1

function label_ids(result::HDBSCANResult)
    ids = result |> cluster_labels |> unique
    sort!(ids)
    return ids
end

noise_id(result::HDBSCANResult) = 0

cluster_ids(result::HDBSCANResult) = label_ids(result)[2:end]

n_clusters(result::HDBSCANResult) = length(label_ids(result)) - 1

function get_cluster(points::Vector{T}, clustering_result::HDBSCANResult) where {T<:SVector}
    labels = cluster_labels(clustering_result)
    ids = cluster_ids(clustering_result)
    clusters = [points[labels .== i] for i in ids]
    noise = points[labels .== noise_id(clustering_result)]
    return clusters, noise
end

end