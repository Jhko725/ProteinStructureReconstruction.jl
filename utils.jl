using DataFrames
using StaticArrays

function split_dataframe_per_probe(localization_df::DataFrame)
    probes = sort!(Vector(localization_df[:, :probe]))
    probe_ids = unique(probes)
    return [localization_df[localization_df.probe.==id, :] for id in probe_ids]
end

function pointcloud_from_dataframe(localization_df::DataFrame)
    point_cloud = @inbounds @views [SVector{3,Float32}(localization_df[i, [:x, :y, :z]]...) for i = 1:size(localization_df)[1]]
    return point_cloud
end

function invertdims(array::AbstractArray)::AbstractArray
    return permutedims(array, reverse(1:ndims(array)))
end