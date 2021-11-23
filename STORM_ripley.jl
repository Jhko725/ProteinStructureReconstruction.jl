##
using Revise
using GLMakie
import Pkg
Pkg.activate(".")
##
import CSV
using DataFrames
using StaticArrays
import Statistics
using NearestNeighbors
using ProgressMeter
##
filepath = "./Data/STORM/actin_desmin_600nm.csv"
localization_df = DataFrame(CSV.File(filepath))
##
function subset_xy_bounding_box(localization_df::DataFrame, xlim, ylim)
    selection_criteria_x = (xlim[1] .< localization_df.x .< xlim[2])
    selection_criteria_y = (ylim[1] .< localization_df.y .< ylim[2])
    selection_criteria = selection_criteria_x .& selection_criteria_y
    return localization_df[selection_criteria, :]
end

function split_dataframe_per_probe(localization_df::DataFrame)
    probes = sort!(Vector(localization_df[:, :probe]))
    probe_ids = unique(probes)
    return [localization_df[localization_df.probe .== id, :] for id in probe_ids]
end

function pointcloud_from_dataframe(localization_df::DataFrame)
    point_cloud = @inbounds @views [SVector{3, Float32}(localization_df[i, [:x, :y, :z]]...) for i = 1:size(localization_df)[1]]
    return point_cloud
end
##
xlim, ylim = (25000, 30000), (25000, 30000)
df = subset_xy_bounding_box(localization_df, xlim, ylim)
df_list =  split_dataframe_per_probe(df)
point_clouds = pointcloud_from_dataframe.(df_list)
αactinin_tree = KDTree(point_clouds[2])
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
