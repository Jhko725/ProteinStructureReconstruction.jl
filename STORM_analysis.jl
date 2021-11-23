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
filepath = "./Data/STORM/desmin_alphaactinin_600nm.csv"
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
xlim, ylim = (25000, 30000), (10000, 15000)
df = subset_xy_bounding_box(localization_df, xlim, ylim)
df_list =  split_dataframe_per_probe(df)
point_clouds = pointcloud_from_dataframe.(df_list)
##
import StatsBase: fit, Histogram
function point_to_set_correlation(point_coordinates, set_coordinates, Δr)
    set_tree = KDTree(set_coordinates)
    _, point_to_set_dists = nn(set_tree, point_coordinates)
    r = collect(0.0:Δr:maximum(point_to_set_dists)+Δr)

    point_to_set_hist = fit(Histogram, point_to_set_dists, r, closed = :left)
    return r[1:end-1], point_to_set_hist.weights
##
import RecursiveArrayTools: VectorOfArray
using Clustering

point_matrix = reshape(reinterpret(Float32, point_clouds[1]), (3, length(point_clouds[1])))
##
point_clouds[1][1] == point_matrix[:, 1]

#size(point_matrix)
##
clustering_result = dbscan(point_matrix, 5, min_cluster_size = 2)
##
VectorOfArray isa AbstractMatrix
##
import StatsBase: fit, Histogram
bin_size = 20
point_set_dist = desmin_pt_alphaactinin_set_dist
distances = collect(0.0:bin_size:maximum(point_set_dist)+bin_size)
h = fit(Histogram, point_set_dist, distances, closed=:left)
##
let 
    fig = Figure(resolution = (1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Point(desmin)-to-Set(α-actinin) correlation function: Δr = $(bin_size)nm; x ∈ $(Tuple(xlim./1000)) μm; y ∈ $(Tuple(ylim./1000)) μm", xlabel = "Distance (nm)", ylabel = "Correlation (a.u.)")
    #ax2 = GLMakie.Axis(fig[1, 2], yscale = log10, title = "Point(desmin)-to-Set(α-actinin) correlation function (Δr = $(bin_size)nm)", xlabel = "Distance (pixels)", ylabel = "Correlation (a.u.)")
    scatterlines!(ax1, distances[1:end-1], h.weights)
    ax1.xticks = LinearTicks(10)
    #scatterlines!(ax2, distances[1:end-1], h.weights)
    fig
end