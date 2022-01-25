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
const localization_df = DataFrame(CSV.File(filepath))
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
end
##
function xy_region_to_point_to_set_distances(x_limit::Tuple{N, N}, y_limit::Tuple{N, N}, localization_df::DataFrame) where {N<:Real}
    df = subset_xy_bounding_box(localization_df, x_limit, y_limit)
    df_list =  split_dataframe_per_probe(df)
    point_clouds = pointcloud_from_dataframe.(df_list)

    set_tree = KDTree(point_clouds[2])
    _, point_to_set_dist = nn(set_tree, point_clouds[1])
    return point_to_set_dist
end
##

##
xlim, ylim = (25000, 30000), (10000, 15000)
import StatsBase: fit, Histogram
bin_size = 20
point_set_dist = xy_region_to_point_to_set_distances((25000, 30000), (10000, 15000), localization_df)
distances = collect(0.0:bin_size:(maximum(point_set_dist)+bin_size))
h = fit(Histogram, point_set_dist, distances, closed=:left)
##
import StatsBase: fit, Histogram
function point_to_set_correlation(point_to_set_distances::Vector{N}, bin_size::N) where {N<:Number}
    distances = collect(0.0:bin_size:(maximum(point_to_set_distances)+bin_size))
    h = fit(Histogram, point_to_set_distances, distances, closed=:left)
    return distances[1:end-1], h.weights
end
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
##
x_lims = [(5000, 10000), (10000, 15000), (15000, 20000), (20000, 25000), (25000, 30000), (30000, 35000),
(0000, 5000), (5000, 10000), (10000, 15000), (15000, 20000), (20000, 25000), (25000, 30000), (30000, 35000)]
y_lims = [(5000, 10000), (7500, 12500), (2500, 7500), (7500, 12500), (7500, 12500), (10000, 15000),
(25000, 30000), (27500, 32500), (30000, 35000), (30000, 35000), (27500, 32500), (30000, 35000), (27500, 32500)]

point_set_dist_list = @showprogress [xy_region_to_point_to_set_distances(x_lim, y_lim, localization_df) for (x_lim, y_lim) in zip(x_lims, y_lims)]
##
bin_size = 10.0f0
p_to_s_corr_list = @showprogress [point_to_set_correlation(p_t_s_dist, bin_size) for p_t_s_dist in point_set_dist_list]
##
let 
    fig = Figure(resolution = (1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Point(desmin)-to-Set(α-actinin) correlation function: Δr = $(bin_size)nm", xlabel = "Distance (nm)", ylabel = "Correlation (a.u.)")
    #ax2 = GLMakie.Axis(fig[1, 2], yscale = log10, title = "Point(desmin)-to-Set(α-actinin) correlation function (Δr = $(bin_size)nm)", xlabel = "Distance (pixels)", ylabel = "Correlation (a.u.)")
    @showprogress for (r, corr) in p_to_s_corr_list
        lines!(ax1, r, corr)
    end
    ax1.xticks = LinearTicks(10)
    #scatterlines!(ax2, distances[1:end-1], h.weights)
    fig
end
##
total_dists = cat(point_set_dist_list..., dims = 1)
total_corr = point_to_set_correlation(total_dists, bin_size)
let 
    fig = Figure(resolution = (1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Point(desmin)-to-Set(α-actinin) correlation function: Δr = $(bin_size)nm", xlabel = "Distance (nm)", ylabel = "Correlation (a.u.)")
    #ax2 = GLMakie.Axis(fig[1, 2], yscale = log10, title = "Point(desmin)-to-Set(α-actinin) correlation function (Δr = $(bin_size)nm)", xlabel = "Distance (pixels)", ylabel = "Correlation (a.u.)")
    
    lines!(ax1, total_corr...)
    
    ax1.xticks = LinearTicks(10)
    #scatterlines!(ax2, distances[1:end-1], h.weights)
    fig
end