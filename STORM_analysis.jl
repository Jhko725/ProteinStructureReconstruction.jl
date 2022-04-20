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

include("ripley.jl")
using .Ripley
include("pointcloud.jl")
using .Pointcloud

##
filepath = "./Data/STORM/desmin_alphaactinin_600nm.csv"
const localization_df = DataFrame(CSV.File(filepath))
##
localization_df.x
hist(localization_df.z, bins = 100)
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

function pointcloud_from_dataframe2d(localization_df::DataFrame)
    point_cloud = @inbounds @views [SVector{2, Float32}(localization_df[i, [:x, :y]]...) for i = 1:size(localization_df)[1]]
    return point_cloud
end

##
xlim, ylim = (0, 5000), (25500, 30500)
df = subset_xy_bounding_box(localization_df, xlim, ylim)
df_list =  split_dataframe_per_probe(df)
point_clouds = pointcloud_from_dataframe2d.(df_list)
##
scatter(point_clouds[2])
##
let
    fig = Figure()
    df_list = split_dataframe_per_probe(localization_df)
    point_clouds = pointcloud_from_dataframe2d.(df_list)
    colors = [:red, :green]
    ax = Axis(fig[1, 1], aspect = 1)
    for i in 1:length(df_list)
        
        scatter!(ax, point_clouds[i], color = colors[i], markersize = 1, transparency = true)
    end

    fig
end
##

##
let
    points = pointcloud_from_dataframe.(df_list)[2]
    fig = Figure()
    ax = Axis3(fig[1, 1])
    sc = scatter!(ax, points, color = to_abstractmatrix(points)[:,3], markersize = 1000)
    Colorbar(fig[1, 2], sc)
    fig
end
##

##
using Clustering
import MultivariateStats: PCA, fit, eigvecs
import Statistics.mean
##

ptc = PointCloud(point_clouds[2])
##

##
function cluster_coords(points::AbstractVector{T}, cluster::DbscanCluster) where {T<:SVector}
    indices = vcat(cluster.core_indices, cluster.boundary_indices)
    return points[indices]
end

function pca_orientation(points::AbstractVector{T}) where {T<:SVector}
    PCA_result = fit(PCA, to_abstractmatrix(points))
    return eigvecs(PCA_result)[:, 1]
end
##
let 
    fig = Figure()
    ax = Axis3(fig[1, 1])
    scatter!(ax, point_clouds[2], color = :lightgray, markersize = 1000)
    points = to_abstractmatrix(point_clouds[2])
    clusters = dbscan(points, 450, min_cluster_size = 1000)
    for c in clusters
        c_points = cluster_coords(point_clouds[2], c)
        scatter!(ax, c_points, markersize = 1000)

        centroid = mean(c_points)
        orientation = pca_orientation(c_points)
        arrow_len = 1000
        arrows!(ax, [centroid[1]], [centroid[2]], [centroid[3]], [orientation[1]*arrow_len], [orientation[2]*arrow_len], [orientation[3]*arrow_len], linewidth = 50)
        scatter!(ax, centroid, color = :red, markersize = 3000)
    end
    current_figure()
end
##
let 
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1)
    scatter!(ax, point_clouds[2], color = :lightgray, markersize = 5)
    points = to_abstractmatrix(point_clouds[2])
    clusters = dbscan(points, 300, min_cluster_size = 1000, min_neighbors = 7)
    for c in clusters
        c_points = cluster_coords(point_clouds[2], c)
        scatter!(ax, c_points, markersize = 5)

        centroid = mean(c_points)
        orientation = pca_orientation(c_points)
        arrow_len = 500
        arrows!(ax, [centroid[1]], [centroid[2]], [orientation[1]*arrow_len], [orientation[2]*arrow_len], linewidth = 2.0)
            
        scatter!(ax, centroid, color = :red, markersize = 9)
        
    end
    current_figure()
   
end
##
let 
    import PyCall
    import ColorSchemes: colorschemes
    hdbscan = PyCall.pyimport("hdbscan")
    clusterer = hdbscan.HDBSCAN(min_cluster_size = 1000, min_samples = 10)
    points = to_abstractmatrix(point_clouds[2]) |> transpose
    output = clusterer.fit(points)

    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1, xlabel = "x (nm)", ylabel = "y (nm)", title = "Clustering results for x ∈ $xlim, y ∈ $ylim")
    cluster_labels = output.labels_ .+ 1
    for i in unique(cluster_labels)
        clustered = point_clouds[2][cluster_labels .== i]
        if i == 0
            scatter!(ax, clustered, color = :gray, markersize = 2)
        else
            color = colorschemes[:glasbey_hv_n256][i]
            scatter!(ax, clustered, markersize = 3)
        end
    end
    fig
end
##
let 
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1)
    points = to_abstractmatrix(point_clouds[2])
    clusters = dbscan(points, 240, min_cluster_size = 1000)

    clusters_coords = [cluster_coords(point_clouds[2], c) for c in clusters]
    avg_orient = map(pca_orientation, clusters_coords) |> mean
    
    points2_rot = Pointcloud.rotate(point_clouds[2], -atan(avg_orient[2], avg_orient[1]))
    points2_rot1 = Pointcloud.rotate(point_clouds[1], -atan(avg_orient[2], avg_orient[1]))
    typeof(points2_rot)
    plot1 = scatter!(ax, points2_rot, color = :green, markersize = 5, transparency = true)
    plot2 = scatter!(ax, points2_rot1, color = :red, markersize = 5, transparency = true)

    Legend(fig[1, 2], [plot1, plot2], ["α-actinin", "Desmin"])
    current_figure()
   
end
##
let 
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1)
    points = to_abstractmatrix(point_clouds[2])
    clusters = dbscan(points, 240, min_cluster_size = 1000)

    clusters_coords = [cluster_coords(point_clouds[2], c) for c in clusters]
    avg_orient = map(pca_orientation, clusters_coords) |> mean
    
    points2_rot = Pointcloud.rotate(point_clouds[2], -atan(avg_orient[2], avg_orient[1]))
    points2_rot1 = Pointcloud.rotate(point_clouds[1], -atan(avg_orient[2], avg_orient[1]))
    print(size(to_abstractmatrix(points2_rot)))
    plot1 = scatter!(ax, to_abstractmatrix(points2_rot)[2,:], df_list[2].z, color = :green, markersize = 5, transparency = true)
    plot2 = scatter!(ax, to_abstractmatrix(points2_rot1)[2,:], df_list[1].z, color = :red, markersize = 5, transparency = true)

    Legend(fig[1, 2], [plot1, plot2], ["α-actinin", "Desmin"])
    current_figure()
   
end

##
let
    points = to_abstractmatrix(point_clouds[2])
    clusters = dbscan(points, 240, min_cluster_size = 1000)

    clusters_coords = [cluster_coords(point_clouds[2], c) for c in clusters]
    avg_orient = map(pca_orientation, clusters_coords) |> mean
    points2_rot = Pointcloud.rotate(point_clouds[2], -atan(avg_orient[2], avg_orient[1]))
    points2_rot1 = Pointcloud.rotate(point_clouds[1], -atan(avg_orient[2], avg_orient[1]))

    fig = Figure()
    ax = Axis(fig[1, 1])
    plot1 = hist!(ax, to_abstractmatrix(points2_rot)[2,:], bins = 100, color = (:green, 0.5), normalization = :pdf)
    
    plot2 = hist!(ax, to_abstractmatrix(points2_rot1)[2,:], color = (:red, 0.5), bins = 100, normalization = :pdf)
    Legend(fig[1, 2], [plot1, plot2], ["α-actinin", "Desmin"])
    current_figure()
end
##
scatter(point_clouds[2])
##
tree = KDTree(point_clouds[2])
##
bins, angle_counts = K(Float32(π/180), 1000f0, tree, 1000*2000)
##
_, i_max = findmax(sqrt.(angle_counts[angle_counts.>π/2]))
bins[i_max]
##
lines(bins, sqrt.(angle_counts))
##
function rotate_point(point::SVector{2}, θ)
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    return R*point
end

rotated_cloud = map(pt -> rotate_point(pt, -(1.9334161376300993-0.5π)), point_clouds[2])
rotated_cloud1 = map(pt -> rotate_point(pt, -(1.95-0.5π)), point_clouds[1])
##
rotated_mat = hcat(rotated_cloud...)
rotated_mat1 = hcat(rotated_cloud1...)
scatter(rotated_mat[1,:], rotated_mat[2,:])
##
scatter(rotated_mat[1,:], df_list[2].z)

##
scatter(rotated_mat1[1,:], df_list[1].z)
##
hist(rotated_mat[1,:], bins = 100)
hist!(rotated_mat1[1,:], bins = 100)
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
##
type