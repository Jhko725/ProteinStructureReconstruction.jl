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
include("hdbscan.jl")
using .HDBSCAN
##
filepath = "./Data/STORM/desmin_alphaactinin_600nm.csv"
const localization_df = DataFrame(CSV.File(filepath))
##

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
total_df_list = split_dataframe_per_probe(localization_df)
total_points = pointcloud_from_dataframe.(total_df_list)
xlim, ylim = (32500, 37500), (10300, 15300)
point_clouds = map(x->subset_region(x, xlim, ylim), total_points)
##

##
let
    fig = Figure()

    point_clouds_xy = map(x -> projection(x, 3), point_clouds)
    colors = [:red, :green]
    ax = Axis(fig[1, 1], aspect = 1, xlabel = "x (nm)", ylabel = "y (nm)", title = "Overlay for for x ∈ $xlim, y ∈ $ylim")
    for i in 1:length(point_clouds_xy)
        scatter!(ax, point_clouds_xy[i], color = colors[i], markersize = 2, transparency = true)
    end

    fig
end
##

let
    import ColorSchemes
    points = point_clouds[2]
    fig = Figure()
    ax = Axis3(fig[1, 1])
    #sc = scatter!(ax, points, color = to_abstractmatrix(points)[:,3], markersize = 1000)
    scatter!(ax, point_clouds[2], color = to_abstractmatrix(point_clouds[2])[3,:], markersize = 700, colormap = ColorSchemes.winter.colors)
    scatter!(ax, point_clouds[1], color = to_abstractmatrix(point_clouds[1])[3,:], markersize = 700, colormap = ColorSchemes.autumn1.colors)
    #Colorbar(fig[1, 2], sc)
    fig
    
end

##

##
using Clustering

##

ptc = PointCloud(point_clouds[2])
##

##
function cluster_coords(points::AbstractVector{T}, cluster::DbscanCluster) where {T<:SVector}
    indices = vcat(cluster.core_indices, cluster.boundary_indices)
    return points[indices]
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
points_xy = projection(point_clouds[2], 3) 
output = hdbscan(points_xy, min_cluster_size = 1250, min_samples = 15)
clusters, _ = get_cluster(point_clouds[2], output)

##
import Statistics: mean
import MultivariateStats: PCA, fit, eigvecs

function pca_orientation(points::AbstractVector{T}) where {T<:SVector}
    PCA_result = fit(PCA, to_abstractmatrix(points))
    return eigvecs(PCA_result)[:, 1]
end

centroids = mean.(clusters)
orientations = map(x-> projection(x, 3) |> pca_orientation, clusters)
##
let 
    import ColorSchemes: colorschemes, get
    import Statistics: mean

    fig = Figure()
    ax = Axis(fig[1, 1], aspect = 1, xlabel = "x (nm)", ylabel = "y (nm)", title = "Clustering results for x ∈ $xlim, y ∈ $ylim")
    clusters, noise = get_cluster(point_clouds[2], output)
    
    axis = 3
    scatter!(ax, projection(noise, axis), color = :gray, markersize = 2)
    for (i, cluster) in enumerate(clusters)
        
        clustered = projection(cluster, axis)
    
        color = colorschemes[:glasbey_hv_n256][i]
        scatter!(ax, clustered, markersize = 3)

        centroid = mean(clustered)
        scatter!(ax, centroid, color = :red, markersize = 9)

        orientation = pca_orientation(clustered)
        arrow_len = 500
        arrows!(ax, [centroid[1]], [centroid[2]], [orientation[1]*arrow_len], [orientation[2]*arrow_len], linewidth = 2.0)
        
    end
    fig
end
##
let 
    fig = Figure(resolution = (800, 1000))
    ax = Axis(fig[1, 1], aspect = 1, xlabel = "x (nm)", ylabel = "y (nm)", title = "x-y plane projection")
    
    avg_yaw = map(x-> projection(x, 3) |> pca_orientation, clusters) |> mean 
    
    reorient_points(points) = Pointcloud.rotate(points, -atan(avg_yaw[2], avg_yaw[1]), 0, 0)
    points2_rot = reorient_points(point_clouds[2])
    points2_rot1 = reorient_points(point_clouds[1])
    
    axis = 3
    plot1 = scatter!(ax, projection(points2_rot, axis), color = :green, markersize = 5, transparency = true)
    plot2 = scatter!(ax, projection(points2_rot1, axis), color = :red, markersize = 5, transparency = true)

    Legend(fig[1, 2], [plot1, plot2], ["α-actinin", "Desmin"])

   
    current_figure()
   
end
##
let 
    fig = Figure(resolution = (800, 1000))
    ax = Axis(fig[1, 1], aspect = 1, xlabel = "y (nm)", ylabel = "z (nm)", title = "Projection from the front")
    
    avg_yaw = map(x-> projection(x, 3) |> pca_orientation, clusters) |> mean 
    
    reorient_points(points) = Pointcloud.rotate(points, -atan(avg_yaw[2], avg_yaw[1]), 0, 0)
    points2_rot = reorient_points(point_clouds[2])
    points2_rot1 = reorient_points(point_clouds[1])
    
    axis = 1
    plot1 = scatter!(ax, projection(points2_rot, axis), color = :green, markersize = 5, transparency = true)
    plot2 = scatter!(ax, projection(points2_rot1, axis), color = :red, markersize = 5, transparency = true)

    Legend(fig[1, 2], [plot1, plot2], ["α-actinin", "Desmin"])

    ax2 = Axis(fig[2, :], xlabel = "Normal direction (nm)", ylabel = "Relative localization counts")
    hist1 = hist!(ax2, map(x -> getindex(x, 2), projection(points2_rot, 3)), bins = 100, color = (:green, 0.5), normalization = :pdf)
    hist2 = hist!(ax2, map(x -> getindex(x, 2), projection(points2_rot1, 3)), bins = 100, color = (:red, 0.5), normalization = :pdf)
    
   
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