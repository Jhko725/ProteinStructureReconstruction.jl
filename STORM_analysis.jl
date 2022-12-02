##
using Revise
using WGLMakie
import Pkg
Pkg.activate(".")
##
import CSV
using DataFrames
using StaticArrays
import Statistics: mean
using NearestNeighbors
using ProgressMeter
import ColorSchemes

include("ripley.jl")
using .Ripley
include("pointcloud.jl")
using .Pointcloud
include("hdbscan.jl")
using .HDBSCAN
include("utils.jl")
##
const path_actinin = "./Data/STORM/desmin_alphaactinin_600nm.csv"
const path_actin = "./Data/STORM/actin_desmin_1.2um.csv"
const color_dict = Dict(zip(["desmin", "actinin", "actin"], [:red, :green, :blue]))
##
path = "./Data/STORM/desmin_alphaactinin_2.5um.csv"
points_df = path |> CSV.File |> DataFrame |> split_dataframe_per_probe
points_list = pointcloud_from_dataframe.(points_df)
points_dict = Dict(zip(["desmin", "actinin"], points_list))
total = hcat([reduce(hcat, v) for v in values(points_dict)]...)
print(maximum(total, dims=2) - minimum(total, dims=2))
##
points_list = path_actinin |> CSV.File |> DataFrame |> split_dataframe_per_probe
points_list = pointcloud_from_dataframe.(points_list)
points_dict = Dict(zip(["desmin", "actinin"], points_list))

#points_list = path_actin |> CSV.File |> DataFrame |> split_dataframe_per_probe
#points_list = pointcloud_from_dataframe.(points_list)
#points_dict = Dict(zip(["actin", "desmin"], points_list))
##
roi = SimpleROI((1.5, 6.5), (0.5, 5.5), (-0.2, 0.4))

##
let
    r = 100
    points = total_points[1]
    tree = KDTree(points)
    inboundscount = length.(inrange(tree, points, r, false))
    hist(inboundscount, bins=200)
end
##

filtered_points = Dict(k => Ripley.filter_points(v, 200, 5)[1] / 1000 for (k, v) in points_dict)

##
x_lims = [(2.5, 7.5), (8.0, 13.0), (13.5, 18.5), (13.5, 18.5), (7.0, 12.0), (13.5, 18.5)]
y_lims = [(1.0, 6.0), (1.0, 6.0), (1.0, 6.0), (6.5, 11.5), (6.5, 11.5), (12.0, 17.0)]
z_lims = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]

let
    ind = 6
    xlim, ylim, zlim = x_lims[ind], y_lims[ind], z_lims[ind]
    roi_points = Dict(k => Pointcloud.subset_region(v, xlim, ylim, zlim) for (k, v) in filtered_points)
    fig = Figure(resolution=(1500, 800))

    ax1 = Axis(fig[1, 1], aspect=1, xlabel="x (μm)", ylabel="y (μm)", title="Overlay for for x ∈ $xlim, y ∈ $ylim, z ∈ $zlim")
    ax2 = Axis3(fig[1, 2], xlabel="x (μm)", ylabel="y (μm)", zlabel="z (μm)", aspect=(4, 4, 0.6), elevation=0.15π, azimuth=0.6π, alignmode=Outside(40))
    for (name, points) in roi_points
        scatter!(ax1, points, color=(color_dict[name], 0.5), markersize=2, transparency=true)

        scatter!(ax2, points, color=(color_dict[name], 0.5), markersize=600, transparency=true)
    end
    #save("./actinin_roi_6.png", fig)

    fig
end
##
scatter(filtered_points["desmin"])
##
xlim, ylim, zlim = (1.5, 6.5), (0.5, 5.5), (-0.2, 0.4)
roi_points = Dict(k => Pointcloud.subset_region(v / 1000, xlim, ylim, zlim) for (k, v) in points_dict)
##
roi_points["desmin"]
##
let

    points = point_clouds[1]
    fig = Figure()
    ax = Axis3(fig[1, 1])
    #sc = scatter!(ax, points, color = to_abstractmatrix(points)[:,3], markersize = 1000)
    scatter!(ax, point_clouds[2], color=to_abstractmatrix(point_clouds[1])[3, :], markersize=400, colormap=ColorSchemes.winter.colors)
    scatter!(ax, point_clouds[1], color=to_abstractmatrix(point_clouds[2])[3, :], markersize=700, colormap=ColorSchemes.autumn1.colors)
    #Colorbar(fig[1, 2], sc)
    fig

end

##


##
using Clustering
ptc = PointCloud(point_clouds[1])
##
(@SVector [(1, 2), (3, 4), (4, 5)]) isa SVector{3,Tuple}
##
function cluster_coords(points::AbstractVector{T}, cluster::DbscanCluster) where {T<:SVector}
    indices = vcat(cluster.core_indices, cluster.boundary_indices)
    return points[indices]
end

import Statistics: mean
import MultivariateStats: PCA, fit, eigvecs

function pca_orientation(points::AbstractVector{T}) where {T<:SVector}
    PCA_result = fit(PCA, to_abstractmatrix(points))
    return eigvecs(PCA_result)[:, 1]
end
##
let
    fig = Figure()
    ax = Axis3(fig[1, 1])
    cloud = point_clouds[1]
    scatter!(ax, cloud, color=:lightgray, markersize=1000)
    points = to_abstractmatrix(cloud)
    clusters = dbscan(points, 450, min_cluster_size=1000)
    for c in clusters
        c_points = cluster_coords(cloud, c)
        scatter!(ax, c_points, markersize=1000)

        centroid = mean(c_points)
        orientation = pca_orientation(c_points)
        arrow_len = 1000
        arrows!(ax, [centroid[1]], [centroid[2]], [centroid[3]], [orientation[1] * arrow_len], [orientation[2] * arrow_len], [orientation[3] * arrow_len], linewidth=50)
        scatter!(ax, centroid, color=:red, markersize=3000)
    end
    current_figure()
end
##
let
    cloud = point_clouds[2]
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=1)
    scatter!(ax, cloud, color=:lightgray, markersize=3)
    points = to_abstractmatrix(cloud)
    clusters = dbscan(points, 200, min_cluster_size=800, min_neighbors=10)
    for c in clusters
        c_points = cluster_coords(cloud, c)
        scatter!(ax, c_points, markersize=3)

        centroid = mean(c_points)
        orientation = pca_orientation(c_points)
        arrow_len = 500
        arrows!(ax, [centroid[1]], [centroid[2]], [orientation[1] * arrow_len], [orientation[2] * arrow_len], linewidth=5.0)

        scatter!(ax, centroid, color=:red, markersize=11)
    end
    current_figure()

end
##
points_xy = projection(point_clouds[2], 3)
output = hdbscan(points_xy, min_cluster_size=1250, min_samples=15)
clusters, _ = get_cluster(point_clouds[2], output)

##


centroids = mean.(clusters)
orientations = map(x -> projection(x, 3) |> pca_orientation, clusters)
##
let
    import ColorSchemes: colorschemes, get
    import Statistics: mean

    fig = Figure()
    ax = Axis(fig[1, 1], aspect=1, xlabel="x (nm)", ylabel="y (nm)", title="Clustering results for x ∈ $xlim, y ∈ $ylim")
    clusters, noise = get_cluster(point_clouds[2], output)

    axis = 3
    scatter!(ax, projection(noise, axis), color=:gray, markersize=2)
    for (i, cluster) in enumerate(clusters)

        clustered = projection(cluster, axis)

        color = colorschemes[:glasbey_hv_n256][i]
        scatter!(ax, clustered, markersize=3)

        centroid = mean(clustered)
        scatter!(ax, centroid, color=:red, markersize=9)

        orientation = pca_orientation(clustered)
        arrow_len = 500
        a = arrows!(ax, [centroid[1]], [centroid[2]], [orientation[1] * arrow_len], [orientation[2] * arrow_len], linewidth=2.0)


    end
    fig
end
##
let
    fig = Figure(resolution=(800, 1000))
    ax = Axis(fig[1, 1], aspect=1, xlabel="x (nm)", ylabel="y (nm)", title="x-y plane projection")

    avg_yaw = map(x -> projection(x, 3) |> pca_orientation, clusters) |> mean

    reorient_points(points) = Pointcloud.rotate(points, -atan(avg_yaw[2], avg_yaw[1]), 0, 0)
    points2_rot = reorient_points(point_clouds[2])
    points2_rot1 = reorient_points(point_clouds[1])

    axis = 3
    plot1 = scatter!(ax, projection(points2_rot, axis), color=:green, markersize=5, transparency=true)
    plot2 = scatter!(ax, projection(points2_rot1, axis), color=:red, markersize=5, transparency=true)

    Legend(fig[1, 2], [plot1, plot2], ["α-actinin", "Desmin"])


    current_figure()

end
##
let
    fig = Figure(resolution=(800, 1000))
    ax = Axis(fig[1, 1], aspect=1, xlabel="y (nm)", ylabel="z (nm)", title="Projection from the front")

    avg_yaw = map(x -> projection(x, 3) |> pca_orientation, clusters) |> mean

    reorient_points(points) = Pointcloud.rotate(points, -atan(avg_yaw[2], avg_yaw[1]), 0, 0)
    points2_rot = reorient_points(point_clouds[2])
    points2_rot1 = reorient_points(point_clouds[1])

    axis = 1
    plot1 = scatter!(ax, projection(points2_rot, axis), color=:green, markersize=5, transparency=true)
    plot2 = scatter!(ax, projection(points2_rot1, axis), color=:red, markersize=5, transparency=true)

    Legend(fig[1, 2], [plot1, plot2], ["α-actinin", "Desmin"])

    ax2 = Axis(fig[2, :], xlabel="Normal direction (nm)", ylabel="Relative localization counts")
    hist1 = hist!(ax2, map(x -> getindex(x, 2), projection(points2_rot, 3)), bins=100, color=(:green, 0.5), normalization=:pdf)
    hist2 = hist!(ax2, map(x -> getindex(x, 2), projection(points2_rot1, 3)), bins=100, color=(:red, 0.5), normalization=:pdf)


    current_figure()

end
##
let
    fig = Figure()
    ax = Axis(fig[1, 1], aspect=1)
    points = to_abstractmatrix(point_clouds[2])
    clusters = dbscan(points, 240, min_cluster_size=1000)

    clusters_coords = [cluster_coords(point_clouds[2], c) for c in clusters]
    avg_orient = map(pca_orientation, clusters_coords) |> mean

    points2_rot = Pointcloud.rotate(point_clouds[2], -atan(avg_orient[2], avg_orient[1]))
    points2_rot1 = Pointcloud.rotate(point_clouds[1], -atan(avg_orient[2], avg_orient[1]))
    print(size(to_abstractmatrix(points2_rot)))
    plot1 = scatter!(ax, to_abstractmatrix(points2_rot)[2, :], df_list[2].z, color=:green, markersize=5, transparency=true)
    plot2 = scatter!(ax, to_abstractmatrix(points2_rot1)[2, :], df_list[1].z, color=:red, markersize=5, transparency=true)

    Legend(fig[1, 2], [plot1, plot2], ["α-actinin", "Desmin"])
    current_figure()

end

##
let
    points = to_abstractmatrix(point_clouds[2])
    clusters = dbscan(points, 240, min_cluster_size=1000)

    clusters_coords = [cluster_coords(point_clouds[2], c) for c in clusters]
    avg_orient = map(pca_orientation, clusters_coords) |> mean
    points2_rot = Pointcloud.rotate(point_clouds[2], -atan(avg_orient[2], avg_orient[1]))
    points2_rot1 = Pointcloud.rotate(point_clouds[1], -atan(avg_orient[2], avg_orient[1]))

    fig = Figure()
    ax = Axis(fig[1, 1])
    plot1 = hist!(ax, to_abstractmatrix(points2_rot)[2, :], bins=100, color=(:green, 0.5), normalization=:pdf)

    plot2 = hist!(ax, to_abstractmatrix(points2_rot1)[2, :], color=(:red, 0.5), bins=100, normalization=:pdf)
    Legend(fig[1, 2], [plot1, plot2], ["α-actinin", "Desmin"])
    current_figure()
end
##
scatter(point_clouds[2])
##
tree = KDTree(point_clouds[2])
##
bins, angle_counts = K(Float32(π / 180), 1000.0f0, tree, 1000 * 2000)
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

    point_to_set_hist = fit(Histogram, point_to_set_dists, r, closed=:left)
    return r[1:end-1], point_to_set_hist.weights
end
##
function region_to_point_to_set_distances(x_limit::Tuple{N,N}, y_limit::Tuple{N,N}, localization_df::DataFrame; set_index, point_index) where {N<:Real}

    df_list = split_dataframe_per_probe(localization_df)
    total_points = pointcloud_from_dataframe.(df_list)
    point_clouds = map(x -> subset_region(x, x_limit, y_limit), total_points)

    set_tree = KDTree(point_clouds[set_index])
    _, point_to_set_dist = nn(set_tree, point_clouds[point_index])
    return point_to_set_dist
end

function region_to_point_to_set_distances(x_limit::Tuple{N,N}, y_limit::Tuple{N,N}, z_limit::Tuple{N,N}, localization_df::DataFrame; set_index, point_index) where {N<:Real}

    df_list = split_dataframe_per_probe(localization_df)
    total_points = pointcloud_from_dataframe.(df_list)
    point_clouds = map(x -> subset_region(x, x_limit, y_limit, z_limit), total_points)

    set_tree = KDTree(point_clouds[set_index])
    _, point_to_set_dist = nn(set_tree, point_clouds[point_index])
    return point_to_set_dist
end

function region_to_point_to_set_distances(x_limit::Tuple{N,N}, y_limit::Tuple{N,N}, z_limit::Tuple{N,N}, point_clouds::Dict{String,V}; set_index::String, point_index::String) where {N<:Real,V<:Vector}

    points_roi = Dict(k => Pointcloud.subset_region(v, x_limit, y_limit, z_limit) for (k, v) in point_clouds)

    set_tree = KDTree(points_roi[set_index])
    _, point_to_set_dist = nn(set_tree, points_roi[point_index])
    return point_to_set_dist
end
##
function point_to_set_correlation(point_to_set_distances::Vector{N}, bin_size::N) where {N<:Number}
    distances = collect(0.0:bin_size:1300)
    h = fit(Histogram, point_to_set_distances, distances, closed=:left) |> x -> normalize(x, mode=:pdf)
    return h.weights
end
##
xlim, ylim, zlim = (12300, 16300), (12000, 16000), (-220, 380)
import LinearAlgebra: normalize
bin_size = 10.0f0
point_set_dist = region_to_point_to_set_distances(xlim, ylim, zlim, filtered_points, set_index="actinin", point_index="desmin")
distances = collect(0.0:bin_size:(maximum(point_set_dist)+bin_size))
h = point_to_set_correlation(point_set_dist, 10.0f0)
##


let
    fig = Figure(resolution=(1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title="Point(desmin)-to-Set(actin) correlation function: Δr = $(bin_size)nm; x ∈ $(Tuple(xlim./1000)) μm; y ∈ $(Tuple(ylim./1000)) μm", xlabel="Distance (nm)", ylabel="Correlation (a.u.)")
    #ax2 = GLMakie.Axis(fig[1, 2], yscale = log10, title = "Point(desmin)-to-Set(α-actinin) correlation function (Δr = $(bin_size)nm)", xlabel = "Distance (pixels)", ylabel = "Correlation (a.u.)")
    distances = collect(0.0:bin_size:1300)
    lines!(ax1, distances[1:end-1], h)
    ax1.xticks = LinearTicks(10)
    #scatterlines!(ax2, distances[1:end-1], h.weights)
    fig
end
##
#x_lims = [(7500, 12500), (17900, 22900), (18500, 23500), (25100, 30100), (32500, 37500)]
#y_lims = [(4700, 9700), (31850, 36850), (5800, 10800), (9000, 14000), (10300, 15300)]
x_lims = [(1500, 6500), (6700, 11700), (12000, 17000), (14300, 19300), (11800, 16800), (7200, 12200), (800, 5800)]
y_lims = [(500, 5500), (2000, 7000), (1500, 6500), (7000, 12000), (12500, 17500), (7300, 12300), (6500, 11500)]
z_lims = [(-200, 400), (-200, 400), (-200, 400), (-150, 450), (-150, 450), (-180, 420), (-240, 360)]

let
    using Makie.GeometryBasics
    fig = Figure(resolution=(900, 800))
    gl = fig[1, 1] = GridLayout()
    ax1 = Axis(gl[1, 1], aspect=1, xlabel="x (μm)", ylabel="y (μm)", title="Region selection for the desmin-α-actinin stack")

    for (name, points) in filtered_points
        scatter!(ax1, points / 1000, color=(color_dict[name], 0.5), markersize=2, transparency=true)
    end

    rects = []
    for (x_lim, y_lim) in zip(x_lims, y_lims)
        rect = poly!(ax1, Rect(x_lim[1] / 1000, y_lim[1] / 1000, 5, 5))
        push!(rects, rect)
    end
    Legend(gl[1, 2], rects, ["ROI #$(i)" for i in 1:length(rects)])
    #save("./actinin_roi_6.png", fig)
    rowgap!(gl, 5)
    fig
end
##
point_set_dist_list = @showprogress [region_to_point_to_set_distances(x_lim, y_lim, z_lim, filtered_points, set_index="actinin", point_index="desmin") for (x_lim, y_lim, z_lim) in zip(x_lims, y_lims, z_lims)]
##
bin_size = 10.0f0
p_to_s_corr_list = @showprogress [point_to_set_correlation(p_t_s_dist, bin_size) for p_t_s_dist in point_set_dist_list]
##
import Statistics: mean, std
means_ = [mean(output) for output in zip(p_to_s_corr_list...)]
stds_ = [std(output) for output in zip(p_to_s_corr_list...)]

##
let
    fig = Figure(resolution=(1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title="Point(desmin)-to-Set(α-actinin) correlation function: Δr = $(bin_size)nm", xlabel="Distance (nm)", ylabel="Correlation (a.u.)")
    #ax2 = GLMakie.Axis(fig[1, 2], yscale = log10, title = "Point(desmin)-to-Set(α-actinin) correlation function (Δr = $(bin_size)nm)", xlabel = "Distance (pixels)", ylabel = "Correlation (a.u.)")
    r = collect(0.0:bin_size:1300)[1:end-1]
    plots = [lines!(ax1, r, corr) for corr in p_to_s_corr_list]
    ax1.xticks = LinearTicks(10)
    Legend(fig[1, 2], plots, ["ROI #$(i)" for i in 1:length(plots)])
    #scatterlines!(ax2, distances[1:end-1], h.weights)
    fig
end
##
total_dists = cat(point_set_dist_list..., dims=1)
total_corr = point_to_set_correlation(total_dists, bin_size)
let
    fig = Figure(resolution=(1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title="Point(desmin)-to-Set(α-actinin) correlation function: Δr = $(bin_size)nm", xlabel="Distance (nm)", ylabel="Correlation (a.u.)")
    #ax2 = GLMakie.Axis(fig[1, 2], yscale = log10, title = "Point(desmin)-to-Set(α-actinin) correlation function (Δr = $(bin_size)nm)", xlabel = "Distance (pixels)", ylabel = "Correlation (a.u.)")
    r = collect(0.0:bin_size:1300)[1:end-1]

    band!(ax1, r, means_ - stds_, means_ + stds_, color=:lightgray, transparancy=:true)
    lines!(ax1, r, means_, color=:black)
    ax1.xticks = LinearTicks(10)
    #scatterlines!(ax2, distances[1:end-1], h.weights)
    fig
end
##
