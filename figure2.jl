##
using Revise
using GLMakie
import Pkg
Pkg.activate(".")
##
import CSV
import DataFrames: DataFrame
using NearestNeighbors
import StatsBase: fit, Histogram
import Statistics: mean, std
import LinearAlgebra: normalize
using ProgressMeter

include("ripley.jl")
using .Ripley
include("pointcloud.jl")
using .Pointcloud
include("hdbscan.jl")
using .HDBSCAN
include("utils.jl")

const path_actinin = "./Data/STORM/desmin_alphaactinin_600nm.csv"
const path_actin = "./Data/STORM/actin_desmin_600nm.csv"
const color_dict = Dict(zip(["desmin", "actinin", "actin"], [:red, :green, :blue]))

function points_from_path(filepath::String, channel_names::Vector{String})
    points_list = filepath |> CSV.File |> DataFrame |> split_dataframe_per_probe
    points_list = pointcloud_from_dataframe.(points_list)
    points = Dict(zip(channel_names, points_list))
    return Dict(k => v / 1000 for (k, v) in points)
end

function filtered_points_from_path(filepath::String, channel_names::Vector{String})
    points_list = filepath |> CSV.File |> DataFrame |> split_dataframe_per_probe
    points_list = pointcloud_from_dataframe.(points_list)
    points = Dict(zip(channel_names, points_list))
    filtered_points = Dict(k => Ripley.filter_points(v, 200, 5)[1] / 1000 for (k, v) in points)
    return filtered_points
end

function region_to_point_to_set_distances(x_limit::Tuple{N,N}, y_limit::Tuple{N,N}, z_limit::Tuple{N,N}, point_clouds::Dict{String,V}; set_index::String, point_index::String) where {N<:Real,V<:Vector}

    points_roi = Dict(k => Pointcloud.subset_region(v, x_limit, y_limit, z_limit) for (k, v) in point_clouds)

    set_tree = KDTree(points_roi[set_index])
    _, point_to_set_dist = nn(set_tree, points_roi[point_index])
    return point_to_set_dist
end

function point_to_set_correlation(point_to_set_distances::Vector{N}, bin_size::N) where {N<:Number}
    distances = collect(0.0:bin_size:1000)
    h = fit(Histogram, point_to_set_distances, distances, closed=:left) |> x -> normalize(x, mode=:pdf)
    return h.weights
end
##
let
    ## Code to generate Figure 3a: 600nm Desmin & α-actinin stain overlay
    points = filtered_points_from_path(path_actinin, ["desmin", "actinin"])
    points = Dict(k => Pointcloud.projection(v, 3) for (k, v) in points)

    fig = Figure(resolution=(1500, 500))

    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    gl3 = fig[1, 3] = GridLayout()

    axis_kwargs = Dict(:aspect => 1, :xlabel => "x (μm)", :ylabel => "y (μm)")
    ax1 = Axis(gl1[1, 1]; title="Desmin", axis_kwargs...)
    ax2 = Axis(gl2[1, 1]; title="α-actinin", axis_kwargs...)
    ax3 = Axis(gl3[1, 1]; title="Combined", axis_kwargs...)

    scatter_kwargs = Dict(:markersize => 2)
    scatter!(ax1, points["desmin"]; color=(:red, 0.3), scatter_kwargs...)
    scatter!(ax2, points["actinin"]; color=(:green, 0.3), scatter_kwargs...)
    scatter!(ax3, points["desmin"]; color=(:red, 0.3), scatter_kwargs...)
    scatter!(ax3, points["actinin"]; color=(:green, 0.3), scatter_kwargs...)

    #save("./Figures/Figure 3a.png", fig)
    fig
end
##
let
    ## Code to generate Figure 3c: 600nm actin & desmin stain overlay
    points = filtered_points_from_path(path_actin, ["actin", "desmin"])
    points = Dict(k => Pointcloud.projection(v, 3) for (k, v) in points)

    fig = Figure(resolution=(1500, 500))

    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    gl3 = fig[1, 3] = GridLayout()

    axis_kwargs = Dict(:aspect => 1, :xlabel => "x (μm)", :ylabel => "y (μm)")
    ax1 = Axis(gl1[1, 1]; title="Desmin", axis_kwargs...)
    ax2 = Axis(gl2[1, 1]; title="Actin", axis_kwargs...)
    ax3 = Axis(gl3[1, 1]; title="Combined", axis_kwargs...)

    scatter_kwargs = Dict(:markersize => 2)
    scatter!(ax1, points["desmin"]; color=(:red, 0.3), scatter_kwargs...)
    scatter!(ax2, points["actin"]; color=(:blue, 0.03), scatter_kwargs...)
    scatter!(ax3, points["desmin"]; color=(:red, 0.3), scatter_kwargs...)
    scatter!(ax3, points["actin"]; color=(:blue, 0.03), scatter_kwargs...)

    save("./Figures/Figure 3b.png", fig)
    fig
end
##

let
    #points = filtered_points_from_path(path_actinin, ["desmin", "actinin"])
    points = points_from_path(path_actinin, ["desmin", "actinin"])

    x_lims = [(1.5, 6.5), (6.7, 11.7), (12.0, 17.0), (14.3, 19.3), (11.8, 16.8), (7.2, 12.2), (0.8, 5.8)]
    y_lims = [(0.5, 5.5), (2.0, 7.0), (1.5, 6.5), (7.0, 12.0), (12.5, 17.5), (7.3, 12.3), (6.5, 11.5)]
    z_lims = [(-0.2, 0.4), (-0.2, 0.4), (-0.2, 0.4), (-0.15, 0.45), (-0.15, 0.45), (-0.18, 0.42), (-0.24, 0.36)]

    ind = 2
    roi = SimpleROI(x_lims[ind], y_lims[ind], z_lims[ind])
    #roi = SimpleROI((5.0, 15.0), (5.0, 15.0), (-0.2, 0.4))
    points_roi = Dict(k => Pointcloud.subset_region(v, roi.lims...) for (k, v) in points)

    desmin_inner, L = GetisFranklin_L(points_roi["desmin"], points_roi["desmin"], 0.2, roi.lims...)
    desmin_inner, L_cross = GetisFranklin_L(points_roi["desmin"], points_roi["actinin"], 0.2, roi.lims...)

    fig = Figure(resolution=(1400, 800))
    ax = Axis(fig[1, 1], aspect=1)
    #sc = scatter!(ax, points, color = to_abstractmatrix(points)[:,3], markersize = 1000)
    desmin2d = Pointcloud.projection(desmin_inner, 3)
    actinin2d = Pointcloud.projection(points_roi["actinin"], 3)

    ax2 = Axis(fig[1, 2], aspect=1)
    scatter!(ax, desmin2d, color=(:red, 0.3), markersize=4)
    scatter!(ax, actinin2d, color=(:green, 0.05), markersize=3)

    L_max = max(maximum(L), maximum(L_cross))
    clims = extrema(L)
    scatter!(ax2, desmin2d, color=L, markersize=4, colormap=:Reds, colorrange=clims)
    scatter!(ax2, actinin2d, color=(:green, 0.05), markersize=3)


    ax3 = Axis(fig[1, 3], aspect=1)
    scatter!(ax3, desmin2d, color=L_cross, markersize=4, colormap=:Reds, colorrange=clims)
    scatter!(ax3, actinin2d, color=(:green, 0.05), markersize=3)
    #Colorbar(fig[1, 2], sc)

    ax4 = Axis(fig[2, 1], aspect=1)
    scatter!(ax4, L, L_cross, markersize=4)

    ax5 = Axis(fig[2, 2])
    hist!(ax5, L, bins=30)
    limits!(ax4, 0, L_max, 0, L_max)

    ax5 = Axis(fig[2, 3])
    hist!(ax5, L_cross, bins=30)
    limits!(ax4, 0, L_max, 0, L_max)
    fig
end
##
let
    points = filtered_points_from_path(path_actinin, ["desmin", "actinin"])
    #points = points_from_path(path_actinin, ["desmin", "actinin"])

    x_lims = [(1.5, 6.5), (6.7, 11.7), (12.0, 17.0), (14.3, 19.3), (11.8, 16.8), (7.2, 12.2), (0.8, 5.8)]
    y_lims = [(0.5, 5.5), (2.0, 7.0), (1.5, 6.5), (7.0, 12.0), (12.5, 17.5), (7.3, 12.3), (6.5, 11.5)]
    z_lims = [(-0.2, 0.4), (-0.2, 0.4), (-0.2, 0.4), (-0.15, 0.45), (-0.15, 0.45), (-0.18, 0.42), (-0.24, 0.36)]

    ind = 4
    roi = SimpleROI(x_lims[ind], y_lims[ind], z_lims[ind])
    #roi = SimpleROI((5.0, 15.0), (5.0, 15.0), (-0.2, 0.4))
    points_roi = Dict(k => Pointcloud.subset_region(v, roi.lims...) for (k, v) in points)

    desmin_inner, L = GetisFranklin_L(points_roi["desmin"], points_roi["desmin"], 0.2, roi.lims...)
    desmin_inner, L_cross = GetisFranklin_L(points_roi["desmin"], points_roi["actinin"], 0.2, roi.lims...)

    fig = Figure(resolution=(1500, 500))
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    gl3 = fig[1, 3] = GridLayout()
    axis3_kwargs = Dict(:aspect => (1, 1, 0.3), :elevation => 0.25π, :azimuth => 0.6π)

    ax = Axis3(gl1[1, 1]; title="Overlay: x ∈ $(x_lims[ind]), y ∈ $(y_lims[ind])", axis3_kwargs...)

    ax2 = Axis3(gl2[1, 1]; title="Self clustering map", axis3_kwargs...)
    scatter!(ax, desmin_inner, color=(:red, 0.5), markersize=1200)

    L_max = max(maximum(L), maximum(L_cross))
    clims = (0, maximum(L))
    sc2 = scatter!(ax2, desmin_inner, color=L, markersize=1200, colormap=:Reds, colorrange=clims)
    Colorbar(gl2[2, 1], sc2, label=L"$L_{desmin}(r = 200)$", vertical=false, flipaxis=false)

    ax3 = Axis3(gl3[1, 1]; title="Colocalization map", axis3_kwargs...)
    sc3 = scatter!(ax3, desmin_inner, color=L_cross, markersize=1200, colormap=:Reds, colorrange=clims)
    Colorbar(gl3[2, 1], sc3, label=L"$L_{desmin/α-actinin}(r = 200)$", vertical=false, flipaxis=false)
    #Colorbar(fig[1, 2], sc)
    for ax_ in [ax, ax2, ax3]
        scatter!(ax_, points_roi["actinin"], color=(:green, 0.5), markersize=600)
        ax_.zticks = -0.3:0.3:0.3
    end
    save("./actinin_L_3D_$(ind).png", fig)
    fig
end
##
let
    points = filtered_points_from_path(path_actin, ["actin", "desmin"])

    x_lims = [(2.5, 7.5), (8.0, 13.0), (13.5, 18.5), (13.5, 18.5), (7.0, 12.0), (13.5, 18.5)]
    y_lims = [(1.0, 6.0), (1.0, 6.0), (1.0, 6.0), (6.5, 11.5), (6.5, 11.5), (12.0, 17.0)]
    z_lims = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]

    ind = 6
    roi = SimpleROI(x_lims[ind], y_lims[ind], z_lims[ind])
    #roi = SimpleROI((5.0, 15.0), (5.0, 15.0), (-0.2, 0.4))
    points_roi = Dict(k => Pointcloud.subset_region(v, roi.lims...) for (k, v) in points)

    desmin_inner, L = GetisFranklin_L(points_roi["desmin"], points_roi["desmin"], 0.2, roi.lims...)
    desmin_inner, L_cross = GetisFranklin_L(points_roi["desmin"], points_roi["actin"], 0.2, roi.lims...)

    fig = Figure(resolution=(1500, 500))
    gl1 = fig[1, 1] = GridLayout()
    gl2 = fig[1, 2] = GridLayout()
    gl3 = fig[1, 3] = GridLayout()
    axis3_kwargs = Dict(:aspect => (1, 1, 0.3), :elevation => 0.25π, :azimuth => 0.6π)

    ax = Axis3(gl1[1, 1]; title="Overlay: x ∈ $(x_lims[ind]), y ∈ $(y_lims[ind])", axis3_kwargs...)
    #sc = scatter!(ax, points, color = to_abstractmatrix(points)[:,3], markersize = 1000)

    ax2 = Axis3(gl2[1, 1]; title="Self clustering map", axis3_kwargs...)
    scatter!(ax, desmin_inner, color=(:red, 0.5), markersize=1200)

    L_max = max(maximum(L), maximum(L_cross))
    clims = (0, maximum(L))
    sc2 = scatter!(ax2, desmin_inner, color=L, markersize=1200, colormap=:Reds, colorrange=clims)
    Colorbar(gl2[2, 1], sc2, label=L"L(200)", vertical=false, flipaxis=false)

    ax3 = Axis3(gl3[1, 1]; title="Colocalization map", axis3_kwargs...)
    sc3 = scatter!(ax3, desmin_inner, color=L_cross, markersize=1200, colormap=:Reds, colorrange=clims)
    Colorbar(gl3[2, 1], sc3, label=L"L_{cross}(200)", vertical=false, flipaxis=false)
    #Colorbar(fig[1, 2], sc)
    for ax_ in [ax, ax2, ax3]
        scatter!(ax_, points_roi["actin"], color=(:blue, 0.3), markersize=600)
        ax_.zticks = -0.3:0.3:0.3
    end
    save("./actin_L_3D_$(ind).png", fig)
    fig
end
##
let
    points = filtered_points_from_path(path_actinin, ["desmin", "actinin"])

    x_lims = [(1.5, 6.5), (6.7, 11.7), (12.0, 17.0), (14.3, 19.3), (11.8, 16.8), (7.2, 12.2), (0.8, 5.8)]
    y_lims = [(0.5, 5.5), (2.0, 7.0), (1.5, 6.5), (7.0, 12.0), (12.5, 17.5), (7.3, 12.3), (6.5, 11.5)]
    z_lims = [(-0.2, 0.4), (-0.2, 0.4), (-0.2, 0.4), (-0.15, 0.45), (-0.15, 0.45), (-0.18, 0.42), (-0.24, 0.36)]

    point_set_dist_list = @showprogress [region_to_point_to_set_distances(x_lim, y_lim, z_lim, points, set_index="actinin", point_index="desmin") for (x_lim, y_lim, z_lim) in zip(x_lims, y_lims, z_lims)]

    bin_size = 10.0f0
    p_to_s_corr_list = @showprogress [point_to_set_correlation(p_t_s_dist * 1000, bin_size) for p_t_s_dist in point_set_dist_list]

    total_dists = cat(point_set_dist_list..., dims=1)
    total_corr = point_to_set_correlation(total_dists, bin_size)

    fig = Figure(resolution=(1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title="Point(desmin)-to-Set(α-actinin) correlation function: Δr = $(bin_size)nm", xlabel="Distance (nm)", ylabel="Correlation (a.u.)")

    r = collect(0.0:bin_size:1000)[1:end-1]

    means_ = [mean(output) for output in zip(p_to_s_corr_list...)]
    stds_ = [std(output) for output in zip(p_to_s_corr_list...)]

    band!(ax1, r, means_ - stds_, means_ + stds_, color=:lightgray, transparancy=:true)
    lines!(ax1, r, means_, color=:black)
    ax1.xticks = LinearTicks(10)
    #scatterlines!(ax2, distances[1:end-1], h.weights)

    print(r[findmax(means_)[2]])
    fig
    #save("./Fig 2_actinin_point2set.png", fig)
end
##
let
    filtered_points = filtered_points_from_path(path_actin, ["actin", "desmin"])

    x_lims = [(2.5, 7.5), (8.0, 13.0), (13.5, 18.5), (13.5, 18.5), (7.0, 12.0), (13.5, 18.5)]
    y_lims = [(1.0, 6.0), (1.0, 6.0), (1.0, 6.0), (6.5, 11.5), (6.5, 11.5), (12.0, 17.0)]
    z_lims = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]

    point_set_dist_list = @showprogress [region_to_point_to_set_distances(x_lim, y_lim, z_lim, filtered_points, set_index="actin", point_index="desmin") for (x_lim, y_lim, z_lim) in zip(x_lims, y_lims, z_lims)]

    bin_size = 10.0f0
    p_to_s_corr_list = @showprogress [point_to_set_correlation(p_t_s_dist * 1000, bin_size) for p_t_s_dist in point_set_dist_list]

    total_dists = cat(point_set_dist_list..., dims=1)
    total_corr = point_to_set_correlation(total_dists, bin_size)

    fig = Figure(resolution=(1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title="Point(desmin)-to-Set(actin) correlation function: Δr = $(bin_size)nm", xlabel="Distance (nm)", ylabel="Correlation (a.u.)")

    r = collect(0.0:bin_size:1000)[1:end-1]

    means_ = [mean(output) for output in zip(p_to_s_corr_list...)]
    stds_ = [std(output) for output in zip(p_to_s_corr_list...)]

    band!(ax1, r, means_ - stds_, means_ + stds_, color=:lightgray, transparancy=:true)
    lines!(ax1, r, means_, color=:black)
    ax1.xticks = LinearTicks(10)
    #scatterlines!(ax2, distances[1:end-1], h.weights)

    print(r[findmax(means_)[2]])
    fig
    #save("./Fig 2_actin_point2set.png", fig)
end
##