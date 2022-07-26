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
##
let
    points = filtered_points_from_path(path_actinin, ["desmin", "actinin"])
    #points = points_from_path(path_actinin, ["desmin", "actinin"])

    x_lims = [(1.5, 6.5), (6.7, 11.7), (12.0, 17.0), (14.3, 19.3), (11.8, 16.8), (7.2, 12.2), (0.8, 5.8)]
    y_lims = [(0.5, 5.5), (2.0, 7.0), (1.5, 6.5), (7.0, 12.0), (12.5, 17.5), (7.3, 12.3), (6.5, 11.5)]
    z_lims = [(-0.2, 0.4), (-0.2, 0.4), (-0.2, 0.4), (-0.15, 0.45), (-0.15, 0.45), (-0.18, 0.42), (-0.24, 0.36)]

    ind = 7
    roi = SimpleROI(x_lims[ind], y_lims[ind], z_lims[ind])
    #roi = SimpleROI((5.0, 15.0), (5.0, 15.0), (-0.2, 0.4))
    points_roi = Dict(k => Pointcloud.subset_region(v, roi.lims...) for (k, v) in points)

    desmin2d = Pointcloud.projection(points_roi["desmin"], 3)
    actinin2d = Pointcloud.projection(points_roi["actinin"], 3)



    fig = Figure(resolution=(800, 600))
    gl = fig[1, 1] = GridLayout()
    ax = Axis(gl[1, 1], xlabel="θ (rad)", ylabel="H(θ)", title="Angular Ripleys function: r = 2.0μm, x ∈ $(x_lims[ind]), y ∈ $(y_lims[ind])")

    for (name, point_cloud, color) in zip(["desmin", "α-actinin"], [desmin2d, actinin2d], [:red, :green])
        angles, H = H_angular(point_cloud, 2.0, π / 120, x_lims[ind], y_lims[ind])
        lines!(ax, angles, H, label=name, color=color)
    end

    ax.xticks = 0:π/4:2π
    ax.xtickformat = xs -> ["$(x/pi)π" for x in xs]
    gl[1, 2] = Legend(fig, ax)

    #save("./angular_ripley_$(ind).png", fig)
    fig
end
##
### Code to generate Figure 3a
let
    import Statistics: std
    points = filtered_points_from_path(path_actinin, ["desmin", "actinin"])
    #points = points_from_path(path_actinin, ["desmin", "actinin"])

    x_lims = [(1.5, 6.5), (6.7, 11.7), (12.0, 17.0), (14.3, 19.3), (11.8, 16.8), (7.2, 12.2), (0.8, 5.8)]
    y_lims = [(0.5, 5.5), (2.0, 7.0), (1.5, 6.5), (7.0, 12.0), (12.5, 17.5), (7.3, 12.3), (6.5, 11.5)]
    z_lims = [(-0.2, 0.4), (-0.2, 0.4), (-0.2, 0.4), (-0.15, 0.45), (-0.15, 0.45), (-0.18, 0.42), (-0.24, 0.36)]

    function ind_to_H(point_cloud, ind)
        points_roi = Pointcloud.subset_region(point_cloud, x_lims[ind], y_lims[ind], z_lims[ind])
        points2d = Pointcloud.projection(points_roi, 3)

        _, H = H_angular(points2d, 2.0, π / 120, x_lims[ind], y_lims[ind])
        return H
    end

    angles = collect(0:π/120:2π)[1:end-1]

    H_desmin_list = @showprogress [ind_to_H(points["desmin"], i) for i in 1:7]
    H_actinin_list = @showprogress [ind_to_H(points["actinin"], i) for i in 1:7]

    function find_align_index(angles, H_actinin)
        H_actinin_half = @view H_actinin[angles.<=π]
        _, max_ind = findmax(H_actinin_half)
        return max_ind
    end

    for (H_des, H_act) in zip(H_desmin_list, H_actinin_list)
        max_ind = find_align_index(angles, H_act)
        H_des .= circshift(H_des, -max_ind)
        H_act .= circshift(H_act, -max_ind)
    end

    fig = Figure(resolution=(800, 400))
    gl = fig[1, 1] = GridLayout()
    ax = Axis(gl[1, 1], xlabel="θ (rad)", ylabel="H(θ)", title="Averaged 2D angular Ripleys function: r = 2.0μm")

    for (name, H_list, color) in zip(["desmin", "α-actinin"], [H_desmin_list, H_actinin_list], [:red, :green])
        means_ = [mean(output) for output in zip(H_list...)]
        stds_ = [std(output) for output in zip(H_list...)]

        band!(ax, angles, means_ - stds_, means_ + stds_, color=(color, 0.25), transparancy=:true)
        lines!(ax, angles, means_, color=(color, 0.75), label=name, linewidth=3)
    end

    ax.xticks = 0:π/4:2π
    ax.xtickformat = xs -> ["$(x/pi)π" for x in xs]
    gl[1, 2] = Legend(fig, ax)

    #save("./Figures/Figure 4a.png", fig)
    fig
end
##
let
    path = "./Data/STORM/desmin_alphaactinin_2.5um.csv"

    points_list = path |> CSV.File |> DataFrame |> split_dataframe_per_probe
    points_list = pointcloud_from_dataframe.(points_list)
    points_dict = Dict(zip(["desmin", "actinin"], points_list))
    points_dict = Dict(k => Pointcloud.subset_region(v, (0.0, 40000.0), (0.0, 40000.0), (-1200.0, 0.0)) for (k, v) in points_dict)

    fig = Figure(resolution=(1800, 800))
    axis_kwargs = Dict(:aspect => 1, :xlabel => "x (μm)", :ylabel => "y (μm)")
    ax1 = Axis(fig[1, 1]; title="Raw Overlay", axis_kwargs...)
    ax2 = Axis(fig[1, 2]; title="Filtered Overlay", axis_kwargs...)
    ax3 = Axis(fig[1, 3]; title="Noise Overlay", axis_kwargs...)

    linkxaxes!(ax1, ax2, ax3)
    linkyaxes!(ax1, ax2, ax3)

    scatter_kwargs = Dict{Symbol,Any}(:markersize => 1, :transparency => true)
    for (name, points) in points_dict
        color = color_dict[name]
        scatter_kwargs[:color] = (color, 0.5)
        scatter!(ax1, points / 1000; scatter_kwargs...)

        acc, rej = Ripley.filter_points(points, 200, 5) #240, 5
        scatter!(ax2, acc / 1000; scatter_kwargs...)
        scatter!(ax3, rej / 1000; scatter_kwargs...)
    end
    #save("./Supplementary fig 1-1. Denoising results for desmin-α-actinin 600nm (overlay).png", fig)
    fig
end
##
let
    path = "./Data/STORM/desmin_alphaactinin_2.5um.csv"
    points_list = path |> CSV.File |> DataFrame |> split_dataframe_per_probe
    points_list = pointcloud_from_dataframe.(points_list)
    points_dict = Dict(zip(["desmin", "actinin"], points_list))
    points_dict = Dict(k => Pointcloud.subset_region(v, (0.0, 40000.0), (0.0, 40000.0), (-2500.0, 0.0)) for (k, v) in points_dict)

    fig = Figure(resolution=(1800, 600))
    kwargs = Dict(:aspect => (1, 1, 0.25), :xlabel => "x (μm)", :ylabel => "y (μm)", :zlabel => "z (μm)", :elevation => 0.2π, :azimuth => 0.2π)
    ax1 = Axis3(fig[1, 1]; title="Raw Overlay", kwargs...)
    ax2 = Axis3(fig[1, 2]; title="Filtered Overlay", kwargs...)
    ax3 = Axis3(fig[1, 3]; title="Noise Overlay", kwargs...)


    scatter_kwargs = Dict{Symbol,Any}(:markersize => 400, :transparency => true)
    for (name, points) in points_dict
        color = color_dict[name]
        scatter_kwargs[:color] = (color, 0.5)
        scatter!(ax1, points / 1000; scatter_kwargs...)

        acc, rej = Ripley.filter_points(points, 200, 5) #240, 5
        scatter!(ax2, acc / 1000; scatter_kwargs...)
        scatter!(ax3, rej / 1000; scatter_kwargs...)
    end
    #save("./Supplementary fig 1-2. Denoising results for desmin-α-actinin 600nm (3D).png", fig)
    fig
end
##
let
    #path = "./Data/STORM/actin_desmin_1.2um.csv"
    path = "./Data/STORM/desmin_alphaactinin_2.5um.csv"
    points = filtered_points_from_path(path, ["desmin", "actinin"])

    points = Dict(k => Pointcloud.subset_region(v, (0.0, 40.0), (0.0, 40.0), (-2.5, 0.0)) for (k, v) in points)

    fig = Figure(resolution=(1200, 800))
    kwargs = Dict(:aspect => 1, :xlabel => "x (μm)", :ylabel => "y (μm)")
    scatter_kwargs = Dict{Symbol,Any}(:markersize => 2, :transparency => true)

    x_lim, y_lim = (20.0, 30.0), (5.0, 15.0)
    gl = fig[1, 1] = GridLayout()

    for (i, z_lim) in enumerate([(-1.5, -0.9), (-1.8, -0.6), (-2.5, 0.0)])
        points_roi = Dict(k => Pointcloud.subset_region(v, x_lim, y_lim, z_lim) for (k, v) in points)
        points_2d = Dict(k => Pointcloud.projection(v, 3) for (k, v) in points_roi)
        z_coords = Dict(k => map(x -> x[3], v) for (k, v) in points_roi)
        
        ax_desmin = Axis(gl[1, i]; title = "$(round(z_lim[2]-z_lim[1], digits = 1))μm slice", kwargs...)
        ax_actinin = Axis(gl[2, i]; kwargs...)

        sc1 = scatter!(ax_desmin, points_2d["desmin"], color = z_coords["desmin"], colormap = :linear_kry_0_97_c73_n256, colorrange = (-2.5, 0.0); scatter_kwargs...)
        sc2 = scatter!(ax_actinin, points_2d["actinin"], color = z_coords["actinin"], colormap = :linear_kgy_5_95_c69_n256, colorrange = (-2.5, 0.0); scatter_kwargs...)
        
        hide_kwargs = Dict(:grid => false)
        hidedecorations!(ax_desmin; hide_kwargs...)
        if i != 1
            hidedecorations!(ax_actinin; hide_kwargs...)
        else
            Colorbar(gl[1, 4], sc1, label = "desmin z (μm)")
            Colorbar(gl[2, 4], sc2, label = "α-actinin z (μm)")
        end

        for ax_ in [ax_desmin, ax_actinin]
            ax_.xticks = x_lim[1]:2.5:x_lim[2]
            ax_.yticks = y_lim[1]:2.5:y_lim[2]
        end

    end
    
    colgap!(gl, 15)
    rowgap!(gl, 15)
    #save("./Figures/Figure 3b.png", fig)
    fig
end
##
let
    #path = "./Data/STORM/actin_desmin_1.2um.csv"
    path = "./Data/STORM/desmin_alphaactinin_2.5um.csv"
    points = filtered_points_from_path(path, ["desmin", "actinin"])

    points = Dict(k => Pointcloud.subset_region(v, (0.0, 40.0), (0.0, 40.0), (-2.0, 0.5)) for (k, v) in points)

    fig = Figure(resolution=(1200, 800))
    kwargs = Dict(:aspect => (1, 1, 0.3), :xlabel => "x (μm)", :ylabel => "y (μm)", :zlabel => "z (μm)")
    scatter_kwargs = Dict{Symbol,Any}(:markersize => 400, :transparency => true)

    x_lim, y_lim = (20.0, 30.0), (5.0, 15.0)
    for (i, z_lim) in enumerate([(-1.5, -0.9), (-1.8, -0.6), (-2.3, 0.2)])
        desmin_points = Pointcloud.subset_region(points["desmin"], x_lim, y_lim, z_lim)
        actinin_points = Pointcloud.subset_region(points["actinin"], x_lim, y_lim, z_lim)


        ax_desmin = Axis3(fig[1, i]; kwargs...)
        ax_actinin = Axis3(fig[2, i]; kwargs...)

        scatter!(ax_desmin, desmin_points, color = (color_dict["desmin"], 0.3); scatter_kwargs...)
        scatter!(ax_actinin, actinin_points, color = (color_dict["actinin"], 0.3); scatter_kwargs...)

    end

    fig
end
##
let
    path = "./Data/STORM/desmin_alphaactinin_2.5um.csv"
    points = filtered_points_from_path(path, ["desmin", "actinin"])

    fig = Figure(resolution=(1500, 400))

    x_lim, y_lim = (20.0, 30.0), (4.0, 14.0)
    for (i, z_lim) in enumerate([(-1.5, -0.9), (-1.8, -0.6), (-2.5, 0.0)])
        points_roi = Dict(k => Pointcloud.subset_region(v, x_lim, y_lim, z_lim) for (k, v) in points)

        desmin2d = Pointcloud.projection(points_roi["desmin"], 3)
        actinin2d = Pointcloud.projection(points_roi["actinin"], 3)

        gl = fig[1, i] = GridLayout()
        ax = Axis(gl[1, 1], xlabel="θ (rad)", ylabel="H(θ)", title="Angular Ripleys function: r = 2.0μm, z ∈ $(z_lim)")

        for (name, point_cloud, color) in zip(["desmin", "α-actinin"], [desmin2d, actinin2d], [:red, :green])
            angles, H = H_angular(point_cloud, 2.0, π / 120, x_lim, y_lim)
            lines!(ax, angles, H, label=name, color=color)
        end
        ax.xticks = 0:π/4:2π
        ax.xtickformat = xs -> ["$(x/pi)π" for x in xs]
     end

    #save("./angular_ripley_$(ind).png", fig)
    fig
end
