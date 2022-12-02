##
using Revise
using GLMakie
import Pkg
Pkg.activate(".")

import CSV
import DataFrames: DataFrame
using Makie.GeometryBasics
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

function filtered_points_from_path(filepath::String, channel_names::Vector{String})
    points_list = filepath |> CSV.File |> DataFrame |> split_dataframe_per_probe
    points_list = pointcloud_from_dataframe.(points_list)
    points = Dict(zip(channel_names, points_list))
    filtered_points = Dict(k => Ripley.filter_points(v, 200, 5)[1] / 1000 for (k, v) in points)
    return filtered_points
end
##
let
    points_dict = filtered_points_from_path(path_actinin, ["desmin", "actinin"])

    x_lims = [(1.5, 6.5), (6.7, 11.7), (12.0, 17.0), (14.3, 19.3), (11.8, 16.8), (7.2, 12.2), (0.8, 5.8)]
    y_lims = [(0.5, 5.5), (2.0, 7.0), (1.5, 6.5), (7.0, 12.0), (12.5, 17.5), (7.3, 12.3), (6.5, 11.5)]
    z_lims = [(-0.2, 0.4), (-0.2, 0.4), (-0.2, 0.4), (-0.15, 0.45), (-0.15, 0.45), (-0.18, 0.42), (-0.24, 0.36)]


    fig = Figure(resolution=(900, 800))
    gl = fig[1, 1] = GridLayout()
    ax1 = Axis(gl[1, 1], aspect=1, xlabel="x (μm)", ylabel="y (μm)", title="Region selection for the desmin-α-actinin stack")

    for (name, points) in points_dict
        scatter!(ax1, points, color=(color_dict[name], 0.5), markersize=2, transparency=true)
    end

    rects = []
    for (x_lim, y_lim) in zip(x_lims, y_lims)
        rect = poly!(ax1, Rect(x_lim[1], y_lim[1], 5, 5), color=:transparent, strokewidth=2.0)
        push!(rects, rect)
    end
    Legend(gl[1, 2], rects, ["ROI #$(i)" for i in 1:length(rects)])
    rowgap!(gl, 5)
    #save("./Supplementary fig 2_actinin_roi_overview.png", fig)
    fig
end
##
let
    points_dict = filtered_points_from_path(path_actinin, ["desmin", "actinin"])

    x_lims = [(1.5, 6.5), (6.7, 11.7), (12.0, 17.0), (14.3, 19.3), (11.8, 16.8), (7.2, 12.2), (0.8, 5.8)]
    y_lims = [(0.5, 5.5), (2.0, 7.0), (1.5, 6.5), (7.0, 12.0), (12.5, 17.5), (7.3, 12.3), (6.5, 11.5)]
    z_lims = [(-0.2, 0.4), (-0.2, 0.4), (-0.2, 0.4), (-0.15, 0.45), (-0.15, 0.45), (-0.18, 0.42), (-0.24, 0.36)]


    for (i, (xlim, ylim, zlim)) in enumerate(zip(x_lims, y_lims, z_lims))
        roi_points = Dict(k => Pointcloud.subset_region(v, xlim, ylim, zlim) for (k, v) in points_dict)

        fig = Figure(resolution=(1500, 800))
        gl = fig[1, 1] = GridLayout()

        ax1 = Axis(gl[1, 1], aspect=1, xlabel="x (μm)", ylabel="y (μm)", title="Overlay for for x ∈ $xlim, y ∈ $ylim, z ∈ $zlim")

        ax2 = Axis3(gl[1, 2], xlabel="x (μm)", ylabel="y (μm)", zlabel="z (μm)", aspect=(4, 4, 0.6), elevation=0.15π, azimuth=0.6π, alignmode=Outside(40))

        for (name, points) in roi_points
            scatter!(ax1, points, color=(color_dict[name], 0.5), markersize=2, transparency=true)

            scatter!(ax2, points, color=(color_dict[name], 0.5), markersize=600, transparency=true)
        end
        colgap!(gl, 5)
        #save("./Supplementary fig 2_actinin_roi_$(i).png", fig)
    end

end
##
let
    points_dict = filtered_points_from_path(path_actin, ["actin", "desmin"])

    x_lims = [(2.5, 7.5), (8.0, 13.0), (13.5, 18.5), (13.5, 18.5), (7.0, 12.0), (13.5, 18.5)]
    y_lims = [(1.0, 6.0), (1.0, 6.0), (1.0, 6.0), (6.5, 11.5), (6.5, 11.5), (12.0, 17.0)]
    z_lims = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]


    fig = Figure(resolution=(900, 800))
    gl = fig[1, 1] = GridLayout()
    ax1 = Axis(gl[1, 1], aspect=1, xlabel="x (μm)", ylabel="y (μm)", title="Region selection for the actin-desmin stack")

    for (name, points) in points_dict
        scatter!(ax1, points, color=(color_dict[name], 0.5), markersize=2, transparency=true)
    end

    rects = []
    for (x_lim, y_lim) in zip(x_lims, y_lims)
        rect = poly!(ax1, Rect(x_lim[1], y_lim[1], 5, 5), color=:transparent, strokecolor=:grey30, strokewidth=2.0)
        push!(rects, rect)
    end
    Legend(gl[1, 2], rects, ["ROI #$(i)" for i in 1:length(rects)])
    rowgap!(gl, 5)
    #save("./Supplementary fig 2_actin_roi_overview.png", fig)
    fig
end
##
let
    points_dict = filtered_points_from_path(path_actin, ["actin", "desmin"])

    x_lims = [(2.5, 7.5), (8.0, 13.0), (13.5, 18.5), (13.5, 18.5), (7.0, 12.0), (13.5, 18.5)]
    y_lims = [(1.0, 6.0), (1.0, 6.0), (1.0, 6.0), (6.5, 11.5), (6.5, 11.5), (12.0, 17.0)]
    z_lims = [(-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3)]

    for (i, (xlim, ylim, zlim)) in enumerate(zip(x_lims, y_lims, z_lims))
        roi_points = Dict(k => Pointcloud.subset_region(v, xlim, ylim, zlim) for (k, v) in points_dict)

        fig = Figure(resolution=(1500, 800))
        gl = fig[1, 1] = GridLayout()

        ax1 = Axis(gl[1, 1], aspect=1, xlabel="x (μm)", ylabel="y (μm)", title="Overlay for for x ∈ $xlim, y ∈ $ylim, z ∈ $zlim")

        ax2 = Axis3(gl[1, 2], xlabel="x (μm)", ylabel="y (μm)", zlabel="z (μm)", aspect=(4, 4, 0.6), elevation=0.15π, azimuth=0.6π, alignmode=Outside(40))

        for (name, points) in roi_points
            scatter!(ax1, points, color=(color_dict[name], 0.5), markersize=2, transparency=true)

            scatter!(ax2, points, color=(color_dict[name], 0.5), markersize=600, transparency=true)
        end
        colgap!(gl, 5)
        save("./Supplementary fig 2_actin_roi_$(i).png", fig)
    end

end