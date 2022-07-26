##
using Revise
using GLMakie
import Pkg
Pkg.activate(".")
##
import CSV
import DataFrames: DataFrame
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
##
let
    points_list = path_actinin |> CSV.File |> DataFrame |> split_dataframe_per_probe
    points_list = pointcloud_from_dataframe.(points_list)
    points_dict = Dict(zip(["desmin", "actinin"], points_list))

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
    save("./Supplementary fig 1-1. Denoising results for desmin-α-actinin 600nm (overlay).png", fig)
    fig
end
##
let
    points_list = path_actinin |> CSV.File |> DataFrame |> split_dataframe_per_probe
    points_list = pointcloud_from_dataframe.(points_list)
    points_dict = Dict(zip(["desmin", "actinin"], points_list))

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
    save("./Supplementary fig 1-2. Denoising results for desmin-α-actinin 600nm (3D).png", fig)
    fig
end
##
let
    points_list = path_actin |> CSV.File |> DataFrame |> split_dataframe_per_probe
    points_list = pointcloud_from_dataframe.(points_list)
    points_dict = Dict(zip(["actin", "desmin"], points_list))

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
    save("./Supplementary fig 1-3. Denoising results for actin-desmin 600nm (overlay).png", fig)
    fig
end
##
let
    points_list = path_actin |> CSV.File |> DataFrame |> split_dataframe_per_probe
    points_list = pointcloud_from_dataframe.(points_list)
    points_dict = Dict(zip(["actin", "desmin"], points_list))

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
    save("./Supplementary fig 1-4. Denoising results for actin-desmin 600nm (3D).png", fig)
    fig
end