##
using Revise
using GLMakie
import Pkg
Pkg.activate(".")
using Images
using FileIO

function preprocess_image(img_path)
    img = load(img_path)
    return @. Float32(Images.Gray(img))
end
##
desmin, actin = map(preprocess_image, ["./Data/SR2_desmin_image.jpg", "./Data/SR3_actin_image.jpg"])
##
let
    fig = Figure(resolution = (1500, 800))
    ax = GLMakie.Axis(fig[1, 1])
    image!(ax, actin)
    fig
end
desmin = desmin[end-2987+1:end, end-3004+1:end]
##
let
    desmin_vec, actin_vec = map(arr -> reshape(arr, length(arr)), [desmin, actin]) 
    fig = Figure(resolution = (1500, 800))
    ax = GLMakie.Axis(fig[1, 1])
    scatter!(ax, desmin_vec, actin_vec)
    fig
end