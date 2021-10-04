##
using Revise
using GLMakie
using Pkg
Pkg.activate(".")
using ProteinStructureReconstruction
using HDF5
using Images
import Statistics: median, mean
import GeometryBasics: Point2f0
##
filepath = "./Data/Fed_X63_Z3_SIM.h5"
h5file = h5open(filepath, "r")
##
# TODO: don't load everything in memory at once -  just draw overlay first, then 
##
x_range, y_range = 2000:2300, 1600:1900;
img_data = h5file["Data"][x_range, y_range, :, :] |> x -> reinterpret(N0f16, x);
channels = read(h5file["Data"]["channels"])
##

##
overlay = make_overlay(read(h5file["Data"]), dims = 3);
##

##
let
    x1, x2 = first(x_range), last(x_range)
    y1, y2 = first(y_range), last(y_range)
    rect = Point2f0[(x1, y1), (x1, y2), (x2, y2), (x2, y1)]
    fig = Figure(resolution = (1500, 800))
    axes = [GLMakie.Axis(fig[1, i], title = "$(titlecase(c)) overlay", titlesize = 20.0f0) for (i, c) in enumerate(channels)]
  

    for (i, ax) in enumerate(axes)
        image!(ax, overlay[:, :, i])
        poly!(ax, rect, color = nothing, strokecolor = :red, strokewidth = 2.0)
    end
    fig
end
##
let
    fig = Figure(resolution = (1500, 800))
    axes = [GLMakie.Axis(fig[1, i], title = "$(titlecase(c)) overlay", titlesize = 20.0f0) for (i, c) in enumerate(channels)]
    cmaps = [:Greens_9, :Reds_9]
    for (i, ax) in enumerate(axes)
        image!(ax, overlay[x_range, y_range, i], colormap = cmaps[i])
    end
    fig
end
##
size(img_data)
##
let
    actin_vec = reshape(overlay[x_range, y_range, 1], length(overlay[x_range, y_range, 1]))
    desmin_vec = reshape(overlay[x_range, y_range, 2], length(overlay[x_range, y_range, 2]))

    fig = Figure(resolution = (800, 800))
    ax = GLMakie.Axis(fig[1, 1], xlabel = "$(titlecase(channels[1])) channel intensity", ylabel = "$(titlecase(channels[2])) channel intensity", title = "Intensity correlation for overlay (raw)")
    scatter!(ax, actin_vec, desmin_vec)
    xlims!(0.0, 0.5)
    ylims!(0.0, 0.5)
    fig
end

##
size(h5file["Data"])
##


##
plot_intensity_histogram!(axis, intensities::AbstractArray; hist_kwargs...) = hist!(axis, vec(intensities), normalization = :probability; hist_kwargs...)

plot_intensity_histogram!(axis, intensities::AbstractArray, rejection_intensity; hist_kwargs...) = plot_intensity_histogram!(axis, intensities[intensities .> rejection_intensity]; hist_kwargs...) 

function plot_intensity_histogram!(axis, intensities::AbstractVector{T}, args...; hist_kwargs...) where {T<:AbstractArray}
    num_plots = length(intensities)
    for (i, intensity) in enumerate(intensities)
        plot_intensity_histogram!(axis, intensity, args...; offset = -i/num_plots, hist_kwargs...)
    end
end
##
let 
    img_eq = equalize_zstack(img_data)
    img = img_eq
    fig = Figure(resolution = (1100, 700))
    Δz = 7
    zslice_nbs = 1:Δz:size(img)[3]
    num_slices = length(zslice_nbs)
    yticks = (-collect(1:num_slices)/num_slices, ["Slice #$nb" for nb in zslice_nbs])
    axes = [GLMakie.Axis(fig[1, i], title = "Intensity histograms for $(protein) z slices", yticks = yticks, xlabel = "Intensity") for (i, protein) in enumerate(channels)]
    
    for j in 1:size(img)[4]
        imgs = [img[:, :, i, j] for i in zslice_nbs]
        plot_intensity_histogram!(axes[j], imgs, zero(eltype(img)), bins = collect(0.0:1.0/2^8:1.0), color = (:slategray, 0.6))
    end
    
    fig
end
##
let
   img_eq = equalize_zstack(img_data)
   fig = Figure()
   ax = GLMakie.Axis(fig[1, 1])
   plot_intensity_histogram!(ax, img_eq, bins = collect(0.0:1.0/2^8:1.0))
   
   fig
end
##
let 
    img_eq = equalize_zstack(img_data)
    overlay_eq = mean(img_eq, dims = 3) |> x -> dropdims(x, dims = 3)

    fig = Figure(resolution = (1500, 800))
    axes = [GLMakie.Axis(fig[1, i], title = "$(titlecase(c)) overlay (intensity normalized)", titlesize = 20.0f0) for (i, c) in enumerate(channels)]
    cmaps = [:Greens_9, :Reds_9]
    for (i, ax) in enumerate(axes)
        image!(ax, overlay_eq[:, :, i], colormap = cmaps[i])
    end
    fig

end
##
let
    img_eq = equalize_zstack(img_data)
    overlay_eq = mean(img_eq, dims = 3) |> x -> dropdims(x, dims = 3)
    actin_vec = reshape(overlay_eq[:, :, 1], length(overlay_eq[:, :, 1]))
    desmin_vec = reshape(overlay_eq[:, :, 2], length(overlay_eq[:, :, 2]))

    fig = Figure(resolution = (800, 800))
    ax = GLMakie.Axis(fig[1, 1], xlabel = "$(titlecase(channels[1])) channel intensity", ylabel = "$(titlecase(channels[2])) channel intensity", title = "Intensity correlation for overlay (normalized)")
    scatter!(ax, actin_vec, desmin_vec)
    xlims!(0.0, 1.0)
    ylims!(0.0, 1.0)
    fig
end

##
import ImageBinarization: binarize, Otsu, MinimumError
img_eq = equalize_zstack(img_data)
mask = mapslices(x -> binarize(x, MinimumError()), img_eq, dims = [1, 2, 3])

let
    z = 40
    fig = Figure()
    axes = [GLMakie.Axis(fig[1, 1]), GLMakie.Axis(fig[1, 2])]
    image!(axes[1], img_eq[:, :, z, 1])
    image!(axes[2], mask[:, :, z, 1])
    fig
end
##


img_eq = equalize_zstack(img_data);
##
let 
    img = img_eq.*mask
    fig = Figure(resolution = (1200, 800))
    axes = [Axis3(fig[1, i], title = "$(titlecase(c))(Raw)") for (i, c) in enumerate(channels)]
    cmaps = [:Greens_3, :Reds_3]
    for (i, ax) in enumerate(axes)
        @views contour!(ax, img[:, :, :, i], levels = 4, alpha = 0.6, colormap = cmaps[i], tranparancy = true)
    end
    
    fig
end
##
let 
    img = img_eq.*mask
    fig = Figure(resolution = (1200, 800))
    axes = [Axis3(fig[1, i], title = "$(titlecase(c))(Normalized)") for (i, c) in enumerate(channels)]
    cmaps = [:Greens_3, :Reds_3]
    for (i, ax) in enumerate(axes)
        @views volume!(ax, img[100:300, 100:300, :, i], algorithm = :iso, colormap = cmaps[i], isoval = 0.002, isorange = 0.2)
    end
    
    fig
end
##

##
read(h5file["Data"]["channels"])
##
img_eq = equalize_zstack(img_data);
##
let 
    fig = Figure(resolution = (1200, 800))
    axes = [Axis3(fig[1, i], title = "$(titlecase(c))(Normalized)") for (i, c) in enumerate(channels)]
    cmaps = [:Greens_3, :Reds_3]
    for (i, ax) in enumerate(axes)
        contour!(ax, img_eq[1:100, 1:100, :, i], levels = 4, alpha = 0.4, colormap = cmaps[i])
    end
    
    fig
end
##
let 
    fig = Figure(resolution = (1200, 800))
    axes = [Axis3(fig[1, 1], title = "Actin (Raw)"), Axis3(fig[1, 2], title = "Actin (Equalized)"), Axis3(fig[2, 1], title = "Desmin (Raw)"), Axis3(fig[2, 2], title = "Desmin (Equalized)")]
    volumes = [img_data, img_eq]
    cmaps = [:Greens_3, :Reds_3]
    
    contour!(axes[1], interpolate_zstack(volumes[1][1:100, 1:100, :, 1], pixel_sizes), levels = 4, alpha = 0.6, colormap = cmaps[1])
    contour!(axes[2], interpolate_zstack(volumes[2][1:100, 1:100, :, 1], pixel_sizes), levels = 4, alpha = 0.6, colormap = cmaps[1])
    contour!(axes[3], interpolate_zstack(volumes[1][1:100, 1:100, :, 2], pixel_sizes), levels = 4, alpha = 0.6, colormap = cmaps[2])
    contour!(axes[4], interpolate_zstack(volumes[2][1:100, 1:100, :, 2], pixel_sizes), levels = 4, alpha = 0.6, colormap = cmaps[2])
    
    fig
end
##
let 
    img = img_data
    flatten(A::AbstractArray) = reshape(A, length(A))
    actin_vec = flatten(img[:, :, :, 1])
    desmin_vec = flatten(img[:, :, :, 2])

    fig = Figure(resolution = (800, 800))
    ax = GLMakie.Axis(fig[1, 1], xlabel = "$(titlecase(channels[1])) channel intensity", ylabel = "$(titlecase(channels[2])) channel intensity", title = "Intensity correlation for overlay (normalized)")
    scatter!(ax, actin_vec, desmin_vec)
    xlims!(0.0, 0.5)
    ylims!(0.0, 0.5)
    fig
    
end
##
import FFTW: fft, fftshift
test = fftshift(fft(Float32.(img_eq[:, :, 20, 1])));
let
    fig = Figure(resolution = (1500, 800))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Actin 2D Fourier log intensity", titlesize = 20.0f0)
    ax2 = GLMakie.Axis(fig[1, 2], title = "Desmin 2D Fourier log intensity", titlesize = 20.0f0)
    axes = [ax1, ax2]
    for (i, ax) in enumerate(axes)
        img_FFT = Float32.(img_eq[:, :, 20, i]) |> fft |> fftshift
        heatmap!(ax, log.(abs.(img_FFT.*img_FFT)).+1)
    end
    fig
end

##
using Interpolations
function interpolate_zstack(zstack::AbstractArray{T, 3}, pixel_sizes) where {T}
    scale_x, scale_y, scale_z = pixel_sizes/minimum(pixel_sizes)
    nx, ny, nz = size(zstack)
    grids = ((0:nx-1)*scale_x, (0:ny-1)*scale_y, (0:nz-1)*scale_z)
    algorithm = BSpline(Linear())
    itp = interpolate(zstack, algorithm)
    sitp = scale(itp, grids...)
    return sitp(0:(nx-1)*scale_x, 0:(ny-1)*scale_y, 0:(nz-1)*scale_z)
end

##
pixel_sizes = read(h5file["Data"]["pixel_sizes"])[end:-1:1]
##
let 
    fig = Figure(resolution = (1200, 800))
    axes = [Axis3(fig[1, i], title = "$(titlecase(c))(Raw)") for (i, c) in enumerate(channels)]
    cmaps = [:Greens_3, :Reds_3]
    for (i, ax) in enumerate(axes)
        @views vol_interp = interpolate_zstack(img_eq[1:100, 1:100, :, i], pixel_sizes)
        contour!(ax, vol_interp, levels = 4, alpha = 0.6, colormap = cmaps[i])
    end
    
    fig
end
##
let 
    #mask = mapslices(x -> binarize(x, MinimumError()), img_data, dims = [1, 2, 3])
    img = img_data
    #img = img_eq.*mask
    fig = Figure(resolution = (1200, 800))
    axes = [Axis3(fig[1, i], title = "$(titlecase(c))(Raw)") for (i, c) in enumerate(channels)]
    cmaps = [:Greens_3, :Reds_3]
    for (i, ax) in enumerate(axes)
        @views vol_interp = interpolate_zstack(img[1:100, 1:100, :, i], pixel_sizes)
        volume!(ax, vol_interp, colormap = cmaps[i], algorithm = :iso, isoval = 0.0001, isorange = 0.2)
    end
    
    fig
end
##
let 
desmin_interp = interpolate_zstack(img_eq[:, :, :, 1], pixel_sizes)
actin_interp = interpolate_zstack(img_eq[:, :, :, 2], pixel_sizes)

using NPZ
npzwrite("./Data/interp.npz", Dict("desmin" => desmin_interp, "actin" => actin_interp))
end