##
using Revise
using GLMakie
using Pkg
Pkg.activate(".")
using ProteinStructureReconstruction
using HDF5
using Images
##
filepath = "./Data/Fed_X63_Z3_SIM.h5"
h5file = h5open(filepath, "r")

x_range, y_range = 2000:2300, 1600:1900;
img_data = h5file["Data"][x_range, y_range, :, :] |> x -> reinterpret(N0f16, x);
channels = read(h5file["Data"]["channels"])
##
import ImageBinarization: binarize, MinimumError
img_eq = equalize_zstack(img_data);
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

pixel_sizes = read(h5file["Data"]["pixel_sizes"])[end:-1:1]
img_interp = mapslices(x -> interpolate_zstack(x, pixel_sizes), img_eq, dims = [1, 2, 3]);
size(img_interp)
##
img_mask = mapslices(x -> binarize(x, MinimumError()), img_interp, dims = [1, 2, 3]) |> x -> float32.(x);
##
using StaticArrays
function pointcloud_from_image(image::AbstractArray{T, N}) where {T, N}
    coords = map(findall(x -> x > zero(T), image)) do ind
        ind |> Tuple |> SVector{N, T}
    end
    return coords
end

img_coords = [pointcloud_from_image(vol) for vol in eachslice(img_mask, dims = 4)];
size(img_coords[1])
##
using NearestNeighbors
trees = map(KDTree, img_coords)

##
# for each point in desmin, find closest distance to actin set
_, desmin_point_actin_set_distance = nn(trees[1], img_coords[2])
##
maximum(desmin_point_actin_set_distance)
##
let   
    fig = Figure()
    ax = GLMakie.Axis(fig[1, 1])
    hist!(ax, desmin_point_actin_set_distance*pixel_sizes[1], bins = 30)
    fig
end
##
import StatsBase: fit, Histogram
bin_size = 0.8
distances = collect(0.0:bin_size:maximum(desmin_point_actin_set_distance)+bin_size)
h = fit(Histogram, desmin_point_actin_set_distance, distances, closed=:left)
##
let 
    fig = Figure(resolution = (1200, 600))
    ax1 = GLMakie.Axis(fig[1, 1], title = "Point(desmin)-to-Set(actin) correlation function (Δr = 0.8)", xlabel = "Distance (pixels)", ylabel = "Correlation (a.u.)")
    ax2 = GLMakie.Axis(fig[1, 2], yscale = log10, title = "Point(desmin)-to-Set(actin) correlation function (Δr = 0.8)", xlabel = "Distance (pixels)", ylabel = "Correlation (a.u.)")
    scatterlines!(ax1, distances[1:end-1], h.weights)
    scatterlines!(ax2, distances[1:end-1], h.weights)
    fig
end
##
pixel_sizes