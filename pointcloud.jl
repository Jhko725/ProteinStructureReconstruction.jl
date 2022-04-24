module Pointcloud

using StaticArrays
import Makie

export PointCloud, to_abstractmatrix, rotate, coords, subset_region, projection

struct PointCloud{N, T<:AbstractFloat}
    coords::Vector{SVector{N, T}}
end

coords(ptc::PointCloud) = ptc.coords

Base.length(ptc::PointCloud) = ptc |> coords |> Base.length

function Makie.convert_arguments(P::Makie.PointBased, ptc::PointCloud{N, T}) where {N, T}
    if N<=3
        return Makie.convert_arguments(P, coords(ptc))
    else
        throw(DomainError(ptc, "only point clouds of dimension < 4 can directly be plotted"))
    end
end

to_abstractmatrix(x::Vector{SVector{N, T}}) where {N, T} = reinterpret(reshape, T, x)

to_abstractmatrix(x::PointCloud) = x |> coords |> to_abstractmatrix

function subset_region(pts::Vector{SVector{N, T}}, selector::Function) where {N, T}
    is_inside = map(selector, pts)
    return pts[is_inside]
end

function subset_region(pts::Vector{SVector{N, T}}, xlims, ylims) where {N, T}
    _selector = pt -> 
    (xlims[1] <= pt[1] <= xlims[2]) &&
    (ylims[1] <= pt[2] <= ylims[2])
    
    return subset_region(pts, _selector)
end

function subset_region(pts::Vector{SVector{3, T}}, xlims, ylims, zlims) where {N, T}
    _selector = pt -> 
    (xlims[1] <= pt[1] <= xlims[2]) &&
    (ylims[1] <= pt[2] <= ylims[2]) && 
    (zlims[1] <= pt[3] <= zlims[2])
    
    return subset_region(pts, _selector)
end

function projection(point::SVector{N, T}, projection_axis::Int) where {N, T}
    inds = deleteat!(collect(1:N), projection_axis)
    return SVector{N-1}(point[inds])
end

function projection(points::Vector{SVector{N, T}}, projection_axis) where {N, T}
    return map(p -> projection(p, projection_axis), points)
end

projection(points::PointCloud, projection_axis) = projection(coords(points), projection_axis) |> PointCloud

function rotate(point::SVector{2}, θ)
    s, c = sincos(θ)
    R = @SMatrix [c -s; s c]
    return R*point
end

function rotate(point::SVector{3}, yaw, pitch, roll)
    sα, cα = sincos(yaw)
    sβ, cβ = sincos(pitch)
    sγ, cγ = sincos(roll)
    R = @SMatrix [cα*cβ cα*sβ*sγ-sα*cγ cα*sβ*cγ+sα*sγ;
    sα*cβ sα*sβ*sγ+cα*cγ sα*sβ*cγ-cα*sγ;
    -sβ cβ*sγ cβ*cγ]
    return R*point
end

function rotate(points::Vector{SVector{N, T}}, args...) where {N, T}
    return map(p -> rotate(p, args...), points)
end

function rotate(points::PointCloud, args...)::PointCloud
    points = rotate(coords(points), args...)
    return PointCloud(points)
end



end