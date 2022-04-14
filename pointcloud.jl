module Pointcloud

using StaticArrays
import Makie

export PointCloud, to_abstractmatrix, rotate, coords

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

function rotate(point::SVector{2}, θ)
    s, c = sincos(θ)
    R = @SMatrix [c -s; s c]
    return R*point
end

function rotate(point::SVector{3}, yaw, pitch, roll)
    sα, cα = sincos(yaw)
    sβ, cβ = sincos(pitch)
    sγ, cγ = snicos(roll)
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