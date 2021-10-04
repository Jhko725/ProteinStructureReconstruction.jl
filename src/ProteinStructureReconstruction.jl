module ProteinStructureReconstruction
import Statistics: mean
using Images

export make_overlay, equalize_zstack
# Write your package code here.

function make_overlay(zstack_dataset::AbstractArray; dims = 0)
    overlay = zstack_dataset |> x -> reinterpret(N0f16, x) |> x ->  mean(x, dims = dims) |> x -> dropdims(x, dims = dims)
    return overlay
end

linearmap(img::AbstractArray{T}) where {T} = scaleminmax(T, extrema(img)...).(img)

function equalize_zstack(zstack::AbstractArray)
    out = mapslices(linearmap, zstack, dims = [1, 2])
    return out
end

end
