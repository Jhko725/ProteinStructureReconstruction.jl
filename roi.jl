abstract type AbstractROI{D,T<:AbstractFloat} end

struct SimpleROI{D,T} <: AbstractROI{D,T}
    lims::SVector{D,Tuple{T,T}}

    function SimpleROI{D,T}(lims) where {D,T}
        is_ordered = all([a < b for (a, b) in lims])
        is_ordered ? new(lims) : error("Each axis limit in lims must be a ordered tuple")
    end
end

function SimpleROI(xlims::Tuple{T,T}, ylims::Tuple{T,T}) where {T}
    lims = @SVector[xlims, ylims]
    SimpleROI{2,T}(lims)
end

function SimpleROI(xlims::Tuple{T,T}, ylims::Tuple{T,T}, zlims::Tuple{T,T}) where {T}
    lims = @SVector[xlims, ylims, zlims]
    SimpleROI{3,T}(lims)
end

function Base.length(roi::SimpleROI)
    return reduce(*, [b - a for (a, b) in roi.lims])
end

function Base.in(p::AbstractVector, roi::SimpleROI)::Bool
    return all([a <= pᵢ <= b for (pᵢ, (a, b)) in zip(p, roi.lims)])
end

function erode(x::SimpleROI{D,T}, r::T) where {D,T}
    new_lims = SVector{D,Tuple{T,T}}([(a + r, b - r) for (a, b) in x.lims])
    # Might need to catch error when eroding too much
    return SimpleROI{D,T}(new_lims)
end