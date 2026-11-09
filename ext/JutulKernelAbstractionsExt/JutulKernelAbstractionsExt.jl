module JutulKernelAbstractionsExt

using Jutul
import KernelAbstractions as KA
import Jutul.KAPreconditioners as KP
import Jutul.KernelExecution as KE

KP.is_cpu_backend(::KA.CPU) = true
KP.array_backend(array::AbstractArray) = KA.get_backend(array)
KP.backend_allocate(backend::KA.Backend, ::Type{T}, dims::Integer...) where {T} =
    KA.allocate(backend, T, dims...)
KP.backend_zeros(backend::KA.Backend, ::Type{T}, dims::Integer...) where {T} =
    KA.zeros(backend, T, dims...)
function KP.synchronize_backend(backend::KA.Backend)
    if applicable(KA.synchronize, backend)
        KA.synchronize(backend)
    end
    return nothing
end

KE.is_kernel_backend(::KA.Backend) = true
KA.get_backend(ctx::Jutul.KernelAbstractionsContext) = ctx.backend
KA.get_backend(buffer::KP.BackendBufferPrefix) =
    KA.get_backend(KP.backend_view(buffer))

# KA CPU descriptors share the existing host implementations.
KP.fill_backend!(x::Array, value, ::KA.CPU, block_size) = fill!(x, value)
KP.axpy!(x::Vector, y::Vector, alpha, ::KA.CPU, block_size) =
    KP.axpy!(x, y, alpha, KP.CPUBackend(), block_size)
KP.restrict!(bc::Vector, Pt::KP.TransposeMap, P::KP.Prolongation, r::Vector, ::KA.CPU, block_size) =
    KP.restrict!(bc, Pt, P, r, KP.CPUBackend(), block_size)
KP.prolong!(x::Vector, P::KP.Prolongation, xc::Vector, ::KA.CPU, block_size) =
    KP.prolong!(x, P, xc, KP.CPUBackend(), block_size)
KP.prolong_to!(dst::Vector, src::Vector, P::KP.Prolongation, xc::Vector, ::KA.CPU, block_size) =
    KP.prolong_to!(dst, src, P, xc, KP.CPUBackend(), block_size)

include("KAPreconditioners.jl")
include("KernelExecution.jl")

end
