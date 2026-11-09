# Host algebra does not need KernelAbstractions. Its extension implements these
# hooks for accelerator arrays and for explicitly supplied KA backends.
struct CPUBackend end

is_cpu_backend(backend) = false
is_cpu_backend(::CPUBackend) = true

array_backend(array::Array) = CPUBackend()
array_backend(array::BitArray) = CPUBackend()
array_backend(array::Union{SubArray, Base.ReshapedArray, Base.ReinterpretArray}) =
    array_backend(parent(array))

backend_allocate(::CPUBackend, ::Type{T}, dims::Integer...) where {T} =
    Array{T}(undef, dims...)
backend_zeros(::CPUBackend, ::Type{T}, dims::Integer...) where {T} =
    zeros(T, dims...)
synchronize_backend(::CPUBackend) = nothing

function backend_allocate(backend, ::Type{T}, dims::Integer...) where {T}
    throw(ArgumentError("Load KernelAbstractions with `using KernelAbstractions` to use this backend."))
end

# A per-index operation can also run on ordinary Julia arrays. The extension
# wraps the same operation in a KA kernel for accelerator backends.
struct CPUKernel{F}
    operation::F
    minbatch::Int
end

function (kernel::CPUKernel)(args...; ndrange::Integer)
    foreach_cpu_row(Int(ndrange), kernel.minbatch) do index
        kernel.operation(index, args...)
    end
    return nothing
end
