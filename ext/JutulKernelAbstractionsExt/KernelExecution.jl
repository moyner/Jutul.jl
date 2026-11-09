module KernelExecutionKernels

using Jutul
using KernelAbstractions
using LinearAlgebra, SparseArrays, StaticArrays
import Adapt, ForwardDiff
import Jutul: factorize_linear_system, refactorize_linear_system!, prepare_host_transfer!, backend_copyto!, transfer_csr_to_backend
import Jutul.KernelExecution: HostBackendFactorization, adjoint_block_order_kernel!, backend_copyto!, factorize_linear_system, host_backend_factorization, ka_csr_mul_kernel!, minbatch, nthreads, prepare_host_transfer!, refactorize_linear_system!, secondary_variable_update_kernel!, synchronize, threaded_loop_kernel, transfer, transfer_csr_to_backend
using Jutul: StaticSparsityMatrixCSR, KAPreconditioners

@kernel function ka_threaded_loop_kernel(f)
    index = @index(Global)
    f(index)
end

@kernel function ka_secondary_variable_update_kernel!(
        dest, var, model,
        @Const(dependencies)
    )
    i = @index(Global)
    Jutul.update_secondary_variable!(dest, var, model, dependencies, i:i)
end

@kernel function ka_adjoint_block_order_kernel!(
        destination, source, n, bz,
        to_canonical
    )
    index = @index(Global)
    if index <= n * bz
        block = (index - 1) ÷ n + 1
        entity = (index - 1) % n + 1
        block_major = (entity - 1) * bz + block
        if to_canonical
            @inbounds destination[index] = source[block_major]
        else
            @inbounds destination[block_major] = source[index]
        end
    end
end

@kernel function ka_ka_csr_mul_kernel!(y, nzval, colval, rowptr, x, alpha, beta)
    row = @index(Global)
    value_row = zero(eltype(y))
    @inbounds for pos in rowptr[row]:(rowptr[row + 1] - 1)
        value_row += nzval[pos] * x[colval[pos]]
    end
    if iszero(beta)
        @inbounds y[row] = alpha * value_row
    else
        @inbounds y[row] = alpha * value_row + beta * y[row]
    end
end

function prepare_host_transfer!(
        backend::KernelAbstractions.Backend,
        value::Array
    )
    KernelAbstractions.pagelock!(backend, value)
    return value
end

function backend_copyto!(destination::AbstractArray, source::AbstractArray)
    if length(destination) != length(source)
        throw(
            DimensionMismatch(
                "backend copy requires equal lengths, got $(length(destination)) and $(length(source))"
            )
        )
    end
    if destination === source
        return destination
    end
    if isempty(destination)
        return destination
    end
    backend = KernelAbstractions.get_backend(destination)
    source_backend = KernelAbstractions.get_backend(source)
    backend_is_cpu = backend isa KernelAbstractions.CPU
    source_backend_is_cpu = source_backend isa KernelAbstractions.CPU
    same_eltype = eltype(destination) == eltype(source)
    if backend_is_cpu && !source_backend_is_cpu
        has_native_transfer = applicable(KernelAbstractions.copyto!, source_backend, destination, source)
        if same_eltype && has_native_transfer
            # Repeated host mirrors are pinned during backend transfer. Queue
            # the device-to-host copy directly into that storage instead of
            # allocating a temporary host array for every state field.
            KernelAbstractions.copyto!(source_backend, destination, source)
        else
            # Avoid scalar iteration for backends without a direct copy hook.
            copyto!(destination, Adapt.adapt(Array, source))
        end
    elseif !backend_is_cpu && source_backend_is_cpu && !same_eltype
        converted = convert.(eltype(destination), source)
        KernelAbstractions.copyto!(backend, destination, converted)
        # The backend copy may be queued asynchronously. Keep the temporary
        # conversion buffer alive until that copy has completed.
        KernelAbstractions.synchronize(backend)
    elseif applicable(KernelAbstractions.copyto!, backend, destination, source)
        # Host-evaluated submodels own their source buffers for the duration of
        # the simulation. Queue their copies on the backend so a structured
        # state transfer requires one synchronization instead of one per field.
        KernelAbstractions.copyto!(backend, destination, source)
    else
        copyto!(destination, source)
    end
    return destination
end

function ka_csr_mul_kernel! end

function LinearAlgebra.mul!(
        y::AbstractVector,
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, B},
        x::AbstractVector, alpha::Number, beta::Number
    ) where {
        Tv, Ti <: Integer, V, I, R, B <: KernelAbstractions.Backend,
    }
    length(y) == size(A, 1) || throw(DimensionMismatch())
    length(x) == size(A, 2) || throw(DimensionMismatch())
    if A.use_vendor_linalg && KAPreconditioners.vendor_spmv!(
            KAPreconditioners.logical_backend_buffer(y), A,
            KAPreconditioners.logical_backend_buffer(x), alpha, beta
        )
        return y
    end
    scalar_type = KAPreconditioners.matrix_scalar_type(eltype(y))
    alpha = convert(scalar_type, alpha)
    beta = convert(scalar_type, beta)
    kernel! = ka_csr_mul_kernel!(A.backend)
    event = kernel!(y, A.nzval, A.colval, A.rowptr, x, alpha, beta; ndrange = size(A, 1))
    if !isnothing(event)
        wait(event)
    end
    return y
end

function LinearAlgebra.mul!(
        y::AbstractVector,
        A::StaticSparsityMatrixCSR{Tv, Ti, V, I, R, B},
        x::AbstractVector, alpha::Number, beta::Number
    ) where {
        Tv, Ti <: Integer, V <: Vector, I <: Vector, R <: Vector,
        B <: KernelAbstractions.CPU,
    }
    length(y) == size(A, 1) || throw(DimensionMismatch())
    length(x) == size(A, 2) || throw(DimensionMismatch())
    scalar_type = KAPreconditioners.matrix_scalar_type(eltype(y))
    alpha = convert(scalar_type, alpha)
    beta = convert(scalar_type, beta)
    KAPreconditioners.foreach_cpu_row(size(A, 1), A.minbatch) do row
        value_row = KAPreconditioners.csr_row_product(A, x, row, eltype(y))
        if iszero(beta)
            @inbounds y[row] = alpha * value_row
        else
            @inbounds y[row] = alpha * value_row + beta * y[row]
        end
    end
    return y
end

function factorize_linear_system(
        constructor,
        matrix::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B,
        }
    ) where {
        Tv, Ti <: Integer, V, I, R, B <: KernelAbstractions.Backend,
    }
    return host_backend_factorization(matrix)
end

function factorize_linear_system(
        ::typeof(lu),
        matrix::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B,
        }
    ) where {
        Tv, Ti <: Integer, V, I, R, B <: KernelAbstractions.Backend,
    }
    return KAPreconditioners.setup_sparse_lu(matrix)
end

function refactorize_linear_system!(
        update!,
        factorization::HostBackendFactorization,
        matrix::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B,
        }
    ) where {
        Tv, Ti <: Integer, V, I, R, B <: KernelAbstractions.Backend,
    }
    factorization.factorization = lu(KAPreconditioners.sparse_matrix(matrix))
    return factorization
end

function transfer_csr_to_backend(
        reference::StaticSparsityMatrixCSR{
            Tv, Ti, V, I, R, B,
        },
        matrix::StaticSparsityMatrixCSR
    ) where {
        Tv, Ti <: Integer, V, I, R, B <: KernelAbstractions.Backend,
    }
    values = similar(reference.nzval, eltype(matrix.nzval), length(matrix.nzval))
    columns = similar(reference.colval, eltype(matrix.colval), length(matrix.colval))
    rows = similar(reference.rowptr, eltype(matrix.rowptr), length(matrix.rowptr))
    copyto!(values, matrix.nzval)
    copyto!(columns, matrix.colval)
    copyto!(rows, matrix.rowptr)
    return StaticSparsityMatrixCSR(
        values, columns, rows, size(matrix, 1), size(matrix, 2),
        reference.backend; nthreads = reference.nthreads,
        minbatch = reference.minbatch, thread_type = reference.thread_type,
        use_vendor_linalg = reference.use_vendor_linalg
    )
end

threaded_loop_kernel(backend::KernelAbstractions.Backend, args...) = ka_threaded_loop_kernel(backend, args...)
secondary_variable_update_kernel!(backend::KernelAbstractions.Backend, args...) = ka_secondary_variable_update_kernel!(backend, args...)
adjoint_block_order_kernel!(backend::KernelAbstractions.Backend, args...) = ka_adjoint_block_order_kernel!(backend, args...)
ka_csr_mul_kernel!(backend::KernelAbstractions.Backend, args...) = ka_ka_csr_mul_kernel!(backend, args...)

end
